from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import unittest

from lfm.model.chip_acquisition import (
    AcquisitionGroupResult,
    ChipAcquisitionResult,
)
from lfm.model.chip_assembly import assemble_and_write_chip
from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    MixedPercentageNumberSplitConfig,
    NumberSplitConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitCounts,
    SplitPercentages,
)
from lfm.model.chip_labels import preflight_label
from lfm.model.chip_preflight import PreparedChipRequest
from lfm.model.chip_publication import (
    ChipPublicationError,
    build_dataset_manifest,
    publish_chip_pair,
    validate_dataset_publication,
    write_dataset_manifest,
)
from lfm.model.chip_reprojection import (
    ChipReprojectionResult,
    ReprojectedModality,
)
from lfm.model.chip_splits import SplitAssignment, SplitPlan, plan_splits
from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    ChipResult,
    GeographicAOI,
    SourceSelector,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig


HAS_RASTER_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "osgeo", "rasterio", "torch")
)


def simple_config(root: Path) -> ChipConfig:
    source = TileSourceConfig(
        "wac",
        root / "source",
        root / "source/index.gpkg",
        selection_mode="product_id",
    )
    return ChipConfig(
        output_root=root / "dataset",
        label_source=root / "source-labels",
        acquisition_groups=(
            AcquisitionGroupConfig(
                "coarse",
                TileConfig(root / "tiles", 5, (source,)),
            ),
        ),
        output_modalities=(
            OutputModalityConfig(
                "coarse",
                "wac",
                "wac",
                output_band_names=("VIS",),
            ),
        ),
        split_config=SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
    )


def geographic_grid() -> TargetGrid:
    return TargetGrid(
        crs_wkt="GEOGCRS[Moon]",
        transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        bounds=(0.0, 0.0, 3.0, 2.0),
        width=3,
        height=2,
    )


def failed_prepared(
    sample_id: str,
    split: str | None,
    *,
    group: str,
) -> PreparedChipRequest:
    request = ChipRequest(
        sample_id=sample_id,
        target_grid=geographic_grid(),
        geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 3.0),
        split_group_key=group,
    )
    assignment = SplitAssignment(
        sample_id=sample_id,
        split_group_key=group,
        assigned_split=split,
        source="automatic_percentage" if split is not None else "unassigned",
    )
    return PreparedChipRequest(
        request=request,
        assignment=assignment,
        preflight=ChipPreflight(
            status="failed" if split is not None else "skipped",
            assigned_split=split,
        ),
    )


class ChipManifestTestCase(unittest.TestCase):
    def test_manifest_serializes_mixed_and_number_split_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    MixedPercentageNumberSplitConfig(
                        fixed_counts=SplitCounts(test=1),
                        fixed_priority=("test",),
                        remaining_percentages=SplitPercentages(1.0, 0.0, 0.0),
                    ),
                    "test",
                    "mixed_percentage_number",
                ),
                (
                    NumberSplitConfig(
                        fixed_counts=SplitCounts(train=1),
                        fixed_priority=("train",),
                        remainder_split="test",
                    ),
                    "train",
                    "number",
                ),
            )
            for split_config, split, expected_type in cases:
                with self.subTest(split_type=expected_type):
                    config = replace(
                        simple_config(root),
                        split_config=split_config,
                    )
                    prepared = failed_prepared(
                        "M1_r0_c0",
                        split,
                        group="site-a",
                    )
                    result = ChipResult(
                        prepared.request,
                        "failed",
                        prepared.preflight,
                    )
                    plan = SplitPlan(assignments=(prepared.assignment,))

                    document = build_dataset_manifest(
                        (prepared,),
                        (result,),
                        plan,
                        config,
                    )

                    self.assertEqual(
                        document["split_policy"]["type"],
                        expected_type,
                    )
                    self.assertIsNotNone(
                        document["split_policy"]["requested_counts"]
                    )
                    self.assertEqual(
                        document["configuration"]["split_config"]["type"],
                        expected_type,
                    )

    def test_failed_samples_are_sorted_and_retained_in_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            second = failed_prepared("M2_r0_c0", "val", group="site-b")
            first = failed_prepared("M1_r0_c0", None, group="site-a")
            prepared = (second, first)
            results = (
                ChipResult(second.request, "failed", second.preflight),
                ChipResult(first.request, "skipped", first.preflight),
            )
            plan = SplitPlan(assignments=(second.assignment, first.assignment))

            document = build_dataset_manifest(prepared, results, plan, config)
            first_path = write_dataset_manifest(
                prepared,
                results,
                plan,
                config,
                output_path=root / "first.json",
            )
            second_path = write_dataset_manifest(
                tuple(reversed(prepared)),
                tuple(reversed(results)),
                plan,
                config,
                output_path=root / "second.json",
            )

            self.assertEqual(
                [item["sample_id"] for item in document["samples"]],
                ["M1_r0_c0", "M2_r0_c0"],
            )
            self.assertEqual(
                document["publication_counts"]["statuses"],
                {"failed": 1, "skipped": 1},
            )
            self.assertEqual(document["samples"][1]["chip_path"], None)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                document["split_policy"]["hash_algorithm"],
                "blake2b",
            )

    def test_manifest_is_accepted_as_a_prior_split_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            prepared = failed_prepared("M1_r0_c0", "val", group="site-a")
            result = ChipResult(prepared.request, "failed", prepared.preflight)
            plan = SplitPlan(assignments=(prepared.assignment,))
            path = write_dataset_manifest(
                (prepared,),
                (result,),
                plan,
                config,
            )
            locked_config = SimpleSplitConfig(
                SplitPercentages(1.0, 0.0, 0.0),
                prior_manifest_path=path,
            )
            new_request = ChipRequest(
                sample_id="M2_r0_c0",
                target_grid=geographic_grid(),
                geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 3.0),
                split_group_key="site-a",
            )

            locked = plan_splits((new_request,), locked_config)

            self.assertEqual(locked.assignments[0].assigned_split, "val")
            self.assertEqual(locked.assignments[0].source, "prior_manifest")

    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            first = failed_prepared("M1_r0_c0", "train", group="same-site")
            second = failed_prepared("M2_r0_c0", "test", group="same-site")
            results = (
                ChipResult(first.request, "failed", first.preflight),
                ChipResult(second.request, "failed", second.preflight),
            )
            plan = SplitPlan(assignments=(first.assignment, second.assignment))

            with self.assertRaisesRegex(ValueError, "crosses splits"):
                build_dataset_manifest((first, second), results, plan, config)

    def test_invalid_request_retains_explicit_split_intent_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            request = ChipRequest(
                sample_id="M1_r0_c0",
                target_grid=geographic_grid(),
                geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 3.0),
                split_group_key="site-a",
                assigned_split="train",
            )
            prepared = PreparedChipRequest(
                request=request,
                assignment=SplitAssignment(
                    request.sample_id,
                    request.split_group_key,
                    None,
                    "unassigned",
                ),
                preflight=ChipPreflight(status="failed", assigned_split="train"),
            )
            result = ChipResult(request, "failed", prepared.preflight)

            document = build_dataset_manifest(
                (prepared,),
                (result,),
                SplitPlan(assignments=()),
                config,
            )

            sample = document["samples"][0]
            self.assertEqual(sample["requested_split"], "train")
            self.assertIsNone(sample["assigned_split"])
            self.assertIsNone(sample["chip_path"])

    def test_failed_sample_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            prepared = failed_prepared("M1_r0_c0", "train", group="site-a")
            result = ChipResult(prepared.request, "failed", prepared.preflight)
            plan = SplitPlan(assignments=(prepared.assignment,))
            artifact = (
                config.output_root
                / "train/chips/M1_r0_c0_input_wac_chip.tif"
            )
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"unexpected")

            with self.assertRaisesRegex(ValueError, "membership mismatch"):
                build_dataset_manifest((prepared,), (result,), plan, config)


@unittest.skipUnless(HAS_RASTER_DEPS, "GDAL, NumPy, Rasterio, and Torch required")
class ChipPublicationRasterTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import osr

        self.np = np
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(4326)
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.grid = TargetGrid(
            crs_wkt=spatial_reference.ExportToWkt(),
            transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            bounds=(0.0, 0.0, 3.0, 2.0),
            width=3,
            height=2,
        )

    def run_training_loader(self, script, *arguments):
        repo_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(repo_root) + (
            ""
            if not existing_pythonpath
            else os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, *(str(item) for item in arguments)],
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "Training-package loader smoke test failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    def request(self, sample_id, split):
        return ChipRequest(
            sample_id=sample_id,
            target_grid=self.grid,
            geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 3.0),
            split_group_key=f"group-{sample_id}",
            assigned_split=split,
        )

    def written_chip(self, request, assignment, config):
        preflight = preflight_label(
            request,
            label_source=config.label_source,
            assigned_split=assignment.assigned_split,
        )
        prepared = PreparedChipRequest(request, assignment, preflight)
        selector = SourceSelector("coarse", "wac", request.sample_id.split("_")[0])
        group_result = AcquisitionGroupResult(
            sample_id=request.sample_id,
            acquisition_group="coarse",
            zoom_level=5,
            output_dir=config.intermediate_root / request.sample_id / "coarse",
            logical_aoi=request.geographic_aoi,
            query_parts=(request.geographic_aoi,),
            selectors=(selector,),
            status="complete",
        )
        acquisition = ChipAcquisitionResult(
            prepared_request=prepared,
            status="complete",
            group_results=(group_result,),
        )
        pixels = self.np.arange(6, dtype=self.np.float64).reshape(1, 2, 3) + 1
        modality = ReprojectedModality(
            acquisition_group="coarse",
            source_name="wac",
            alias="wac",
            resampling="bilinear",
            status="complete",
            target_grid=self.grid,
            band_names=("source_vis",),
            pixels=pixels,
            valid_mask=self.np.ones(pixels.shape, dtype=bool),
            nodata_values=(config.common_nodata,),
            zone_groups=(),
        )
        reprojection = ChipReprojectionResult(
            acquisition=acquisition,
            target_grid=self.grid,
            modalities=(modality,),
        )
        return prepared, assemble_and_write_chip(reprojection, config)

    def make_semantic_label(self, config, sample_id):
        config.label_source.mkdir(parents=True, exist_ok=True)
        path = config.label_source / f"{sample_id}_label.npy"
        self.np.save(path, self.np.arange(6, dtype=self.np.int16).reshape(2, 3))
        return path

    def make_instance_label(self, config, sample_id):
        config.label_source.mkdir(parents=True, exist_ok=True)
        path = config.label_source / f"{sample_id}_label.npz"
        mask = self.np.zeros((2, 3), dtype=self.np.int16)
        mask[0, 0] = 1
        self.np.savez(
            path,
            mask=mask,
            bboxes=self.np.asarray(((0.0, 0.0, 1.0, 1.0),)),
            num_craters=self.np.asarray(1),
            scientist_metadata=self.np.asarray((42, 84)),
        )
        return path

    def test_semantic_pairs_publish_and_load_from_every_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            requests = tuple(
                self.request(f"M{index}_r0_c0", split)
                for index, split in enumerate(("train", "val", "test"), start=1)
            )
            for request in requests:
                self.make_semantic_label(config, request.sample_id)
            plan = plan_splits(requests, config.split_config)
            prepared = []
            results = []
            source_bytes = {}
            for request in requests:
                assignment = plan.assignment_for(request.sample_id)
                prepared_request, written = self.written_chip(
                    request,
                    assignment,
                    config,
                )
                source_label = prepared_request.preflight.resolved_label_path
                source_bytes[request.sample_id] = source_label.read_bytes()
                prepared.append(prepared_request)
                results.append(publish_chip_pair(written, config))

            manifest_path = write_dataset_manifest(
                prepared,
                results,
                plan,
                config,
            )
            validation = validate_dataset_publication(
                prepared,
                results,
                plan,
                config,
                manifest_path=manifest_path,
            )

            self.assertEqual(validation.successful_count, 3)
            for result in results:
                self.assertEqual(
                    result.label_path.read_bytes(),
                    source_bytes[result.request.sample_id],
                )
                self.assertEqual(
                    result.effective_selectors[0].product_id,
                    result.request.sample_id.split("_")[0],
                )
            self.run_training_loader(
                """
import sys
from pathlib import Path
from lfm.all_models.sem_seg.data.semantic_dataset import SemanticSegmentationDataset

dataset_root = Path(sys.argv[1])
for split in ("train", "val", "test"):
    dataset = SemanticSegmentationDataset(
        dataset_root / split,
        target_size=(2, 3),
        spatial_transform="crop",
        scale_inputs=False,
        require_all_labels=True,
    )
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, 2, 3)
    assert tuple(sample["mask"].shape) == (2, 3)
""",
                config.output_root,
            )

    def test_instance_archive_is_preserved_and_loaded_without_data_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            request = self.request("M9_r0_c0", "test")
            source_label = self.make_instance_label(config, request.sample_id)
            plan = plan_splits((request,), config.split_config)
            prepared, written = self.written_chip(
                request,
                plan.assignments[0],
                config,
            )

            result = publish_chip_pair(written, config)
            self.assertEqual(result.label_path.read_bytes(), source_label.read_bytes())
            self.run_training_loader(
                """
import sys
from pathlib import Path
from lfm.all_models.inst_seg.data.instance_dataset import LunarInstanceMaskDataset

dataset = LunarInstanceMaskDataset(
    Path(sys.argv[1]),
    target_size=(2, 3),
    scale_inputs=False,
)
sample = dataset[0]
assert tuple(sample["image"].shape) == (1, 2, 3)
assert tuple(sample["mask"].shape) == (2, 3)
assert int(sample["num_craters"]) == 1
""",
                config.output_root / "test",
            )

    def test_second_publication_failure_rolls_back_first_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = simple_config(root)
            request = self.request("M9_r0_c0", "train")
            self.make_semantic_label(config, request.sample_id)
            plan = plan_splits((request,), config.split_config)
            _, written = self.written_chip(
                request,
                plan.assignments[0],
                config,
            )
            real_link = os.link
            calls = 0

            def fail_second_link(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected label publication failure")
                return real_link(source, destination)

            with mock.patch(
                "lfm.model.chip_publication.os.link",
                side_effect=fail_second_link,
            ):
                with self.assertRaises(ChipPublicationError) as context:
                    publish_chip_pair(written, config)

            self.assertEqual(context.exception.code, "pair_publication_failed")
            self.assertEqual(
                tuple((config.output_root / "train/chips").iterdir()),
                (),
            )
            self.assertEqual(
                tuple((config.output_root / "train/labels").iterdir()),
                (),
            )


if __name__ == "__main__":
    unittest.main()
