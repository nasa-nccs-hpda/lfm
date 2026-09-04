from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.chip_acquisition import (
    AcquisitionDiagnostic,
    AcquisitionGroupResult,
    ChipAcquisitionResult,
)
from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    MixedPercentageNumberSplitConfig,
    NoSplitConfig,
    NumberSplitConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitCounts,
    SplitPercentages,
)
from lfm.model.chip_creation import (
    ChipBatchResult,
    create_chip,
    create_chips,
    create_chips_from_reference_directory,
)
from lfm.model.chip_preflight import BatchPreflightResult, PreparedChipRequest
from lfm.model.lunar_crs import load_lunar_geographic_wkt
from lfm.model.chip_requests import materialize_requests
from lfm.model.chip_splits import SplitAssignment, SplitPlan, plan_splits
from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    ChipResult,
    GeographicAOI,
    LabelMismatchError,
    LabelValidationDiagnostic,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig
from lfm.model.tiling_results import TileCubeRecord


HAS_RASTER_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "osgeo")
)


class ChipCreationTestCase(unittest.TestCase):
    def grid(self):
        return TargetGrid(
            crs_wkt="GEOGCRS[Moon]",
            transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            bounds=(0.0, 0.0, 2.0, 2.0),
            width=2,
            height=2,
        )

    def request(self, index, *, split=None):
        return ChipRequest(
            sample_id=f"M{index}_r0_c0",
            target_grid=self.grid(),
            geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 2.0),
            split_group_key=f"site-{index}",
            assigned_split=split,
        )

    def group(self, root, name="coarse", zoom=5):
        source = TileSourceConfig(
            "static",
            root / name / "source",
            root / name / "index.gpkg",
        )
        return AcquisitionGroupConfig(
            name,
            TileConfig(root / "unused", zoom, (source,)),
        )

    def config(self, root, *, retention="on_failure", split_config=None, groups=None):
        groups = tuple(groups or (self.group(root),))
        return ChipConfig(
            output_root=root / "dataset",
            intermediate_root=root / "intermediate",
            label_source=root / "labels",
            acquisition_groups=groups,
            output_modalities=tuple(
                OutputModalityConfig(group.name, "static", group.name)
                for group in groups
            ),
            split_config=split_config
            or SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
            intermediate_retention=retention,
        )

    def prepared(self, request, *, status="passed", split="train", diagnostics=()):
        assigned = split if status != "skipped" else None
        return PreparedChipRequest(
            request=request,
            assignment=SplitAssignment(
                request.sample_id,
                request.split_group_key,
                assigned,
                "automatic_percentage" if assigned is not None else "unassigned",
            ),
            preflight=ChipPreflight(
                status=status,
                assigned_split=assigned,
                resolved_label_path=(
                    Path(f"/labels/{request.sample_id}_label.npy")
                    if status == "passed"
                    else None
                ),
                label_diagnostics=tuple(diagnostics),
            ),
        )

    def record(self, path):
        return TileCubeRecord(
            source_name="static",
            zone="1N",
            zoom_level=5,
            tile_x=1,
            tile_y=2,
            product_id=None,
            path=path,
            band_names=("band",),
            crs_wkt="PROJCRS[test]",
            nodata_values=(-32768.0,),
        )

    def acquisition(self, prepared, config, *, status="complete", records=()):
        group = config.acquisition_groups[0]
        group_result = AcquisitionGroupResult(
            sample_id=prepared.request.sample_id,
            acquisition_group=group.name,
            zoom_level=group.tile_config.zoom_level,
            output_dir=config.intermediate_root / prepared.request.sample_id / group.name,
            logical_aoi=prepared.request.geographic_aoi,
            query_parts=(prepared.request.geographic_aoi,),
            selectors=(),
            status=status,
            records=tuple(records),
            inventory_paths=tuple(record.path for record in records),
            diagnostics=(
                (
                    AcquisitionDiagnostic(
                        "missing_required_source",
                        "required source failed",
                        "error",
                        acquisition_group=group.name,
                        source_name="static",
                        zone="1N",
                        zoom_level=5,
                        tile_x=3,
                        tile_y=2,
                    ),
                )
                if status == "failed"
                else ()
            ),
            attempted_query_parts=(prepared.request.geographic_aoi,),
            failed_query_part=(
                prepared.request.geographic_aoi if status == "failed" else None
            ),
        )
        return ChipAcquisitionResult(
            prepared_request=prepared,
            status=status,
            group_results=(group_result,),
            diagnostics=group_result.diagnostics,
        )

    @mock.patch("lfm.model.chip_creation.publish_chip_pair")
    @mock.patch("lfm.model.chip_creation.assemble_and_write_chip")
    @mock.patch("lfm.model.chip_creation.reproject_acquisition")
    @mock.patch("lfm.model.chip_creation.acquire_prepared_request")
    def test_create_chip_runs_all_stages_and_cleans_only_its_sample(
        self,
        acquire,
        reproject,
        write,
        publish,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root, retention="never")
            prepared = self.prepared(self.request(1))
            sample_dir = config.intermediate_root / prepared.request.sample_id
            other_dir = config.intermediate_root / "other-sample"
            sample_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)
            (sample_dir / "cube.tif").write_bytes(b"cube")
            acquisition = self.acquisition(prepared, config)
            acquire.return_value = acquisition
            reproject.return_value = object()
            write.return_value = object()
            published = ChipResult(
                prepared.request,
                "success",
                prepared.preflight,
                chip_path=config.output_root / "train/chips/chip.tif",
                label_path=config.output_root / "train/labels/label.npy",
            )
            publish.return_value = published

            result = create_chip(prepared, config)

            self.assertEqual(result.status, "success")
            self.assertIsNotNone(result.elapsed_seconds)
            self.assertTrue(result.diagnostic_path.is_file())
            self.assertFalse(sample_dir.exists())
            self.assertTrue(other_dir.is_dir())
            reproject.assert_called_once_with(acquisition, config)
            publish.assert_called_once_with(write.return_value, config, overwrite=False)

    @mock.patch("lfm.model.chip_creation.acquire_prepared_request")
    def test_partial_acquisition_retains_records_inventory_and_unattempted_work(
        self,
        acquire,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = (self.group(root), self.group(root, "fine", 11))
            config = self.config(root, groups=groups)
            prepared = self.prepared(self.request(1))
            cube_path = (
                config.intermediate_root
                / prepared.request.sample_id
                / "coarse/partial.tif"
            )
            cube_path.parent.mkdir(parents=True)
            cube_path.write_bytes(b"partial")
            record = self.record(cube_path)
            partial = self.acquisition(
                prepared,
                config,
                status="failed",
                records=(record,),
            )
            first_part = GeographicAOI(2.0, 179.0, 0.0, 180.0)
            second_part = GeographicAOI(2.0, -180.0, 0.0, -179.0)
            group_result = replace(
                partial.group_results[0],
                logical_aoi=GeographicAOI(2.0, 179.0, 0.0, 181.0),
                query_parts=(first_part, second_part),
                attempted_query_parts=(first_part,),
                failed_query_part=first_part,
            )
            acquire.return_value = replace(
                partial,
                group_results=(group_result,),
            )

            result = create_chip(prepared, config)

            self.assertEqual(result.status, "partial")
            self.assertEqual(result.cube_records, (record,))
            self.assertTrue(cube_path.is_file())
            codes = {item.code for item in result.diagnostics}
            self.assertIn("later_tiles_not_attempted", codes)
            self.assertIn("unattempted_query_part", codes)
            self.assertIn("unattempted_acquisition_group", codes)
            document = json.loads(result.diagnostic_path.read_text())
            group_document = document["acquisition_groups"][0]
            self.assertEqual(group_document["completed_records"][0]["tile_x"], 1)
            self.assertEqual(
                tuple(item["state"] for item in group_document["query_parts"]),
                ("failed", "unattempted"),
            )
            self.assertEqual(document["unattempted_acquisition_groups"], ["fine"])

    @mock.patch("lfm.model.chip_creation.acquire_prepared_request")
    def test_explicit_overwrite_clears_only_current_sample_before_acquisition(
        self,
        acquire,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root, retention="always")
            prepared = self.prepared(self.request(1))
            sample_dir = config.intermediate_root / prepared.request.sample_id
            other_dir = config.intermediate_root / "other-sample"
            sample_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)
            (sample_dir / "stale.tif").write_bytes(b"stale")

            def fail_after_check(prepared_request, actual_config):
                self.assertFalse(sample_dir.exists())
                self.assertTrue(other_dir.is_dir())
                return ChipAcquisitionResult(
                    prepared_request=prepared_request,
                    status="failed",
                    diagnostics=(
                        AcquisitionDiagnostic(
                            "acquisition_error",
                            "injected acquisition failure",
                            "error",
                        ),
                    ),
                )

            acquire.side_effect = fail_after_check

            result = create_chip(prepared, config, overwrite=True)

            self.assertEqual(result.status, "failed")
            self.assertTrue(other_dir.is_dir())

    @mock.patch("lfm.model.chip_creation.acquire_prepared_request")
    def test_create_chip_raises_typed_label_failure_before_acquisition(self, acquire):
        diagnostic = LabelValidationDiagnostic(
            "label_shape_mismatch",
            "label and chip shapes differ",
        )
        prepared = self.prepared(
            self.request(1),
            status="failed",
            diagnostics=(diagnostic,),
        )
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))

            with self.assertRaises(LabelMismatchError) as context:
                create_chip(prepared, config)

        self.assertEqual(context.exception.sample_id, prepared.request.sample_id)
        acquire.assert_not_called()

    @mock.patch("lfm.model.chip_creation.preflight_chip_requests")
    def test_batch_records_label_failure_and_continues_to_later_sample(
        self,
        preflight,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            failed_request = self.request(1)
            skipped_request = self.request(2)
            diagnostic = LabelValidationDiagnostic(
                "missing_label",
                "label is absent",
            )
            failed = self.prepared(
                failed_request,
                status="failed",
                diagnostics=(diagnostic,),
            )
            skipped = self.prepared(skipped_request, status="skipped")
            plan = SplitPlan(assignments=(failed.assignment, skipped.assignment))
            preflight.return_value = BatchPreflightResult((failed, skipped), plan)

            batch = create_chips((skipped_request, failed_request), config)

            self.assertIsInstance(batch, ChipBatchResult)
            self.assertEqual(
                tuple(result.status for result in batch.results),
                ("failed", "skipped"),
            )
            self.assertTrue(batch.manifest_path.is_file())
            manifest = json.loads(batch.manifest_path.read_text())
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertEqual(manifest["samples"][0]["assigned_split"], "train")

    @mock.patch("lfm.model.chip_creation.create_chips")
    @mock.patch("lfm.model.chip_creation.chip_requests_from_reference_directory")
    def test_reference_directory_convenience_forwards_sorted_request_builder(
        self,
        discover,
        create_many,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            requests = (self.request(1), self.request(2))
            expected = object()
            discover.return_value = requests
            create_many.return_value = expected
            grouping = lambda reference: reference.sample_id.split("_", 1)[0]

            result = create_chips_from_reference_directory(
                root / "references",
                config,
                split_group_key=grouping,
                recursive=True,
                overwrite=True,
            )

            self.assertIs(result, expected)
            discover.assert_called_once_with(
                root / "references",
                split_group_key=grouping,
                recursive=True,
                edge_samples=21,
                sample_limit=None,
            )
            create_many.assert_called_once_with(requests, config, overwrite=True)

    def _failed_preflight_batch(self, requests, config):
        ordered = materialize_requests(requests, sample_limit=config.sample_limit)
        plan = plan_splits(ordered, config.split_config)
        prepared = tuple(
            PreparedChipRequest(
                request,
                plan.assignment_for(request.sample_id),
                ChipPreflight(
                    status=(
                        "failed"
                        if plan.assignment_for(request.sample_id).assigned_split
                        is not None
                        else "skipped"
                    ),
                    assigned_split=plan.assignment_for(request.sample_id).assigned_split,
                    label_diagnostics=(
                        LabelValidationDiagnostic("missing_label", "label is absent"),
                    ),
                ),
            )
            for request in ordered
        )
        return BatchPreflightResult(prepared, plan)

    def _passed_preflight_batch(self, requests, config):
        ordered = materialize_requests(requests, sample_limit=config.sample_limit)
        plan = plan_splits(ordered, config.split_config)
        config.label_source.mkdir(parents=True, exist_ok=True)
        prepared = []
        for request in ordered:
            assignment = plan.assignment_for(request.sample_id)
            label_path = config.label_source / f"{request.sample_id}_label.npy"
            label_path.write_bytes(b"stable-label")
            prepared.append(
                PreparedChipRequest(
                    request,
                    assignment,
                    ChipPreflight(
                        status=(
                            "passed"
                            if assignment.assigned_split is not None
                            else "skipped"
                        ),
                        assigned_split=assignment.assigned_split,
                        resolved_label_path=(
                            label_path
                            if assignment.assigned_split is not None
                            else None
                        ),
                    ),
                )
            )
        return BatchPreflightResult(tuple(prepared), plan)

    def _publish_fake_chip(self, prepared, config, *, overwrite=False):
        split = prepared.assignment.assigned_split
        if split is None:
            return ChipResult(
                prepared.request,
                "skipped",
                prepared.preflight,
            )
        publication_root = (
            config.output_root
            if split == "unsplit"
            else config.output_root / split
        )
        chip_path = (
            publication_root
            / "chips"
            / f"{prepared.request.sample_id}{config.final_output_suffix}"
        )
        label_path = (
            publication_root
            / "labels"
            / f"{prepared.request.sample_id}_label.npy"
        )
        for path, payload in (
            (chip_path, b"stable-chip"),
            (label_path, b"stable-label"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            path.write_bytes(payload)
        return ChipResult(
            prepared.request,
            "success",
            prepared.preflight,
            chip_path=chip_path,
            label_path=label_path,
        )

    @mock.patch("lfm.model.chip_creation.preflight_chip_requests")
    @mock.patch("lfm.model.chip_creation.create_chip")
    def test_repeated_batches_are_manifest_deterministic_for_every_split_config(
        self,
        create_one,
        preflight,
    ):
        requests = tuple(self.request(index) for index in range(1, 7))
        split_configs = (
            SimpleSplitConfig(SplitPercentages(0.5, 0.25, 0.25), seed=7),
            MixedPercentageNumberSplitConfig(
                fixed_counts=SplitCounts(test=2),
                fixed_priority=("test",),
                remaining_percentages=SplitPercentages(0.75, 0.25, 0.0),
                seed=7,
            ),
            NumberSplitConfig(
                fixed_counts=SplitCounts(train=2, test=2),
                fixed_priority=("train", "test"),
                remainder_split="val",
                seed=7,
            ),
            NoSplitConfig(),
        )
        preflight.side_effect = self._passed_preflight_batch
        create_one.side_effect = self._publish_fake_chip
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, split_config in enumerate(split_configs):
                with self.subTest(split_config=type(split_config).__name__):
                    config = self.config(
                        root / str(index),
                        split_config=split_config,
                    )
                    first = create_chips(requests, config)
                    first_bytes = first.manifest_path.read_bytes()
                    first_membership = tuple(
                        sorted(
                            str(path.relative_to(config.output_root))
                            for split in ("train", "val", "test", "unsplit")
                            for role in ("chips", "labels")
                            for output_dir in (
                                (
                                    config.output_root
                                    if split == "unsplit"
                                    else config.output_root / split
                                )
                                / role,
                            )
                            if output_dir.is_dir()
                            for path in output_dir.iterdir()
                        )
                    )
                    second = create_chips(
                        tuple(reversed(requests)),
                        config,
                        overwrite=True,
                    )
                    self.assertEqual(first_bytes, second.manifest_path.read_bytes())
                    self.assertEqual(
                        first_membership,
                        tuple(
                            sorted(
                                str(path.relative_to(config.output_root))
                                for split in ("train", "val", "test", "unsplit")
                                for role in ("chips", "labels")
                                for output_dir in (
                                    (
                                        config.output_root
                                        if split == "unsplit"
                                        else config.output_root / split
                                    )
                                    / role,
                                )
                                if output_dir.is_dir()
                                for path in output_dir.iterdir()
                            )
                        ),
                    )
                    self.assertEqual(
                        tuple(item.assigned_split for item in first.split_plan.assignments),
                        tuple(item.assigned_split for item in second.split_plan.assignments),
                    )

    @mock.patch("lfm.model.chip_creation.preflight_chip_requests")
    def test_unmet_number_target_warns_without_stopping_batch(self, preflight):
        requests = tuple(self.request(index) for index in range(1, 4))
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(
                Path(directory),
                split_config=NumberSplitConfig(
                    fixed_counts=SplitCounts(test=100),
                    fixed_priority=("test",),
                ),
            )
            preflight.side_effect = self._failed_preflight_batch

            with self.assertWarns(UserWarning):
                batch = create_chips(requests, config)

        self.assertEqual(len(batch.results), len(requests))
        self.assertEqual(len(batch.split_plan.warnings), 1)


@unittest.skipUnless(HAS_RASTER_DEPS, "GDAL and NumPy are required")
class ChipCreationRasterTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import gdal, osr

        self.np = np
        self.gdal = gdal
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(load_lunar_geographic_wkt())
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.grid = TargetGrid(
            crs_wkt=spatial_reference.ExportToWkt(),
            transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            bounds=(0.0, 0.0, 2.0, 2.0),
            width=2,
            height=2,
        )

    def config(self, root):
        source = TileSourceConfig(
            "static",
            root / "source",
            root / "source/index.gpkg",
            band_names=("elevation",),
        )
        return ChipConfig(
            output_root=root / "dataset",
            intermediate_root=root / "intermediate",
            label_source=root / "labels",
            acquisition_groups=(
                AcquisitionGroupConfig(
                    "coarse",
                    TileConfig(root / "unused", 5, (source,)),
                ),
            ),
            output_modalities=(
                OutputModalityConfig(
                    "coarse",
                    "static",
                    "static",
                    output_band_names=("elevation",),
                ),
            ),
            split_config=NoSplitConfig(),
            intermediate_retention="never",
        )

    def write_cube(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(path),
            2,
            2,
            1,
            self.gdal.GDT_Float32,
        )
        dataset.SetProjection(self.grid.crs_wkt)
        dataset.SetGeoTransform(self.grid.transform)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(-32768.0)
        band.SetDescription("elevation")
        band.SetMetadataItem("Name", "elevation")
        band.WriteArray(self.np.asarray(((1.0, 2.0), (3.0, 4.0))))
        dataset.FlushCache()
        dataset = None

    @mock.patch("lfm.model.chip_creation.acquire_prepared_request")
    def test_complete_serial_stage_integration_publishes_pair(self, acquire):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            config.label_source.mkdir(parents=True)
            label_path = config.label_source / "SAMPLE_1_label.npy"
            self.np.save(label_path, self.np.zeros((2, 2), dtype=self.np.uint8))
            request = ChipRequest(
                sample_id="SAMPLE_1",
                target_grid=self.grid,
                geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 2.0),
                split_group_key="site-1",
                label_path=label_path,
            )
            cube_path = (
                config.intermediate_root / request.sample_id / "coarse/cube.tif"
            )
            self.write_cube(cube_path)
            record = TileCubeRecord(
                source_name="static",
                zone="1N",
                zoom_level=5,
                tile_x=0,
                tile_y=0,
                product_id=None,
                path=cube_path,
                band_names=("elevation",),
                crs_wkt=self.grid.crs_wkt,
                nodata_values=(-32768.0,),
            )
            def acquired(prepared_request, actual_config):
                self.assertIs(actual_config, config)
                group_result = AcquisitionGroupResult(
                    sample_id=request.sample_id,
                    acquisition_group="coarse",
                    zoom_level=5,
                    output_dir=cube_path.parent,
                    logical_aoi=request.geographic_aoi,
                    query_parts=(request.geographic_aoi,),
                    selectors=(),
                    status="complete",
                    records=(record,),
                    inventory_paths=(cube_path,),
                    attempted_query_parts=(request.geographic_aoi,),
                )
                return ChipAcquisitionResult(
                    prepared_request=prepared_request,
                    status="complete",
                    group_results=(group_result,),
                )

            acquire.side_effect = acquired

            batch = create_chips((request,), config)
            result = batch.results[0]

            self.assertEqual(result.status, "success")
            self.assertEqual(result.preflight.assigned_split, "unsplit")
            self.assertTrue(result.chip_path.is_file())
            self.assertTrue(result.label_path.is_file())
            self.assertEqual(result.chip_path.parent, config.output_root / "chips")
            self.assertEqual(result.label_path.parent, config.output_root / "labels")
            self.assertEqual(result.label_path.read_bytes(), label_path.read_bytes())
            self.assertFalse((config.intermediate_root / request.sample_id).exists())
            self.assertTrue(batch.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
