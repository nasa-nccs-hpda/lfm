import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    NumberSplitConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitCounts,
    SplitPercentages,
)
from lfm.model.chip_labels import preflight_label, resolve_label_path
from lfm.model.chip_preflight import preflight_chip_requests
from lfm.model.chip_requests import UnsupportedCoverageError
from lfm.model.chip_types import (
    ChipRequest,
    GeographicAOI,
    LabelMismatchError,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig


@unittest.skipUnless(HAS_NUMPY, "NumPy is unavailable")
class ChipLabelTestCase(unittest.TestCase):
    def grid(self):
        return TargetGrid(
            crs_wkt="GEOGCRS[Moon]",
            transform=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0),
            bounds=(0.0, 0.0, 4.0, 3.0),
            width=4,
            height=3,
        )

    def request(self, sample_id="M1_r0_c0", **overrides):
        values = {
            "sample_id": sample_id,
            "target_grid": self.grid(),
            "geographic_aoi": GeographicAOI(3.0, 0.0, 0.0, 4.0),
            "split_group_key": sample_id.split("_", 1)[0],
        }
        values.update(overrides)
        return ChipRequest(**values)

    def config(self, label_source, split_config=None):
        source = TileSourceConfig(
            name="wac",
            data_dir=Path("/data/wac"),
            index_path=Path("/data/wac/index.gpkg"),
            selection_mode="product_id",
        )
        group = AcquisitionGroupConfig(
            "wac_grid",
            TileConfig(Path("/cubes"), 5, (source,)),
        )
        return ChipConfig(
            output_root=Path(label_source) / "dataset_output",
            label_source=label_source,
            acquisition_groups=(group,),
            output_modalities=(
                OutputModalityConfig("wac_grid", "wac", "wac"),
            ),
            split_config=split_config
            or SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
        )

    def test_full_offset_identity_resolves_distinct_labels_for_one_product(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "M1_r0_c0_label.npy"
            second = root / "M1_r300_c0_label.npy"
            np.save(first, np.zeros((3, 4), dtype=np.uint8))  # type: ignore[union-attr]
            np.save(second, np.ones((3, 4), dtype=np.uint8))  # type: ignore[union-attr]

            first_result = resolve_label_path(self.request("M1_r0_c0"), root)
            second_result = resolve_label_path(self.request("M1_r300_c0"), root)

        self.assertEqual(first_result.name, first.name)
        self.assertEqual(second_result.name, second.name)

    def test_missing_duplicate_and_misidentified_labels_raise_typed_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request = self.request()
            with self.assertRaisesRegex(LabelMismatchError, "No label matched"):
                resolve_label_path(request, root)
            np.save(root / "M1_r0_c0_label.npy", np.zeros((3, 4), dtype=np.uint8))
            np.savez(
                root / "M1_r0_c0_label.npz",
                mask=np.zeros((3, 4), dtype=np.uint8),
                bboxes=np.empty((0, 4), dtype=np.float32),
                num_craters=np.asarray(0),
            )
            with self.assertRaisesRegex(LabelMismatchError, "Multiple labels"):
                resolve_label_path(request, root)
            wrong = root / "M1_r300_c0_label.npy"
            np.save(wrong, np.zeros((3, 4), dtype=np.uint8))
            with self.assertRaisesRegex(LabelMismatchError, "normalizes"):
                resolve_label_path(self.request(label_path=wrong), root)

    def test_semantic_label_requires_integer_2d_target_shape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "M1_r0_c0_label.npy"
            np.save(path, np.zeros((3, 4), dtype=np.uint8))

            result = preflight_label(
                self.request(),
                label_source=root,
                assigned_split="train",
            )

            self.assertEqual(result.status, "passed")
            self.assertEqual(result.resolved_label_path, path)
            self.assertEqual(result.label_diagnostics[-1].code, "label_grid_unverified")
            np.save(path, np.zeros((4, 3), dtype=np.uint8))
            with self.assertRaisesRegex(LabelMismatchError, "does not match"):
                preflight_label(
                    self.request(),
                    label_source=root,
                    assigned_split="train",
                )
            np.save(path, np.zeros((3, 4), dtype=np.float32))
            with self.assertRaisesRegex(LabelMismatchError, "integer dtype"):
                preflight_label(
                    self.request(),
                    label_source=root,
                    assigned_split="train",
                )

    def test_instance_label_validates_archive_counts_boxes_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "M1_r0_c0_label.npz"
            mask = np.zeros((3, 4), dtype=np.uint8)
            mask[0, 0] = 1
            mask[1, 2] = 2
            np.savez(
                path,
                mask=mask,
                bboxes=np.asarray([[0, 0, 1, 1], [2, 1, 1, 1]], dtype=np.float32),
                num_craters=np.asarray(2, dtype=np.int64),
            )

            result = preflight_label(
                self.request(),
                label_source=root,
                assigned_split="test",
            )

            self.assertEqual(result.status, "passed")
            np.savez(
                path,
                mask=mask,
                bboxes=np.asarray([[0, 0, 1, 1]], dtype=np.float32),
                num_craters=np.asarray(2, dtype=np.int64),
            )
            with self.assertRaisesRegex(LabelMismatchError, "bboxes shape"):
                preflight_label(
                    self.request(),
                    label_source=root,
                    assigned_split="test",
                )

    def test_matching_and_mismatching_geospatial_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "M1_r0_c0_label.npy"
            np.save(path, np.zeros((3, 4), dtype=np.uint8))
            sidecar = path.with_suffix(path.suffix + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "sample_id": "M1_r0_c0",
                        "target_grid": {
                            "crs_wkt": self.grid().crs_wkt,
                            "transform": self.grid().transform,
                            "bounds": self.grid().bounds,
                            "width": 4,
                            "height": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = preflight_label(
                self.request(),
                label_source=root,
                assigned_split="val",
            )
            self.assertNotIn(
                "label_grid_unverified",
                {item.code for item in result.label_diagnostics},
            )
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["target_grid"]["transform"][0] = 1.0
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(LabelMismatchError):
                preflight_label(
                    self.request(),
                    label_source=root,
                    assigned_split="val",
                )

    def test_structured_label_grid_must_match_target_grid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "M1_r0_c0_label.npy"
            np.save(path, np.zeros((3, 4), dtype=np.uint8))
            matching = preflight_label(
                self.request(label_grid=self.grid()),
                label_source=root,
                assigned_split="train",
            )
            self.assertNotIn(
                "label_grid_unverified",
                {item.code for item in matching.label_diagnostics},
            )
            mismatched_grid = TargetGrid(
                crs_wkt=self.grid().crs_wkt,
                transform=(1.0, 1.0, 0.0, 3.0, 0.0, -1.0),
                bounds=(1.0, 0.0, 5.0, 3.0),
                width=4,
                height=3,
            )
            with self.assertRaisesRegex(LabelMismatchError, "transform"):
                preflight_label(
                    self.request(label_grid=mismatched_grid),
                    label_source=root,
                    assigned_split="train",
                )

    @mock.patch("lfm.model.chip_preflight.validate_request_geographic_aoi")
    @mock.patch("lfm.model.tiling.create_tiles_for_aoi")
    def test_batch_preflight_isolates_label_failure_without_tiling_or_outputs(
        self,
        create_tiles_for_aoi,
        validate_request_geographic_aoi,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            np.save(root / "M1_r0_c0_label.npy", np.zeros((3, 4), dtype=np.uint8))
            np.save(root / "M2_r0_c0_label.npy", np.zeros((2, 4), dtype=np.uint8))
            config = self.config(root)

            result = preflight_chip_requests(
                (self.request("M2_r0_c0"), self.request("M1_r0_c0")),
                config,
            )

            statuses = {
                item.request.sample_id: item.preflight.status
                for item in result.requests
            }
            self.assertEqual(statuses, {"M1_r0_c0": "passed", "M2_r0_c0": "failed"})
            self.assertFalse(config.output_root.exists())
            create_tiles_for_aoi.assert_not_called()
            self.assertEqual(validate_request_geographic_aoi.call_count, 2)

    @mock.patch("lfm.model.chip_preflight.validate_request_geographic_aoi")
    def test_unassigned_number_remainder_skips_label_resolution(
        self,
        validate_request_geographic_aoi,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            np.save(root / "M1_r0_c0_label.npy", np.zeros((3, 4), dtype=np.uint8))
            np.save(root / "M2_r0_c0_label.npy", np.zeros((3, 4), dtype=np.uint8))
            config = self.config(
                root,
                NumberSplitConfig(
                    fixed_counts=SplitCounts(test=1),
                    fixed_priority=("test",),
                ),
            )

            with mock.patch(
                "lfm.model.chip_labels.resolve_label_path",
                wraps=resolve_label_path,
            ) as resolve_label:
                result = preflight_chip_requests(
                    (self.request("M1_r0_c0"), self.request("M2_r0_c0")),
                    config,
                )

        statuses = {item.preflight.status for item in result.requests}
        self.assertEqual(statuses, {"passed", "skipped"})
        self.assertEqual(
            sum(item.eligible_for_acquisition for item in result.requests),
            1,
        )
        resolve_label.assert_called_once()
        resolved_request = resolve_label.call_args.args[0]
        assigned_request = next(
            item.request
            for item in result.requests
            if item.assignment.assigned_split is not None
        )
        self.assertEqual(resolved_request.sample_id, assigned_request.sample_id)
        self.assertEqual(validate_request_geographic_aoi.call_count, 2)

    @mock.patch("lfm.model.chip_preflight.validate_request_geographic_aoi")
    def test_invalid_geography_fails_one_sample_and_continues(
        self,
        validate_request_geographic_aoi,
    ):
        def validate(request):
            if request.sample_id == "M1_r0_c0":
                raise UnsupportedCoverageError("north of numbered LTM coverage")
            return (request.geographic_aoi,)

        validate_request_geographic_aoi.side_effect = validate
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            np.save(root / "M2_r0_c0_label.npy", np.zeros((3, 4), dtype=np.uint8))

            result = preflight_chip_requests(
                (self.request("M1_r0_c0"), self.request("M2_r0_c0")),
                self.config(root),
            )

        prepared = {item.request.sample_id: item for item in result.requests}
        self.assertEqual(prepared["M1_r0_c0"].preflight.status, "failed")
        self.assertEqual(
            prepared["M1_r0_c0"].preflight.label_diagnostics[0].code,
            "unsupported_polar_coverage",
        )
        self.assertEqual(prepared["M2_r0_c0"].preflight.status, "passed")


if __name__ == "__main__":
    unittest.main()
