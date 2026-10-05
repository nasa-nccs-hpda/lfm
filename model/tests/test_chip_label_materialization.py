"""A3 source safety, categorical resampling, and verified staging contracts."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfm.model.chip_label_materialization import (
    _MaskReader, _nearest_indices, _staging_path, materialize_semantic_label,
)
from lfm.model.chip_label_planning import plan_label_preparation
from lfm.model.chip_requests import target_grid_from_pixel_bounds
from lfm.model.chip_types import LabelInput, LabelMismatchError, LabelPreparationPlan
from lfm.model.tests.test_chip_label_planning import grid, request, snapshot


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_GDAL = importlib.util.find_spec("osgeo") is not None


class SemanticMaterializationSafetyTestCase(unittest.TestCase):
    def test_changed_source_fails_before_planning_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "full.npy"
            source.write_bytes(b"changed source")
            item = request(source, grid(), relation="exact")
            plan = LabelPreparationPlan(item.label_input, item.target_grid, "exact", "a" * 64)
            with patch("lfm.model.chip_label_materialization.plan_label_preparation") as planner:
                with self.assertRaises(LabelMismatchError) as caught:
                    materialize_semantic_label(item, plan, staging_root=root / "work")
            planner.assert_not_called()
            self.assertEqual(caught.exception.diagnostics[0].code, "label_source_changed")
            self.assertFalse((root / "work").exists())
            self.assertEqual(source.read_bytes(), b"changed source")

    def test_instance_mask_cannot_be_published_as_a_semantic_label(self):
        item = request("full.npz", grid(), relation="exact")
        plan = LabelPreparationPlan(item.label_input, item.target_grid, "exact", "a" * 64)
        with self.assertRaises(LabelMismatchError) as caught:
            materialize_semantic_label(item, plan)
        self.assertEqual(caught.exception.diagnostics[0].code, "unsupported_label_materialization")

    def test_request_grid_must_match_plan(self):
        item = request("full.npy", grid(), relation="exact")
        plan = LabelPreparationPlan(item.label_input, grid(width=5), "exact", "a" * 64)
        with self.assertRaises(LabelMismatchError) as caught:
            materialize_semantic_label(item, plan)
        self.assertEqual(caught.exception.diagnostics[0].code, "label_plan_mismatch")

    def test_request_source_grid_and_relation_cannot_be_overridden_by_plan(self):
        item = request("full.npy", grid())
        for source in (replace(item.label_input, source_grid=grid(width=5)),
                       replace(item.label_input, relation="exact")):
            plan = LabelPreparationPlan(source, item.target_grid, "exact", "a" * 64)
            with self.subTest(source=source), self.assertRaises(LabelMismatchError) as caught:
                materialize_semantic_label(item, plan)
            self.assertEqual(caught.exception.diagnostics[0].code, "label_plan_mismatch")

    def test_staging_cannot_own_source_or_traverse_sample_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "M1_r0_c0" / "full.npy"
            item = request(source, grid())
            plan = LabelPreparationPlan(item.label_input, grid(), "nearest_warp", "a" * 64)
            with self.assertRaises(LabelMismatchError):
                _staging_path(item, plan, root)
            item = request(root / "full.npy", grid())
            plan = replace(plan, source=item.label_input)
            elsewhere = root / "other-sample"
            elsewhere.mkdir()
            (root / item.sample_id).symlink_to(elsewhere, target_is_directory=True)
            with self.assertRaises(LabelMismatchError):
                _staging_path(item, plan, root)
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_exact_plan_reuses_source_without_staging_io(self):
        # Mock only already-tested format planning; exercise the no-copy branch
        # and real checksums without requiring NumPy in the lightweight suite.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.npy"
            source.write_bytes(b"format validation mocked")
            item = request(source, grid(), relation="exact")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            plan = LabelPreparationPlan(item.label_input, item.target_grid, "exact", digest)
            before = snapshot(root)
            with patch("lfm.model.chip_label_materialization.plan_label_preparation", return_value=plan):
                artifact = materialize_semantic_label(item, plan, staging_root=root / "unused")
            self.assertEqual(artifact.path, source)
            self.assertEqual(artifact.sha256, digest)
            self.assertEqual(snapshot(root), before)


@unittest.skipUnless(HAS_NUMPY, "NumPy is unavailable")
class IntegerSamplingTestCase(unittest.TestCase):
    def test_shared_npz_reader_slices_only_mask_without_changing_archive(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full.npz"
            mask = np.arange(12, dtype=np.uint64).reshape(3, 4)
            np.savez(path, mask=mask, bboxes=np.tile([[0, 0, 4, 3]], (11, 1)), num_craters=np.array(11))
            before = path.read_bytes()
            item = request(path, grid())
            with _MaskReader(item, item.label_input) as reader:
                np.testing.assert_array_equal(reader.read(1, 1, 2, 2), mask[1:3, 1:3])
            self.assertEqual(path.read_bytes(), before)

    def test_nearest_matches_window_and_rejects_padding(self):
        import numpy as np
        source = grid()
        target = target_grid_from_pixel_bounds(source, (1, 1, 3, 3))
        rows, cols = _nearest_indices(request(target=target), source, (0, 0, 2, 2), None)
        mask = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(mask[rows, cols], mask[1:3, 1:3])
        outside = grid(2, 2, (-100, 100, 0, 300, 0, -100))
        with self.assertRaises(LabelMismatchError):
            _nearest_indices(request(target=outside), source, (0, 0, 2, 2), None)

    def test_ties_and_tiny_coordinate_noise_are_deterministic(self):
        import numpy as np
        source = grid()
        for noise in (-1e-10, 0, 1e-10):
            mapper = lambda col, row: (1 + noise, 1 + noise)
            rows, cols = _nearest_indices(request(), source, (0, 0, 1, 1), mapper)
            np.testing.assert_array_equal(rows, [[1]])
            np.testing.assert_array_equal(cols, [[1]])


@unittest.skipUnless(HAS_NUMPY and HAS_GDAL, "NumPy/GDAL are unavailable")
class SemanticMaterializationRasterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_array(self, values, *, source_grid=None, target=None, name="scene.npy", relation="clip_to_target"):
        import numpy as np
        path = self.root / name
        np.save(path, values)
        source_grid = source_grid or grid(width=values.shape[1], height=values.shape[0])
        item = request(path, source_grid, target or source_grid, relation=relation)
        return item, plan_label_preparation(item, path)

    def materialize(self, item, plan, root="work"):
        return materialize_semantic_label(item, plan, staging_root=self.root / root)

    def test_exact_array_remains_byte_identical_and_creates_no_directory(self):
        import numpy as np
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint16), relation="exact")
        before = snapshot(self.root)
        artifact = self.materialize(item, plan)
        self.assertEqual(artifact.path, item.label_path)
        self.assertEqual(artifact.sha256, plan.source_sha256)
        self.assertEqual(snapshot(self.root), before)

    def test_window_preserves_labels_grid_hash_and_deterministic_bytes(self):
        import numpy as np
        values = np.arange(12, dtype=np.int16).reshape(3, 4)
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(values, target=target)
        original = item.label_path.read_bytes()
        first, second = self.materialize(item, plan), self.materialize(item, plan, "repeat")
        np.testing.assert_array_equal(np.load(first.path), [[5, 6], [9, 10]])
        self.assertEqual(first.target_grid, target)
        self.assertEqual(first.sha256, hashlib.sha256(first.path.read_bytes()).hexdigest())
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.path, self.root / "work" / item.sample_id / "labels" / f"{item.sample_id}_label.npy")
        self.assertEqual(list(first.path.parent.iterdir()), [first.path])
        self.assertEqual(item.label_path.read_bytes(), original)

    def test_large_integer_ids_survive_nearest_without_float_conversion(self):
        import numpy as np
        for dtype, labels in ((np.uint64, [0, 2**53 + 1, 2**63 + 1, 2**64 - 1]),
                              (np.int64, [-2**63, -1, 2**53 + 1, 2**63 - 1])):
            values = np.array(labels, dtype=dtype).reshape(2, 2)
            target = grid(4, 4, (0, 50, 0, 300, 0, -50))
            item, plan = self.make_array(values, target=target, name=f"{np.dtype(dtype).name}.npy")
            artifact = self.materialize(item, plan, np.dtype(dtype).name)
            self.assertEqual(plan.method, "nearest_warp")
            actual = np.load(artifact.path)
            self.assertEqual(actual.dtype, values.dtype)
            np.testing.assert_array_equal(actual, values.repeat(2, axis=0).repeat(2, axis=1))

    def test_rotated_grid_window_and_warp(self):
        import numpy as np
        values = np.arange(100, dtype=np.uint16).reshape(10, 10)
        source = grid(10, 10, (0, 80, 60, 1000, 60, -80))
        target = target_grid_from_pixel_bounds(source, (4, 4, 6, 6))
        item, plan = self.make_array(values, source_grid=source, target=target)
        first = self.materialize(item, plan, "window")
        np.testing.assert_array_equal(np.load(first.path), [[44, 45], [54, 55]])
        target = grid(2, 2, (650, 50, 0, 950, 0, -50))
        item = replace(item, target_grid=target)
        plan = plan_label_preparation(item, item.label_path)
        self.assertEqual(plan.method, "nearest_warp")
        warped = self.materialize(item, plan, "warp")
        np.testing.assert_array_equal(np.load(warped.path), [[44, 45], [54, 55]])

    def test_different_crs_nearest_preserves_a0_values(self):
        import numpy as np
        from osgeo import osr
        target = grid(4, 4, (250000, 50, 0, 1200, 0, -50))
        srs = osr.SpatialReference()
        srs.ImportFromWkt(target.crs_wkt)
        srs.SetProjParm("false_easting", 250100)
        source = replace(grid(2, 2, (250100, 100, 0, 1200, 0, -100)), crs_wkt=srs.ExportToWkt())
        values = np.array([[4, 8], [12, 16]], dtype=np.uint16)
        item, plan = self.make_array(values, source_grid=source, target=target)
        self.assertEqual(plan.method, "nearest_warp")
        artifact = self.materialize(item, plan)
        np.testing.assert_array_equal(np.load(artifact.path), values.repeat(2, axis=0).repeat(2, axis=1))

    def test_background_and_multiple_blocks(self):
        import numpy as np
        source = grid(520, 270)
        target = target_grid_from_pixel_bounds(source, (1, 1, 519, 269))
        item, plan = self.make_array(np.zeros((270, 520), dtype=np.uint8), source_grid=source, target=target)
        artifact = self.materialize(item, plan)
        actual = np.load(artifact.path)
        self.assertEqual(actual.shape, (268, 518))
        self.assertFalse(actual.any())

    def test_two_samples_can_reuse_a_parent_label_without_path_collisions(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.arange(12, dtype=np.uint8).reshape(3, 4), target=target)
        first = self.materialize(item, plan)
        second = self.materialize(replace(item, sample_id="M1_r100_c100"), plan)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.plan.source.path, second.plan.source.path)

    def test_source_change_after_writing_never_installs_artifact(self):
        import numpy as np
        from lfm.model.chip_labels import _label_error
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)
        error = _label_error(item, code="label_source_changed", message="simulated concurrent change")
        with patch("lfm.model.chip_label_materialization._verify_source", side_effect=(None, None, error)):
            with self.assertRaises(LabelMismatchError) as caught:
                self.materialize(item, plan)
        self.assertEqual(caught.exception.diagnostics[0].code, "label_source_changed")
        self.assertEqual(snapshot(self.root), {"scene.npy": item.label_path.read_bytes()})

    def test_staged_pixel_corruption_fails_reopen_validation(self):
        import numpy as np
        from lfm.model.chip_label_materialization import _validate_staged
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)

        def corrupt_then_validate(request, path, dtype, expected_hash):
            pixels = np.load(path, mmap_mode="r+", allow_pickle=False)
            pixels[0, 0] = 1
            pixels.flush()
            pixels._mmap.close()
            return _validate_staged(request, path, dtype, expected_hash)

        with patch("lfm.model.chip_label_materialization._validate_staged", side_effect=corrupt_then_validate):
            with self.assertRaisesRegex(LabelMismatchError, "pixels differ"):
                self.materialize(item, plan)
        self.assertEqual(snapshot(self.root), {"scene.npy": item.label_path.read_bytes()})

    def test_stale_plan_invalid_dtype_and_partial_coverage_write_nothing(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)
        with self.assertRaises(LabelMismatchError):
            self.materialize(item, replace(plan, source_window=(0, 0, 2, 2)))
        np.save(item.label_path, np.zeros((3, 4), dtype=np.float32))
        bad = replace(plan, source_sha256=hashlib.sha256(item.label_path.read_bytes()).hexdigest())
        with self.assertRaises(LabelMismatchError) as caught:
            self.materialize(item, bad)
        self.assertEqual(caught.exception.diagnostics[0].code, "invalid_label_dtype")
        np.save(item.label_path, np.zeros((3, 4), dtype=np.uint8))
        outside = grid(4, 3, (-100, 100, 0, 300, 0, -100))
        item = replace(item, target_grid=outside)
        bad = replace(plan, target_grid=outside, method="nearest_warp", source_window=None)
        with self.assertRaises(LabelMismatchError) as caught:
            self.materialize(item, bad)
        self.assertEqual(caught.exception.diagnostics[0].code, "incomplete_label_coverage")
        self.assertFalse((self.root / "work").exists())

    def test_existing_artifact_is_not_overwritten(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)
        artifact = self.materialize(item, plan)
        before = snapshot(self.root)
        with self.assertRaises(LabelMismatchError) as caught:
            self.materialize(item, plan)
        self.assertEqual(caught.exception.diagnostics[0].code, "label_artifact_exists")
        self.assertEqual(snapshot(self.root), before)

    def test_failed_reopen_validation_removes_only_own_temporary_file(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)
        with patch("lfm.model.chip_label_materialization._validate_staged", side_effect=ValueError("corrupt staged mask")):
            with self.assertRaises(LabelMismatchError):
                self.materialize(item, plan)
        self.assertEqual(snapshot(self.root), {"scene.npy": item.label_path.read_bytes()})

    def test_concurrent_artifact_wins_without_overwrite_or_deletion(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)

        def race(source, destination):
            destination.write_bytes(b"other worker's artifact")
            raise FileExistsError("concurrent install")

        with patch("lfm.model.chip_label_materialization.os.link", side_effect=race):
            with self.assertRaises(LabelMismatchError):
                self.materialize(item, plan)
        destination = self.root / "work" / item.sample_id / "labels" / f"{item.sample_id}_label.npy"
        self.assertEqual(destination.read_bytes(), b"other worker's artifact")
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_changed_sidecar_is_revalidated_before_writing(self):
        import numpy as np
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
        item, plan = self.make_array(np.zeros((3, 4), dtype=np.uint8), target=target)
        item.label_path.with_suffix(".json").write_text(json.dumps({"source_grid": grid(width=5).to_dict()}))
        with self.assertRaises(LabelMismatchError):
            self.materialize(item, plan)
        self.assertFalse((self.root / "work").exists())

    def test_geotiff_exact_window_and_nodata_rejection(self):
        import numpy as np
        from osgeo import gdal
        path = self.root / "full.tif"
        source = grid()
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), 4, 3, 1, gdal.GDT_Int16)
        dataset.SetProjection(source.crs_wkt)
        dataset.SetGeoTransform(source.transform)
        dataset.GetRasterBand(1).SetNoDataValue(-9999)
        dataset.GetRasterBand(1).WriteArray(np.arange(12, dtype=np.int16).reshape(3, 4))
        dataset = None
        original = path.read_bytes()
        for target, root, expected in ((source, "exact", np.arange(12).reshape(3, 4)),
                                      (target_grid_from_pixel_bounds(source, (1, 1, 3, 3)), "window", [[5, 6], [9, 10]])):
            item = request(path, target=target)
            plan = plan_label_preparation(item, path)
            artifact = self.materialize(item, plan, root)
            self.assertEqual(artifact.path.suffix, ".npy")
            np.testing.assert_array_equal(np.load(artifact.path), expected)
        self.assertEqual(path.read_bytes(), original)
        dataset = gdal.Open(str(path), gdal.GA_Update)
        dataset.GetRasterBand(1).WriteArray(np.full((3, 4), -9999, dtype=np.int16))
        dataset = None
        # Updating the expected hash cannot bypass source NoData validation.
        plan = replace(plan, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(LabelMismatchError) as caught:
            self.materialize(item, plan, "invalid")
        self.assertEqual(caught.exception.diagnostics[0].code, "label_nodata_in_target")
        self.assertFalse((self.root / "invalid").exists())

    def test_geotiff_nearest_preserves_uint64_values(self):
        import numpy as np
        from osgeo import gdal
        if not hasattr(gdal, "GDT_UInt64"):
            self.skipTest("GDAL lacks UInt64 raster support")
        path = self.root / "uint64.tif"
        source = grid(2, 2)
        values = np.array([[0, 2**53 + 1], [2**63 + 1, 2**64 - 1]], dtype=np.uint64)
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), 2, 2, 1, gdal.GDT_UInt64)
        dataset.SetProjection(source.crs_wkt)
        dataset.SetGeoTransform(source.transform)
        dataset.GetRasterBand(1).WriteArray(values)
        dataset = None
        target = grid(4, 4, (0, 50, 0, 300, 0, -50))
        item = request(path, target=target)
        plan = plan_label_preparation(item, path)
        artifact = self.materialize(item, plan)
        actual = np.load(artifact.path)
        self.assertEqual(actual.dtype, values.dtype)
        np.testing.assert_array_equal(actual, values.repeat(2, axis=0).repeat(2, axis=1))


if __name__ == "__main__":
    unittest.main()
