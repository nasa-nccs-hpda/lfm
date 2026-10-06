"""A6T: task selection, native integer TIFF sampling and canonical artifacts."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from lfm.model.chip_instance_labels import materialize_instance_label
from lfm.model.chip_label_materialization import materialize_semantic_label
from lfm.model.chip_label_planning import _hash_file, plan_label_preparation
from lfm.model.chip_requests import target_grid_from_pixel_bounds
from lfm.model.chip_types import LabelInput, LabelMismatchError
from lfm.model.tests.test_chip_label_planning import HAS_GDAL, HAS_NUMPY, grid, request


class TiffTaskContractTestCase(unittest.TestCase):
    def test_auto_is_semantic_and_instance_requires_explicit_kind(self):
        for suffix in (".tif", ".TIFF"):
            self.assertEqual(LabelInput("labels" + suffix).kind, "semantic")
            source = LabelInput("labels" + suffix, kind="raster_instance")
            self.assertEqual(LabelInput.from_dict(source.to_dict()), source)


@unittest.skipUnless(HAS_NUMPY and HAS_GDAL, "NumPy/GDAL are unavailable")
class TiffLabelConversionTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import gdal
        self.np, self.gdal = np, gdal
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raster(self, values, source=None, nodata=None, bands=1, dtype=None):
        values = self.np.asarray(values)
        source = source or grid(values.shape[1], values.shape[0])
        path = self.root / "labels.tif"
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(path), source.width, source.height, bands,
            dtype or self.gdal.GDT_Int32)
        dataset.SetGeoTransform(source.transform)
        dataset.SetProjection(source.crs_wkt)
        for index in range(1, bands + 1):
            band = dataset.GetRasterBand(index)
            band.WriteArray(values)
            if nodata is not None:
                band.SetNoDataValue(nodata)
        band = dataset = None
        return path, source

    def convert(self, path, target, kind="raster_instance", stage="work"):
        req = request(path, target=target, kind=kind)
        digest = _hash_file(path)
        plan = plan_label_preparation(req, path)
        converter = materialize_instance_label if kind == "raster_instance" else materialize_semantic_label
        artifact = converter(req, plan, staging_root=self.root / stage)
        self.assertEqual(_hash_file(path), digest)
        return artifact

    def test_exact_disconnected_ids_boxes_and_deterministic_archive(self):
        path, source = self.raster([[90, 0, 12, 0], [0, 38, 12, 0], [90, 0, 0, 0]])
        artifact = self.convert(path, source)
        self.assertEqual(artifact.instance_id_map, ((12, 1), (38, 2), (90, 3)))
        with self.np.load(artifact.path) as labels:
            self.np.testing.assert_array_equal(labels["mask"], [[3, 0, 1, 0], [0, 2, 1, 0], [3, 0, 0, 0]])
            self.np.testing.assert_array_equal(labels["bboxes"], [[2, 0, 1, 2], [1, 1, 1, 1], [0, 0, 1, 3]])
            self.assertEqual(int(labels["num_craters"]), 3)
        second = self.convert(path, source, stage="second")
        self.assertEqual(artifact.sha256, second.sha256)
        with self.assertRaises(LabelMismatchError):
            self.convert(path, source)

    def test_clipped_rotated_window_uses_visible_boxes(self):
        source = grid(4, 3, (0, 100, 10, 300, 20, -100))
        path, _ = self.raster([[90, 0, 12, 0], [0, 38, 12, 0], [90, 0, 0, 0]], source)
        target = target_grid_from_pixel_bounds(source, (1, 0, 3, 2))
        artifact = self.convert(path, target)
        self.assertEqual(artifact.plan.method, "aligned_window")
        self.assertEqual(artifact.instance_id_map, ((12, 1), (38, 2)))
        with self.np.load(artifact.path) as labels:
            self.np.testing.assert_array_equal(labels["bboxes"], [[1, 0, 1, 2], [0, 1, 1, 1]])

    def test_nearest_warns_only_for_intersecting_omitted_ids(self):
        path, source = self.raster([[12, 0, 90, 90], [0, 38, 90, 90], [0, 0, 90, 90]])
        target = grid(1, 1, (0, 200, 0, 300, 0, -200))
        artifact = self.convert(path, target)
        self.assertEqual(artifact.plan.method, "nearest_warp")
        self.assertEqual(artifact.instance_id_map, ((38, 1),))
        omissions = [d for d in artifact.diagnostics if d.code == "subpixel_instance"]
        self.assertEqual(len(omissions), 1)
        self.assertIn("12", omissions[0].message)
        self.assertEqual(omissions[0].severity, "warning")

    def test_empty_mask_has_empty_boxes(self):
        path, source = self.raster(self.np.zeros((3, 4), dtype=int))
        artifact = self.convert(path, source)
        with self.np.load(artifact.path) as labels:
            self.assertEqual(labels["bboxes"].shape, (0, 4))
            self.assertEqual(int(labels["num_craters"]), 0)

    def test_different_crs_nearest_preserves_ids(self):
        from osgeo import osr
        target = grid(4, 4, (250000, 50, 0, 1200, 0, -50))
        srs = osr.SpatialReference()
        srs.ImportFromWkt(target.crs_wkt)
        srs.SetProjParm("false_easting", 250100)
        source = replace(grid(2, 2, (250100, 100, 0, 1200, 0, -100)), crs_wkt=srs.ExportToWkt())
        path, _ = self.raster([[0, 12], [38, 90]], source)
        artifact = self.convert(path, target)
        self.assertEqual(artifact.plan.method, "nearest_warp")
        with self.np.load(artifact.path) as labels:
            self.np.testing.assert_array_equal(labels["mask"],
                self.np.array([[0, 1], [2, 3]]).repeat(2, axis=0).repeat(2, axis=1))

    def test_uint64_ids_beyond_signed_range_are_preserved(self):
        if not hasattr(self.gdal, "GDT_UInt64"):
            self.skipTest("GDAL lacks UInt64 support")
        values = self.np.array([[0, 2**63 + 1], [2**64 - 1, 2**63 + 1]], dtype=self.np.uint64)
        path, source = self.raster(values, dtype=self.gdal.GDT_UInt64)
        artifact = self.convert(path, source)
        self.assertEqual(artifact.instance_id_map, ((2**63 + 1, 1), (2**64 - 1, 2)))
        with self.np.load(artifact.path) as labels:
            self.np.testing.assert_array_equal(labels["mask"], [[0, 1], [2, 1]])

    def test_semantic_preserves_class_values_without_compaction(self):
        values = self.np.array([[0, 12], [38, 90]], dtype=self.np.int32)
        path, source = self.raster(values)
        artifact = self.convert(path, source, kind="semantic")
        self.assertEqual(artifact.path.suffix, ".npy")
        self.assertFalse(artifact.instance_id_map)
        self.np.testing.assert_array_equal(self.np.load(artifact.path), values)

    def test_invalid_encodings_and_nodata_fail_before_staging(self):
        for kwargs, values in (({}, [[0, -1], [0, 1]]),
                               ({"nodata": -32768}, [[0, -32768], [0, 1]]),
                               ({"dtype": self.gdal.GDT_Float32}, [[0, 1.5], [0, 1]]),
                               ({"bands": 2}, [[0, 1], [0, 1]])):
            with self.subTest(kwargs=kwargs):
                path, source = self.raster(values, **kwargs)
                with self.assertRaises(LabelMismatchError):
                    self.convert(path, source)
                self.assertFalse((self.root / "work").exists())

    def test_incomplete_coverage_rejected(self):
        path, _ = self.raster([[0, 1], [0, 1]])
        with self.assertRaises(LabelMismatchError):
            self.convert(path, grid(4, 3))

    def test_multiple_source_blocks(self):
        values = self.np.zeros((3, 520), dtype=self.np.int32)
        values[1, 0] = values[1, 519] = 900
        path, source = self.raster(values)
        artifact = self.convert(path, source)
        with self.np.load(artifact.path) as labels:
            self.np.testing.assert_array_equal(labels["bboxes"], [[0, 1, 520, 1]])


if __name__ == "__main__":
    unittest.main()
