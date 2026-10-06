"""Focused checks for the real-data smoke test's automatic AOI selection."""

import importlib.util
from lfm.data_processing._paths import REPO_ROOT
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = REPO_ROOT / "scripts/python/all_tasks/validate_aoi_chip_creation.py"
SPEC = importlib.util.spec_from_file_location("aoi_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
# The standalone CLI prepends the checkout for notebook-style imports. Do not
# leak that into suite discovery: spawned tests import from the checkout parent.
original_sys_path = sys.path[:]
try:
    SPEC.loader.exec_module(smoke)
finally:
    sys.path[:] = original_sys_path


class SmokeAOITestCase(unittest.TestCase):
    def test_full_margin_and_half_crater(self):
        full, clipped = smoke.smoke_aoi_bounds((20, 22, -6, -4))
        self.assertEqual(full, [-3.6, 19.6, -6.4, 22.4])
        self.assertEqual(clipped, [-3.6, 19.6, -6.4, 21])

    def test_invalid_or_unsupported_extent(self):
        for bounds in ((20, 20, -6, -4), (20, 22, -4, -6),
                       (-179, 179, -6, -4), (20, 22, 81, 83),
                       (20, float("nan"), -6, -4)):
            with self.subTest(bounds=bounds), self.assertRaises(AssertionError):
                smoke.smoke_aoi_bounds(bounds)

    @unittest.skipUnless(importlib.util.find_spec("osgeo"), "GDAL unavailable")
    def test_largest_crater_selected_read_only(self):
        from osgeo import ogr, osr
        from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.gpkg"
            srs = osr.SpatialReference()
            srs.ImportFromWkt(load_lunar_geographic_wkt())
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
            layer = ds.CreateLayer("craters", srs, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn("crater_id", ogr.OFTInteger))
            for instance, size in ((38, .1), (12, .2)):
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("crater_id", instance)
                feature.SetGeometry(ogr.CreateGeometryFromWkt(
                    f"POLYGON ((20 -6, {20+size} -6, {20+size} {-6+size}, 20 {-6+size}, 20 -6))"))
                layer.CreateFeature(feature)
                feature = None
            layer = ds = None
            before = smoke.sha256(path)
            # Only the script call needs its standalone import convention.
            original_sys_path = sys.path[:]
            try:
                sys.path.insert(0, str(SCRIPT.parents[3]))
                aois, selected = smoke.derive_smoke_aois(path, "craters")
            finally:
                sys.path[:] = original_sys_path
            self.assertEqual(selected, 12)
            self.assertEqual(len(aois), 2)
            self.assertGreater(aois[0][3], aois[1][3])
            self.assertEqual(smoke.sha256(path), before)


if __name__ == "__main__":
    unittest.main()
