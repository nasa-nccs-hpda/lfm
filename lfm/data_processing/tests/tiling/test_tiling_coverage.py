"""Known coverage gaps retain channels; broken inventories still fail."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


HAS_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "osgeo"))


@unittest.skipUnless(HAS_DEPS, "GDAL and NumPy required")
class TilingCoverageTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import gdal, ogr, osr
        from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt
        from lfm.data_processing.tiling.tiling_config import TileConfig, TileSourceConfig
        from lfm.data_processing.tiling.configured_tiler import ConfiguredTiler
        from lfm.data_processing.tiling.grid_registry import GeographicCoverage

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.np, self.gdal = np, gdal
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(load_lunar_geographic_wkt())
        self.srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.index = self.root / "index.gpkg"
        ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(self.index))
        layer = ds.CreateLayer("rasters", self.srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("location", ogr.OFTString))
        for name, left in (("elevation", 0), ("radar", 10)):
            path = self.root / f"{name}.tif"
            raster = gdal.GetDriverByName("GTiff").Create(str(path), 4, 2, 1, gdal.GDT_Float32)
            raster.SetProjection(self.srs.ExportToWkt())
            raster.SetGeoTransform((left, 1, 0, 2, 0, -1))
            band = raster.GetRasterBand(1)
            band.SetDescription(name)
            band.SetNoDataValue(-32768.)
            band.WriteArray(np.full((2, 4), 7., dtype=np.float32))
            band = raster = None
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("location", path.name)
            feature.SetGeometry(ogr.CreateGeometryFromWkt(
                f"POLYGON (({left} 0, {left+4} 0, {left+4} 2, {left} 2, {left} 0))"))
            layer.CreateFeature(feature)
            feature = None
        layer = ds = None
        self.source = TileSourceConfig("static", self.root, self.index,
            selection_mode="all_intersecting", band_names=("radar", "elevation"), output_nodata=-32768.)
        self.tiler = ConfiguredTiler(TileConfig(self.root / "out", 5, (self.source,)))
        self.tile = SimpleNamespace(tileWidth=2, tileHeight=2, cellSize=1., srs=self.srs,
            validateTileIndex=lambda x, y: None,
            getTileBbox=lambda x, y: (x * 2., 2., x * 2. + 2., 0.),
            geographic_query_envelopes=lambda x, y: (GeographicCoverage(
                south=0, west=x*2., north=2, east=x*2.+2),),
            getOverlappingTiles=lambda *args: [(1, 0), (0, 0)])

    def test_outside_indexed_band_fills_and_later_tiles_continue(self):
        from lfm.data_processing.tiling.raster_cube import indexed_band_catalog

        before = hashlib.sha256(self.index.read_bytes()).hexdigest()
        part = SimpleNamespace(grid_id="42N", ul_lat=2, ul_lon=0, lr_lat=0, lr_lon=4)
        with patch("lfm.data_processing.tiling.configured_tiler.tile_definition_for_grid", return_value=self.tile), \
             patch("lfm.data_processing.tiling.configured_tiler.route_aoi", return_value=(part,)), \
             patch("lfm.data_processing.tiling.configured_tiler.indexed_band_catalog", wraps=indexed_band_catalog) as catalog, \
             self.assertLogs("lfm.data_processing.tiling.raster_cube", level="WARNING"):
            records = self.tiler.run_aoi(2, 0, 0, 4)
        self.assertEqual(catalog.call_count, 1)
        self.assertEqual([r.tile_x for r in records], [0, 1])
        for record in records:
            self.assertEqual(record.band_names, ("radar", "elevation"))
            self.assertEqual(record.nodata_values, (-32768., -32768.))
            ds = self.gdal.Open(str(record.path))
            self.assertTrue(self.np.all(ds.GetRasterBand(1).ReadAsArray() == -32768))
            self.assertFalse(ds.GetRasterBand(1).GetMaskBand().ReadAsArray().any())
            self.assertTrue(self.np.all(ds.GetRasterBand(2).ReadAsArray() == 7))
            ds = None
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).hexdigest(), before)

    def test_all_declared_sources_outside_tile_still_write_complete_cube(self):
        with patch("lfm.data_processing.tiling.configured_tiler.tile_definition_for_grid", return_value=self.tile), \
             self.assertLogs("lfm.data_processing.tiling.raster_cube", level="WARNING"):
            record = self.tiler.run_tile_index(20, 0, "42N")[0]
        ds = self.gdal.Open(str(record.path))
        self.assertEqual(ds.RasterCount, 2)
        self.assertTrue(self.np.all(ds.ReadAsArray() == -32768))
        ds = None

    def test_intersecting_all_nodata_band_is_not_discarded(self):
        from lfm.data_processing.tiling.raster_cube import warp_source_to_tile
        from dataclasses import replace

        path = self.root / "elevation.tif"
        ds = self.gdal.Open(str(path), self.gdal.GA_Update)
        ds.GetRasterBand(1).Fill(-32768.)
        ds = None
        with self.assertLogs("lfm.data_processing.tiling.raster_cube", level="WARNING"):
            bands = warp_source_to_tile(replace(self.source, band_names=("elevation",)),
                [path], tile_def=self.tile, bounds=(0, 2, 2, 0))
        self.assertEqual([b.name for b in bands], ["elevation"])
        self.assertTrue(self.np.all(bands[0].pixels == -32768))

    def test_unindexed_name_is_not_treated_as_coverage(self):
        from dataclasses import replace
        from lfm.data_processing.tiling.configured_tiler import ConfiguredTiler
        from lfm.data_processing.tiling.tiling_config import TileConfig
        from lfm.data_processing.tiling.tiling_results import TileSourceError

        tiler = ConfiguredTiler(TileConfig(self.root / "bad", 5, (
            replace(self.source, band_names=("typo",)),)))
        with patch("lfm.data_processing.tiling.configured_tiler.tile_definition_for_grid", return_value=self.tile):
            with self.assertRaisesRegex(TileSourceError, "unknown or unindexed"):
                tiler.run_tile_index(0, 0, "42N")

    def test_empty_duplicate_does_not_replace_valid_band_or_hide_ambiguity(self):
        from dataclasses import replace
        from lfm.data_processing.tiling.raster_cube import WarpedBand, _select_bands

        source = replace(self.source, band_names=("radar",))
        empty = WarpedBand("radar", self.np.full((2, 2), -32768.), -32768., -32768.)
        valid = WarpedBand("radar", self.np.ones((2, 2)), -32768., -32768.)
        for bands in ([empty, valid], [valid, empty]):
            self.assertIs(_select_bands(source, bands)[0], valid)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _select_bands(source, [valid, valid, empty])

    def test_unreadable_outside_indexed_raster_still_fails(self):
        from lfm.data_processing.tiling.tiling_results import TileSourceError

        # A valid index points to a directory rather than a readable TIFF.
        path = self.root / "radar.tif"
        path.rename(self.root / "radar.saved.tif")
        path.mkdir()
        with patch("lfm.data_processing.tiling.configured_tiler.tile_definition_for_grid", return_value=self.tile):
            with self.assertRaises(TileSourceError):
                self.tiler.run_tile_index(0, 0, "42N")


if __name__ == "__main__":
    unittest.main()
