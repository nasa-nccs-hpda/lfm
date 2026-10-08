from pathlib import Path
import importlib.util
import tempfile
import unittest


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for polar geometry tests")
class PolarTileDefinitionTestCase(unittest.TestCase):
    def definition(self, grid_id="LPS_N", zoom_level=1):
        from lfm.data_processing.tiling.grid_tile_def import tile_definition_for_grid

        return tile_definition_for_grid(grid_id, zoom_level)

    def test_polar_matrix_metadata_and_projected_bounds(self):
        tile_def = self.definition("LPS_N", 1)

        self.assertEqual(tile_def.grid_id, "LPS_N")
        self.assertEqual((tile_def.matrixWidth, tile_def.matrixHeight), (2, 2))
        self.assertEqual((tile_def.tileWidth, tile_def.tileHeight), (512, 512))
        self.assertAlmostEqual(tile_def.cellSize, 590.1983874448465)
        ulx, uly, lrx, lry = tile_def.getTileBbox(1, 1)
        self.assertAlmostEqual(ulx, 500000.0, places=6)
        self.assertAlmostEqual(uly, 500000.0, places=6)
        self.assertAlmostEqual(lrx, 802181.5743717614, places=6)
        self.assertAlmostEqual(lry, 197818.4256282386, places=6)

    def test_both_poles_transform_to_false_origin_and_round_trip(self):
        for grid_id, latitude in (("LPS_N", 90.0), ("LPS_S", -90.0)):
            with self.subTest(grid_id=grid_id):
                tile_def = self.definition(grid_id, 4)
                x, y = tile_def.latLonToProjected(latitude, 137.0)
                self.assertAlmostEqual(x, 500000.0, places=6)
                self.assertAlmostEqual(y, 500000.0, places=6)
                round_trip_lat, _ = tile_def.projectedToLatLon(x, y)
                self.assertAlmostEqual(round_trip_lat, latitude, places=8)

    def test_polar_point_and_aoi_resolve_valid_tiles(self):
        for grid_id, north, south in (
            ("LPS_N", 85.0, 84.0),
            ("LPS_S", -84.0, -85.0),
        ):
            with self.subTest(grid_id=grid_id):
                tile_def = self.definition(grid_id, 4)
                point = tile_def.llToTileIndex((north + south) / 2, 11.0)
                self.assertIsNotNone(point)
                tiles = tile_def.getOverlappingTiles(
                    north,
                    10.0,
                    south,
                    12.0,
                )
                self.assertIn(point, tiles)
                self.assertEqual(
                    tiles,
                    sorted(set(tiles), key=lambda item: (item[1], item[0])),
                )
                self.assertTrue(
                    all(
                        0 <= x < tile_def.matrixWidth
                        and 0 <= y < tile_def.matrixHeight
                        for x, y in tiles
                    )
                )

    def test_antimeridian_query_parts_produce_deduplicable_addresses(self):
        for zone, north, south in (("LPS_N", 85, 84), ("LPS_S", -84, -85)):
            with self.subTest(zone=zone):
                tile_def = self.definition(zone, 4)
                east = tile_def.getOverlappingTiles(north, 178, south, 180)
                west = tile_def.getOverlappingTiles(north, -180, south, -178)
                combined = set(east + west)
                self.assertTrue(east)
                self.assertTrue(west)
                latitude = (north + south) / 2
                self.assertIn(tile_def.llToTileIndex(latitude, 179), combined)
                self.assertIn(tile_def.llToTileIndex(latitude, -179), combined)
                self.assertLess(len(combined), 20)  # Small seam AOI, not global coverage.

    def test_full_polar_caps_are_projected_as_finite_circular_coverage(self):
        cases = (
            ("LPS_N", 90.0, 82.0),
            ("LPS_S", -82.0, -90.0),
        )
        for grid_id, north, south in cases:
            with self.subTest(grid_id=grid_id):
                tile_def = self.definition(grid_id, 1)
                tiles = tile_def.getOverlappingTiles(
                    north,
                    -180.0,
                    south,
                    180.0,
                )
                self.assertEqual(
                    tiles,
                    [(0, 0), (1, 0), (0, 1), (1, 1)],
                )

    def test_tile_query_envelope_is_seam_safe_and_pole_conservative(self):
        tile_def = self.definition("LPS_N", 1)
        envelopes = tile_def.geographic_query_envelopes(1, 1)

        self.assertEqual(len(envelopes), 1)
        self.assertEqual((envelopes[0].west, envelopes[0].east), (-180, 180))
        self.assertEqual(envelopes[0].north, 90)

    def test_invalid_polar_zoom_and_addresses_are_rejected(self):
        from lfm.data_processing.tiling.grid_tile_def import tile_definition_for_grid

        with self.assertRaisesRegex(KeyError, "LPS_N.*zoom 16"):
            tile_definition_for_grid("LPS_N", 16)
        tile_def = self.definition("LPS_S", 1)
        for tile_x, tile_y in ((-1, 0), (0, -1), (2, 0), (0, 2)):
            with self.subTest(tile_x=tile_x, tile_y=tile_y):
                with self.assertRaises(IndexError):
                    tile_def.getTileBbox(tile_x, tile_y)

    def test_written_polar_cube_has_exact_grid_and_geotransform(self):
        import numpy as np
        from osgeo import gdal, osr

        from lfm.data_processing.tiling.raster_cube import WarpedBand, write_tile_cube
        from lfm.data_processing.tiling.lunar_crs import raster_crs_equivalent
        from lfm.data_processing.tiling.tiling_config import TileSourceConfig

        tile_def = self.definition("LPS_S", 1)
        ulx, uly, _, _ = tile_def.getTileBbox(0, 0)
        source = TileSourceConfig(
            name="dynamic",
            data_dir=Path("/data/dynamic"),
            index_path=Path("/data/dynamic/index.gpkg"),
            required=False,
        )
        pixels = np.ones((512, 512), dtype=np.float32)
        band = WarpedBand("band", pixels, None, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polar.tif"
            record = write_tile_cube(
                path,
                [band],
                source=source,
                product_id=None,
                zone="LPS_S",
                zoom_level=1,
                tile_x=0,
                tile_y=0,
                tile_def=tile_def,
                ulx=ulx,
                uly=uly,
            )
            dataset = gdal.Open(str(path))
            self.assertEqual((dataset.RasterXSize, dataset.RasterYSize), (512, 512))
            self.assertEqual(
                dataset.GetGeoTransform(),
                (ulx, tile_def.cellSize, 0.0, uly, 0.0, -tile_def.cellSize),
            )
            written_srs = dataset.GetSpatialRef()
            self.assertIsNotNone(written_srs)
            written_srs = written_srs.Clone()
            expected_srs = tile_def.srs.Clone()
            # GeoTIFF WKT1 reconstruction can drop custom authorities and
            # represent polar axes with projection-native NORTH directions.
            # LFM raster data still uses traditional easting/northing order,
            # so normalize that runtime mapping and compare the complete
            # projected coordinate-operation signature as a narrow fallback.
            written_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            expected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.assertTrue(
                raster_crs_equivalent(written_srs, expected_srs),
                msg=(
                    "Written polar CRS is not equivalent to the tile grid "
                    "after axis normalization and projected-operation "
                    f"comparison.\nWritten WKT: "
                    f"{written_srs.ExportToWkt()}\nExpected: "
                    f"{expected_srs.ExportToWkt()}\nWritten PROJ.4: "
                    f"{written_srs.ExportToProj4()}\nExpected PROJ.4: "
                    f"{expected_srs.ExportToProj4()}"
                ),
            )
            self.assertEqual(record.grid_id, "LPS_S")
            self.assertTrue(record.crs_wkt)
            dataset = None


if __name__ == "__main__":
    unittest.main()
