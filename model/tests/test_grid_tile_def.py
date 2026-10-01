from pathlib import Path
import importlib.util
import tempfile
import unittest


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for polar geometry tests")
class PolarTileDefinitionTestCase(unittest.TestCase):
    def definition(self, grid_id="LPS_N", zoom_level=1):
        from lfm.model.grid_tile_def import tile_definition_for_grid

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
        tile_def = self.definition("LPS_N", 4)
        east = tile_def.getOverlappingTiles(85, 178, 84, 180)
        west = tile_def.getOverlappingTiles(85, -180, 84, -178)
        combined = east + west

        self.assertTrue(east)
        self.assertTrue(west)
        self.assertLessEqual(len(set(combined)), len(combined))
        self.assertIn(tile_def.llToTileIndex(84.5, 179), set(combined))
        self.assertIn(tile_def.llToTileIndex(84.5, -179), set(combined))

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
        from lfm.model.grid_tile_def import tile_definition_for_grid

        with self.assertRaisesRegex(KeyError, "LPS_N.*zoom 16"):
            tile_definition_for_grid("LPS_N", 16)
        tile_def = self.definition("LPS_S", 1)
        for tile_x, tile_y in ((-1, 0), (0, -1), (2, 0), (0, 2)):
            with self.subTest(tile_x=tile_x, tile_y=tile_y):
                with self.assertRaises(IndexError):
                    tile_def.getTileBbox(tile_x, tile_y)

    def test_written_polar_cube_has_exact_grid_and_geotransform(self):
        import numpy as np
        from osgeo import gdal

        from lfm.model.raster_cube import WarpedBand, write_tile_cube
        from lfm.model.tiling_config import TileSourceConfig

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
            self.assertTrue(dataset.GetSpatialRef().IsSame(tile_def.srs))
            self.assertEqual(record.grid_id, "LPS_S")
            self.assertTrue(record.crs_wkt)
            dataset = None


if __name__ == "__main__":
    unittest.main()
