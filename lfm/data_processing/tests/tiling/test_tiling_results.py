from pathlib import Path
import unittest

from lfm.data_processing.tiling.tiling_results import (
    MissingRequiredSourceError,
    TileCubeRecord,
    tile_cube_filename,
)


class TilingResultsTestCase(unittest.TestCase):
    def record(self, **overrides) -> TileCubeRecord:
        values = {
            "source_name": "wac",
            "zone": "42N",
            "zoom_level": 5,
            "tile_x": 1,
            "tile_y": 63,
            "product_id": "M100",
            "path": Path("/output/cube.tif"),
            "band_names": ("vis_1", "vis_2"),
            "crs_wkt": "PROJCRS[test]",
            "nodata_values": (-9999.0, -9999.0),
        }
        values.update(overrides)
        return TileCubeRecord(**values)

    def test_generic_product_filename(self):
        filename = tile_cube_filename(
            source_name="LRO WAC",
            zone="42N",
            zoom_level=5,
            tile_x=1,
            tile_y=63,
            product_id="M100/unsafe",
        )

        self.assertEqual(
            filename,
            "Cube-LRO-WAC-LTM42N_Zoom-5_Tile-1-63_Product-M100-unsafe.tif",
        )

    def test_contextual_filename_omits_product(self):
        filename = tile_cube_filename(
            source_name="static",
            zone="42N",
            zoom_level=5,
            tile_x=1,
            tile_y=63,
        )

        self.assertEqual(filename, "Cube-static-LTM42N_Zoom-5_Tile-1-63.tif")

    def test_polar_filename_uses_canonical_grid_without_ltm_prefix(self):
        filename = tile_cube_filename(
            source_name="wac",
            grid_id="LPS_N",
            zoom_level=4,
            tile_x=7,
            tile_y=8,
            product_id="M100",
        )

        self.assertEqual(
            filename,
            "Cube-wac-LPS_N_Zoom-4_Tile-7-8_Product-M100.tif",
        )

    def test_zone_and_grid_id_must_not_conflict(self):
        with self.assertRaisesRegex(ValueError, "different grids"):
            tile_cube_filename(
                source_name="wac",
                zone="42N",
                grid_id="LPS_N",
                zoom_level=4,
                tile_x=1,
                tile_y=2,
            )

    def test_record_exposes_grid_id_alias(self):
        self.assertEqual(self.record(zone="LPS_S").grid_id, "LPS_S")

    def test_record_rejects_noncanonical_grid_id(self):
        with self.assertRaisesRegex(ValueError, "Unsupported lunar grid ID"):
            self.record(zone="N")

    def test_record_requires_matching_band_metadata(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            self.record(nodata_values=(None,))

    def test_source_error_retains_completed_records(self):
        completed = self.record()
        error = MissingRequiredSourceError(
            "missing static",
            source_name="static",
            zone="42N",
            tile_x=1,
            tile_y=63,
            completed_records=(completed,),
        )

        self.assertEqual(error.source_name, "static")
        self.assertEqual(error.grid_id, "42N")
        self.assertEqual(error.completed_records, (completed,))

    def test_source_error_retains_product_id(self):
        error = MissingRequiredSourceError(
            "missing product",
            source_name="wac",
            zone="42N",
            tile_x=1,
            tile_y=63,
            product_id="M100",
        )

        self.assertEqual(error.product_id, "M100")


if __name__ == "__main__":
    unittest.main()
