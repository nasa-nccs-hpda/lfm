from pathlib import Path
import unittest

from lfm.model.tiling_results import (
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
        self.assertEqual(error.completed_records, (completed,))


if __name__ == "__main__":
    unittest.main()
