import unittest

from lfm.data_processing.tiling.tile_matrix import (
    TileMatrixGeometry,
    candidate_tile_range,
    projected_to_tile_index,
    tile_bounds,
    validate_tile_index,
)


class TileMatrixGeometryTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix = TileMatrixGeometry(
            origin_x=100.0,
            origin_y=500.0,
            cell_size=2.0,
            tile_width=10,
            tile_height=10,
            matrix_width=4,
            matrix_height=5,
        )

    def test_bounds_use_top_left_origin(self):
        self.assertEqual(
            tile_bounds(self.matrix, 2, 3),
            (140.0, 440.0, 160.0, 420.0),
        )

    def test_projected_point_resolves_and_clips_to_matrix(self):
        self.assertEqual(
            projected_to_tile_index(self.matrix, 145.0, 435.0),
            (2, 3),
        )
        self.assertIsNone(projected_to_tile_index(self.matrix, 99.0, 435.0))

    def test_candidate_range_is_clipped_and_inclusive(self):
        self.assertEqual(
            candidate_tile_range(
                self.matrix,
                min_x=90.0,
                min_y=410.0,
                max_x=145.0,
                max_y=510.0,
            ),
            (0, 2, 0, 4),
        )

    def test_explicit_address_validation_is_contextual(self):
        validate_tile_index(
            self.matrix,
            3,
            4,
            grid_id="LPS_N",
            zoom_level=1,
        )
        with self.assertRaisesRegex(IndexError, "LPS_N.*tile_x 4"):
            validate_tile_index(
                self.matrix,
                4,
                0,
                grid_id="LPS_N",
                zoom_level=1,
            )
        with self.assertRaisesRegex(TypeError, "tile_y must be an integer"):
            validate_tile_index(
                self.matrix,
                0,
                True,
                grid_id="LPS_N",
                zoom_level=1,
            )


if __name__ == "__main__":
    unittest.main()
