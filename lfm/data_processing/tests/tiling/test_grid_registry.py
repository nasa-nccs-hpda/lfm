from pathlib import Path
import shutil
import tempfile
import unittest

from lfm.data_processing.tiling.grid_registry import (
    GridFamily,
    GridRegistry,
    default_grid_registry,
    load_grid_definition,
)


class GridRegistryTestCase(unittest.TestCase):
    def setUp(self):
        default_grid_registry.cache_clear()
        self.registry = default_grid_registry()

    def test_repository_inventory_has_90_ltm_and_two_polar_grids(self):
        self.assertEqual(len(self.registry), 92)
        self.assertEqual(
            len(self.registry.by_family(GridFamily.LTM)),
            90,
        )
        self.assertEqual(
            tuple(
                item.grid_id
                for item in self.registry.by_family(GridFamily.LPS_N)
            ),
            ("LPS_N",),
        )
        self.assertEqual(
            tuple(
                item.grid_id
                for item in self.registry.by_family(GridFamily.LPS_S)
            ),
            ("LPS_S",),
        )

    def test_numbered_ltm_metadata_is_grid_neutral(self):
        definition = self.registry["42N"]

        self.assertIs(definition.family, GridFamily.LTM)
        self.assertEqual(
            (
                definition.geographic_coverage.south,
                definition.geographic_coverage.west,
                definition.geographic_coverage.north,
                definition.geographic_coverage.east,
            ),
            (0.0, 148.0, 82.0, 156.0),
        )
        self.assertEqual(definition.zoom_levels, tuple(range(1, 27)))
        matrix = definition.matrix(5)
        self.assertEqual((matrix.tile_width, matrix.tile_height), (512, 512))
        self.assertEqual((matrix.matrix_width, matrix.matrix_height), (32, 64))
        self.assertIn("Lunar Transverse Mercator", definition.crs_wkt)

    def test_polar_metadata_uses_canonical_ids_and_families(self):
        north = self.registry["LPS_N"]
        south = self.registry["LPS_S"]

        self.assertIs(north.family, GridFamily.LPS_N)
        self.assertIs(south.family, GridFamily.LPS_S)
        self.assertEqual(
            (north.geographic_coverage.south, north.geographic_coverage.north),
            (80.0, 90.0),
        )
        self.assertEqual(
            (south.geographic_coverage.south, south.geographic_coverage.north),
            (-90.0, -80.0),
        )
        self.assertEqual(north.zoom_levels, tuple(range(1, 16)))
        self.assertEqual(south.matrix(1).matrix_width, 2)

    def test_registry_order_is_stable_and_explicit(self):
        self.assertEqual(self.registry.grid_ids[:3], ("1N", "2N", "3N"))
        self.assertEqual(self.registry.grid_ids[44:47], ("45N", "1S", "2S"))
        self.assertEqual(self.registry.grid_ids[-2:], ("LPS_N", "LPS_S"))
        self.assertEqual(
            default_grid_registry().grid_ids,
            self.registry.grid_ids,
        )

    def test_unknown_grid_and_zoom_are_contextual(self):
        with self.assertRaisesRegex(KeyError, "Unknown lunar grid ID"):
            self.registry["N"]
        with self.assertRaisesRegex(KeyError, "42N.*zoom 99"):
            self.registry["42N"].matrix(99)

    def test_repository_inventory_validation_rejects_partial_directory(self):
        source_path = GridRegistry.DEFAULT_DIRECTORY / "tms_LTM_42NRG.json"
        definition = load_grid_definition(source_path)
        partial = GridRegistry((definition,))
        self.assertEqual(partial.grid_ids, ("42N",))

        with tempfile.TemporaryDirectory() as directory:
            shutil.copy2(source_path, Path(directory) / source_path.name)
            with self.assertRaisesRegex(ValueError, "90 numbered LTM"):
                GridRegistry.from_directory(
                    directory,
                    require_repository_inventory=True,
                )


if __name__ == "__main__":
    unittest.main()
