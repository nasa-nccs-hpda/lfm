from pathlib import Path
import unittest

from lfm.model.vector_index_builder import VectorIndexBuildConfig


class VectorIndexBuildConfigTestCase(unittest.TestCase):
    def test_accepts_shapefile_and_geopackage(self):
        shapefile = VectorIndexBuildConfig(
            data_dir=Path("/data/wac"),
            index_path=Path("/data/wac/index.shp"),
        )
        geopackage = VectorIndexBuildConfig(
            data_dir=Path("/data/nac"),
            index_path=Path("/data/nac/index.gpkg"),
            layer_name="nac",
        )

        self.assertEqual(shapefile.index_path.suffix, ".shp")
        self.assertEqual(geopackage.layer_name, "nac")

    def test_rejects_unsupported_index_format(self):
        with self.assertRaisesRegex(ValueError, ".shp or .gpkg"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                index_path=Path("/data/wac/index.geojson"),
            )


if __name__ == "__main__":
    unittest.main()
