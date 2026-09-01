from pathlib import Path
import unittest

from lfm.model.tiling_config import (
    BandNoDataOverride,
    TileConfig,
    TileSourceConfig,
    tile_config_from_dict,
)


class TileConfigTestCase(unittest.TestCase):
    def source(self, **overrides) -> TileSourceConfig:
        values = {
            "name": "wac",
            "data_dir": Path("/data/wac"),
            "index_path": Path("/data/wac/index.shp"),
            "selection_mode": "product_id",
        }
        values.update(overrides)
        return TileSourceConfig(**values)

    def test_source_normalizes_paths_and_sequences(self):
        source = self.source(
            band_names=["vis_1", "vis_2"],
            source_nodata=-9999,
            output_nodata=-32768,
        )

        self.assertEqual(source.data_dir, Path("/data/wac"))
        self.assertEqual(source.index_path, Path("/data/wac/index.shp"))
        self.assertEqual(source.band_names, ("vis_1", "vis_2"))
        self.assertEqual(source.source_nodata, -9999.0)

    def test_source_rejects_invalid_index_suffix(self):
        with self.assertRaisesRegex(ValueError, "index_path"):
            self.source(index_path=Path("/data/wac/index.geojson"))

    def test_source_rejects_both_band_selectors(self):
        with self.assertRaisesRegex(ValueError, "band_names or band_indices"):
            self.source(band_names=("vis",), band_indices=(1,))

    def test_source_rejects_zero_based_band_indices(self):
        with self.assertRaisesRegex(ValueError, "1-based"):
            self.source(band_indices=(0,))

    def test_source_rejects_unknown_selection_mode(self):
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            self.source(selection_mode="dynamic")

    def test_source_rejects_unknown_resampling(self):
        with self.assertRaisesRegex(ValueError, "resampling"):
            self.source(resampling="mode")

    def test_source_finds_band_nodata_override(self):
        override = BandNoDataOverride(
            band_name="delta_cpr",
            output_value=-3.4e38,
            preserve_source=True,
        )
        source = self.source(band_nodata_overrides=(override,))

        self.assertIs(source.nodata_override_for("delta_cpr"), override)
        self.assertIsNone(source.nodata_override_for("vis_1"))

    def test_tile_config_requires_unique_sources(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            TileConfig(
                output_dir=Path("/output"),
                zoom_level=5,
                sources=(self.source(), self.source()),
            )

    def test_tile_config_requires_positive_zoom(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            TileConfig(
                output_dir=Path("/output"),
                zoom_level=0,
                sources=(self.source(),),
            )

    def test_tile_config_source_lookup(self):
        source = self.source()
        config = TileConfig(Path("/output"), 5, (source,))

        self.assertIs(config.source("wac"), source)
        with self.assertRaisesRegex(KeyError, "nac"):
            config.source("nac")

    def test_plain_dictionary_constructor(self):
        config = tile_config_from_dict(
            {
                "output_dir": "/output/cubes",
                "zoom_level": 5,
                "sources": {
                    "wac": {
                        "data_dir": "/data/wac",
                        "index": {
                            "path": "/data/wac/index.gpkg",
                            "layer": "wac_tiles",
                            "location_field": "raster_path",
                        },
                        "selection_mode": "product_id",
                        "bands": {"indices": [1, 2, 3, 4, 5, 6, 7]},
                        "nodata": {"output_value": -3.4e38},
                    },
                    "static": {
                        "data_dir": "/data/static",
                        "index": {"path": "/data/static/index.shp"},
                        "selection_mode": "all_intersecting",
                        "bands": {"names": ["elevation", "slope"]},
                        "resampling": "nearest",
                        "nodata": {
                            "output_value": -32768,
                            "band_overrides": {
                                "delta_cpr": {
                                    "preserve_source": True,
                                }
                            },
                        },
                    },
                },
            }
        )

        self.assertEqual(config.output_dir, Path("/output/cubes"))
        self.assertEqual(config.zoom_level, 5)
        self.assertEqual([source.name for source in config.sources], ["wac", "static"])
        self.assertEqual(config.source("wac").band_indices, tuple(range(1, 8)))
        self.assertEqual(config.source("static").resampling, "nearest")
        self.assertTrue(
            config.source("static")
            .nodata_override_for("delta_cpr")
            .preserve_source
        )

    def test_dictionary_constructor_rejects_missing_source_index(self):
        with self.assertRaisesRegex(KeyError, "index"):
            tile_config_from_dict(
                {
                    "output_dir": "/output",
                    "zoom_level": 5,
                    "sources": {"wac": {"data_dir": "/data/wac"}},
                }
            )

    def test_dictionary_constructor_rejects_unknown_options(self):
        with self.assertRaisesRegex(TypeError, "Unknown tile config"):
            tile_config_from_dict(
                {
                    "output_dir": "/output",
                    "zoom_level": 5,
                    "sources": {},
                    "unexpected": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
