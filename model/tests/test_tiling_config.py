from pathlib import Path
import unittest

from lfm.model.tiling_config import (
    BandNoDataOverride,
    TileConfig,
    TileSourceConfig,
    tile_config_from_dict,
)
from lfm.model.source_modes import compose_tile_sources


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

    def test_source_accepts_custom_product_id_resolver(self):
        resolver = lambda path: path.stem.split("_")[0]

        source = self.source(product_id_resolver=resolver)

        self.assertIs(source.product_id_resolver, resolver)

    def test_source_rejects_noncallable_product_id_resolver(self):
        with self.assertRaisesRegex(TypeError, "product_id_resolver"):
            self.source(product_id_resolver="filename")

    def test_source_requires_bilinear_resampling(self):
        for method in ("nearest", "cubic", "average", "mode"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "bilinear"):
                    self.source(resampling=method)

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
                        "product_id_resolver": lambda path: path.name.split(".")[0],
                        "bands": {"indices": [1, 2, 3, 4, 5, 6, 7]},
                        "nodata": {"output_value": -3.4e38},
                    },
                    "static": {
                        "data_dir": "/data/static",
                        "index": {"path": "/data/static/index.shp"},
                        "selection_mode": "all_intersecting",
                        "bands": {"names": ["elevation", "slope"]},
                        "resampling": "bilinear",
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
        self.assertEqual(
            config.source("wac").product_id_resolver(
                Path("M100.prj.vis.mos.tif")
            ),
            "M100",
        )
        self.assertEqual(config.source("static").resampling, "bilinear")
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


class TileSourceModeTestCase(unittest.TestCase):
    def source(self, name, *, selection_mode="product_id", required=True):
        return TileSourceConfig(
            name=name,
            data_dir=Path(f"/data/{name}"),
            index_path=Path(f"/data/{name}/index.gpkg"),
            selection_mode=selection_mode,
            required=required,
        )

    def test_defaults_enable_both_and_order_dynamic_before_static(self):
        dynamic_first = self.source("wac")
        dynamic_second = self.source("nac", required=False)
        static = self.source("static", selection_mode="all_intersecting")

        sources = compose_tile_sources(
            dynamic_sources=(dynamic_first, dynamic_second),
            static_sources=(static,),
        )

        self.assertEqual(sources, (dynamic_first, dynamic_second, static))
        self.assertTrue(sources[0].required)
        self.assertFalse(sources[1].required)

    def test_rejects_disabling_both_source_classes(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            compose_tile_sources(
                include_dynamic=False,
                include_static=False,
            )

    def test_rejects_non_boolean_inclusion_controls(self):
        with self.assertRaisesRegex(TypeError, "include_dynamic"):
            compose_tile_sources(
                dynamic_sources=(self.source("wac"),),
                include_dynamic=1,
                include_static=False,
            )

    def test_each_enabled_class_requires_a_source(self):
        dynamic = self.source("wac")
        static = self.source("static", selection_mode="all_intersecting")
        with self.assertRaisesRegex(ValueError, "dynamic source"):
            compose_tile_sources(
                static_sources=(static,),
                include_static=True,
            )
        with self.assertRaisesRegex(ValueError, "static source"):
            compose_tile_sources(
                dynamic_sources=(dynamic,),
                include_dynamic=True,
            )

    def test_disabled_source_collection_is_not_iterated_or_validated(self):
        class DisabledSources:
            def __iter__(self):
                raise AssertionError("disabled sources must not be inspected")

        dynamic = self.source("wac")
        sources = compose_tile_sources(
            dynamic_sources=(dynamic,),
            static_sources=DisabledSources(),
            include_static=False,
        )

        self.assertEqual(sources, (dynamic,))

    def test_static_only_accepts_contextual_sources(self):
        static = self.source(
            "static",
            selection_mode="all_intersecting",
            required=False,
        )

        sources = compose_tile_sources(
            static_sources=(static,),
            include_dynamic=False,
        )

        self.assertEqual(sources, (static,))
        self.assertFalse(sources[0].required)

    def test_enabled_static_sources_cannot_be_product_scoped(self):
        with self.assertRaisesRegex(ValueError, "all_intersecting"):
            compose_tile_sources(
                static_sources=(self.source("static"),),
                include_dynamic=False,
            )

    def test_duplicate_names_across_enabled_classes_are_rejected(self):
        dynamic = self.source("shared")
        static = self.source("shared", selection_mode="all_intersecting")

        with self.assertRaisesRegex(ValueError, "unique"):
            compose_tile_sources(
                dynamic_sources=(dynamic,),
                static_sources=(static,),
            )


if __name__ == "__main__":
    unittest.main()
