from pathlib import Path
import unittest

from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    MixedPercentageNumberSplitConfig,
    NumberSplitConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitCounts,
    SplitPercentages,
    chip_config_from_dict,
    default_split_config,
    split_config_from_dict,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig


class ChipConfigTestCase(unittest.TestCase):
    def source(self, name="wac", **overrides):
        values = {
            "name": name,
            "data_dir": Path(f"/data/{name}"),
            "index_path": Path(f"/data/{name}/index.gpkg"),
            "selection_mode": (
                "product_id" if name in ("wac", "nac") else "all_intersecting"
            ),
        }
        values.update(overrides)
        return TileSourceConfig(**values)

    def group(self, name="wac_grid", zoom=5, sources=None):
        configured_sources = sources or (self.source(),)
        return AcquisitionGroupConfig(
            name=name,
            tile_config=TileConfig(
                output_dir=Path(f"/cubes/{name}"),
                zoom_level=zoom,
                sources=configured_sources,
            ),
        )

    def config(self, **overrides):
        values = {
            "output_root": Path("/dataset"),
            "label_source": Path("/labels"),
            "acquisition_groups": (self.group(),),
            "output_modalities": (
                OutputModalityConfig("wac_grid", "wac", "wac"),
            ),
        }
        values.update(overrides)
        return ChipConfig(**values)

    @staticmethod
    def source_dict(name):
        return {
            "data_dir": f"/data/{name}",
            "index": {"path": f"/data/{name}/index.gpkg"},
            "selection_mode": (
                "product_id" if name in ("wac", "nac") else "all_intersecting"
            ),
        }

    def dictionary_config(self, sources, **overrides):
        first_source = next(iter(sources))
        values = {
            "output_root": "/dataset",
            "label_source": "/labels",
            "acquisition_groups": {
                "sensor_grid": {
                    "sources": sources,
                }
            },
            "output_modalities": [
                {
                    "acquisition_group": "sensor_grid",
                    "source": first_source,
                    "alias": first_source,
                }
            ],
        }
        values.update(overrides)
        return chip_config_from_dict(values)

    def test_chip_config_composes_tiling_groups_and_derives_suffix(self):
        config = self.config(
            common_nodata=-9999,
            output_dtype="FLOAT32",
            sample_limit=10,
            intermediate_retention="always",
        )

        self.assertEqual(config.output_root, Path("/dataset"))
        self.assertEqual(config.intermediate_root, Path("/dataset/.intermediate"))
        self.assertEqual(config.final_output_suffix, "_input_wac_chip.tif")
        self.assertEqual(config.common_nodata, -9999.0)
        self.assertEqual(config.output_dtype, "float32")
        self.assertEqual(config.sample_limit, 10)
        self.assertEqual(config.acquisition_group("wac_grid").tile_config.zoom_level, 5)

    def test_output_modality_qualifies_same_source_in_different_groups(self):
        static = self.source("static")
        groups = (
            self.group("wac_grid", 5, (self.source("wac"), static)),
            self.group("nac_grid", 11, (self.source("nac"), static)),
        )
        config = self.config(
            acquisition_groups=groups,
            output_modalities=(
                OutputModalityConfig("wac_grid", "static", "static_wac"),
                OutputModalityConfig("nac_grid", "static", "static_nac"),
            ),
        )

        self.assertEqual(
            config.final_output_suffix,
            "_input_static_wac_static_nac_chip.tif",
        )

    def test_chip_config_rejects_missing_or_duplicate_output_modalities(self):
        with self.assertRaisesRegex(ValueError, "at least one output modality"):
            self.config(output_modalities=())
        duplicate = OutputModalityConfig("wac_grid", "wac", "alternate")
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            self.config(
                output_modalities=(
                    OutputModalityConfig("wac_grid", "wac", "wac"),
                    duplicate,
                )
            )

    def test_chip_config_rejects_unknown_group_or_source(self):
        with self.assertRaisesRegex(ValueError, "unknown acquisition group"):
            self.config(
                output_modalities=(
                    OutputModalityConfig("missing", "wac", "wac"),
                )
            )
        with self.assertRaisesRegex(ValueError, "unknown source"):
            self.config(
                output_modalities=(
                    OutputModalityConfig("wac_grid", "static", "static"),
                )
            )

    def test_output_modality_validates_bands_and_resampling(self):
        modality = OutputModalityConfig(
            "wac_grid",
            "wac",
            "wac",
            band_indices=(1, 2),
            output_band_names=("vis_1", "vis_2"),
            resampling="nearest",
        )
        self.assertEqual(modality.band_indices, (1, 2))
        with self.assertRaisesRegex(ValueError, "selection length"):
            OutputModalityConfig(
                "wac_grid",
                "wac",
                "wac",
                band_indices=(1, 2),
                output_band_names=("vis",),
            )
        with self.assertRaisesRegex(ValueError, "resampling"):
            OutputModalityConfig("wac_grid", "wac", "wac", resampling="cubic")

    def test_dictionary_constructor_resolves_zoom_defaults(self):
        wac = self.dictionary_config({"wac": self.source_dict("wac")})
        nac = self.dictionary_config({"nac": self.source_dict("nac")})
        custom = self.dictionary_config({"custom": self.source_dict("custom")})

        self.assertEqual(wac.acquisition_groups[0].tile_config.zoom_level, 5)
        self.assertEqual(nac.acquisition_groups[0].tile_config.zoom_level, 11)
        self.assertEqual(custom.acquisition_groups[0].tile_config.zoom_level, 5)

    def test_dictionary_constructor_allows_zoom_override(self):
        config = self.dictionary_config(
            {"nac": self.source_dict("nac")},
            acquisition_groups={
                "sensor_grid": {
                    "zoom_level": 7,
                    "sources": {"nac": self.source_dict("nac")},
                }
            },
        )

        self.assertEqual(config.acquisition_groups[0].tile_config.zoom_level, 7)

    def test_dictionary_constructor_requires_explicit_conflicting_zoom(self):
        with self.assertRaisesRegex(ValueError, "must declare zoom_level"):
            self.dictionary_config(
                {
                    "wac": self.source_dict("wac"),
                    "nac": self.source_dict("nac"),
                }
            )

    def test_dictionary_constructor_matches_tiling_shape(self):
        config = self.dictionary_config(
            {
                "wac": {
                    **self.source_dict("wac"),
                    "bands": {"indices": [1, 2]},
                },
                "static": {
                    **self.source_dict("static"),
                    "bands": {"names": ["elevation"]},
                },
            },
            output_modalities=[
                {
                    "acquisition_group": "sensor_grid",
                    "source": "wac",
                    "alias": "wac",
                    "bands": {"indices": [1, 2]},
                    "output_band_names": ["vis_1", "vis_2"],
                    "resampling": "bilinear",
                },
                {
                    "acquisition_group": "sensor_grid",
                    "source": "static",
                    "alias": "static",
                },
            ],
            output_suffix="_input_custom_chip.tif",
        )

        group = config.acquisition_groups[0]
        self.assertEqual(
            group.tile_config.output_dir,
            Path("/dataset/.intermediate/sensor_grid"),
        )
        self.assertEqual(group.tile_config.source("wac").band_indices, (1, 2))
        self.assertEqual(config.final_output_suffix, "_input_custom_chip.tif")


class SplitConfigTestCase(unittest.TestCase):
    def test_repository_default_split(self):
        config = default_split_config()

        self.assertIsInstance(config, MixedPercentageNumberSplitConfig)
        self.assertEqual(config.fixed_counts, SplitCounts(test=100))
        self.assertEqual(config.fixed_priority, ("test",))
        self.assertEqual(
            config.remaining_percentages,
            SplitPercentages(train=0.9, val=0.1, test=0.0),
        )

    def test_all_split_config_variants(self):
        simple = SimpleSplitConfig(SplitPercentages(0.7, 0.2, 0.1), seed=4)
        mixed = MixedPercentageNumberSplitConfig(
            fixed_counts=SplitCounts(test=25, val=10),
            fixed_priority=("test", "val"),
            remaining_percentages=SplitPercentages(1.0, 0.0, 0.0),
            seed=4,
        )
        numbered = NumberSplitConfig(
            fixed_counts=SplitCounts(train=50, test=20, val=10),
            fixed_priority=("train", "test", "val"),
            remainder_split="train",
            seed=4,
        )

        self.assertEqual(simple.percentages.train, 0.7)
        self.assertEqual(mixed.fixed_priority, ("test", "val"))
        self.assertEqual(numbered.remainder_split, "train")

    def test_percentage_config_rejects_invalid_ratios(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            SplitPercentages(0.8, 0.3, 0.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            SplitPercentages(1.1, 0.0, -0.1)

    def test_number_config_rejects_invalid_counts_priority_and_names(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            SplitCounts(test=0)
        with self.assertRaisesRegex(ValueError, "every configured"):
            NumberSplitConfig(
                fixed_counts=SplitCounts(test=10, val=5),
                fixed_priority=("test",),
            )
        with self.assertRaisesRegex(ValueError, "one of train"):
            NumberSplitConfig(
                fixed_counts=SplitCounts(test=10),
                fixed_priority=("holdout",),
            )

    def test_split_dictionary_constructor_builds_each_variant(self):
        configs = (
            split_config_from_dict(
                {
                    "type": "simple",
                    "percentages": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "seed": 9,
                }
            ),
            split_config_from_dict(
                {
                    "type": "mixed_percentage_number",
                    "fixed_counts": {"test": 100},
                    "fixed_priority": ["test"],
                    "remaining_percentages": {"train": 0.9, "val": 0.1},
                }
            ),
            split_config_from_dict(
                {
                    "type": "number",
                    "fixed_counts": {"train": 5, "test": 2},
                    "fixed_priority": ["train", "test"],
                }
            ),
        )

        self.assertIsInstance(configs[0], SimpleSplitConfig)
        self.assertIsInstance(configs[1], MixedPercentageNumberSplitConfig)
        self.assertIsInstance(configs[2], NumberSplitConfig)


if __name__ == "__main__":
    unittest.main()
