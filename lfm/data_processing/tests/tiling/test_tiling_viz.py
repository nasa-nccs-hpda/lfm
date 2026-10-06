import importlib.util
from lfm.data_processing._paths import REPO_ROOT
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE_PATH = (
    REPO_ROOT / "lfm" / "all_models" / "all_tasks" / "viz" / "tiling_viz.py"
)
HAS_VIZ_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("matplotlib", "numpy", "rasterio")
)
if HAS_VIZ_DEPS:
    SPEC = importlib.util.spec_from_file_location(
        "tiling_viz_under_test",
        MODULE_PATH,
    )
    if SPEC is None or SPEC.loader is None:
        raise ImportError(f"Could not load visualization helpers from {MODULE_PATH}")
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
    pair_dynamic_and_static = MODULE.pair_dynamic_and_static
    plot_tiling_records = MODULE.plot_tiling_records
    wrap_subplot_title = MODULE.wrap_subplot_title
else:
    pair_dynamic_and_static = None
    plot_tiling_records = None
    wrap_subplot_title = None


@unittest.skipUnless(HAS_VIZ_DEPS, "Notebook visualization dependencies required")
class TilingVisualizationTestCase(unittest.TestCase):
    def record(self, source_name, product_id=None):
        return SimpleNamespace(
            source_name=source_name,
            product_id=product_id,
            zone="42N",
            grid_id="42N",
            zoom_level=5,
            tile_x=1,
            tile_y=2,
        )

    def test_pairs_multiple_products_with_one_static_record(self):
        static = self.record("static")
        second = self.record("wac", "M200")
        first = self.record("wac", "M100")

        pairs = pair_dynamic_and_static(
            [static, second, first],
            "wac",
        )

        self.assertEqual(pairs, [(first, static), (second, static)])

    def test_subplot_title_lines_are_limited_to_45_characters(self):
        title = (
            "LPS_N z4 tile (123, 456) product M1107459759CE\n"
            "STATIC band 63: an_extremely_long_static_band_filename.tif"
        )

        wrapped = wrap_subplot_title(title)

        self.assertGreater(len(wrapped.splitlines()), 2)
        self.assertTrue(
            all(len(line) <= 45 for line in wrapped.splitlines()),
            wrapped,
        )
        self.assertEqual("".join(wrapped.split()), "".join(title.split()))

    def test_rejects_duplicate_product_on_one_tile(self):
        records = [
            self.record("wac", "M100"),
            self.record("wac", "M100"),
            self.record("static"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            pair_dynamic_and_static(records, "wac")

    def test_mixed_records_use_paired_plot(self):
        records = [self.record("wac", "M100"), self.record("static")]
        with patch.object(
            MODULE,
            "plot_cube_pairs",
            return_value="paired",
        ) as plot_pairs:
            result = plot_tiling_records(
                records,
                dynamic_source="wac",
                dynamic_label="WAC",
                dynamic_band_number=3,
                static_band_name="elevation",
                output_path="mixed.png",
            )

        self.assertEqual(result, "paired")
        self.assertEqual(plot_pairs.call_args.args[0], [(records[0], records[1])])

    def test_dynamic_only_records_use_single_source_plot(self):
        records = [self.record("wac", "M100")]
        with patch.object(
            MODULE,
            "plot_cube_records",
            return_value="dynamic",
        ) as plot_records:
            result = plot_tiling_records(
                records,
                dynamic_source="wac",
                dynamic_label="WAC",
                dynamic_band_number=3,
                static_band_name="elevation",
                output_path="dynamic.png",
            )

        self.assertEqual(result, "dynamic")
        self.assertEqual(plot_records.call_args.kwargs["band_number"], 3)
        self.assertIsNone(plot_records.call_args.kwargs.get("band_name"))

    def test_static_only_records_use_named_band_plot(self):
        records = [self.record("static")]
        with patch.object(
            MODULE,
            "plot_cube_records",
            return_value="static",
        ) as plot_records:
            result = plot_tiling_records(
                records,
                dynamic_source="wac",
                dynamic_label="WAC",
                dynamic_band_number=3,
                static_band_name="elevation",
                output_path="static.png",
            )

        self.assertEqual(result, "static")
        self.assertEqual(plot_records.call_args.kwargs["band_name"], "elevation")
        self.assertIsNone(plot_records.call_args.kwargs.get("band_number"))


if __name__ == "__main__":
    unittest.main()
