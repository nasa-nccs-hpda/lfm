import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
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
else:
    pair_dynamic_and_static = None


@unittest.skipUnless(HAS_VIZ_DEPS, "Notebook visualization dependencies required")
class TilingVisualizationTestCase(unittest.TestCase):
    def record(self, source_name, product_id=None):
        return SimpleNamespace(
            source_name=source_name,
            product_id=product_id,
            zone="42N",
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

    def test_rejects_duplicate_product_on_one_tile(self):
        records = [
            self.record("wac", "M100"),
            self.record("wac", "M100"),
            self.record("static"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            pair_dynamic_and_static(records, "wac")


if __name__ == "__main__":
    unittest.main()
