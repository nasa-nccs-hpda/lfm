"""Full-raster window planning and spatially grouped dataset splits."""

import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.chip import GeographicAOI, LabelInput, SimpleSplitConfig, SourceSelector, TargetGrid
from lfm.data_processing.chip.chip_dataset import plan_raster_chips
from lfm.data_processing.chip.chip_requests import raster_bounds
from lfm.data_processing.chip.chip_splits import plan_splits


class RasterDatasetTestCase(unittest.TestCase):
    def grid(self, width=1024, height=1024, transform=(100, 100, 0, 200000, 0, -100)):
        return TargetGrid("test_crs", transform, raster_bounds(transform, width, height), width, height)

    def plan(self, grid, **kwargs):
        with patch("lfm.data_processing.chip.chip_dataset.geographic_aoi_from_target_grid",
                   return_value=GeographicAOI(2, 10, 1, 11)):
            return plan_raster_chips(grid, product_id="M123", label_input=LabelInput(Path("label.gpkg")),
                source_selectors=(SourceSelector("wac_grid", "wac", "M123"),), **kwargs)

    def test_exact_disjoint_native_windows_ids_and_edge_counts(self):
        with self.assertWarnsRegex(UserWarning, "incomplete windows"):
            plan = self.plan(self.grid(1027, 770), block_size_chips=2)
        self.assertEqual((len(plan.requests), plan.window_rows, plan.window_columns), (12, 3, 4))
        self.assertEqual((plan.dropped_rows, plan.dropped_columns), (2, 3))
        self.assertEqual(plan.dropped_pixels, 1027 * 770 - 12 * 256**2)
        for index, request in enumerate(plan.requests):
            row, col = divmod(index, 4)
            self.assertEqual(request.sample_id, f"M123_r{row * 256}_c{col * 256}")
            self.assertEqual((request.target_grid.width, request.target_grid.height), (256, 256))
            self.assertEqual(request.target_grid.transform,
                (100 + col * 25600, 100, 0, 200000 - row * 25600, 0, -100))
            self.assertEqual(request.split_group_key, f"M123_block2_chip256_r{row // 2}_c{col // 2}")
            self.assertEqual(request.label_input.path, Path("label.gpkg"))

    def test_rotated_rectangle_preserved_shear_rejected(self):
        grid = self.grid(transform=(100, 80, 60, 200000, 60, -80))
        plan = self.plan(grid)
        self.assertEqual(plan.requests[1].target_grid.transform, (20580, 80, 60, 215360, 60, -80))
        with self.assertRaisesRegex(ValueError, "rectangular"):
            self.plan(self.grid(transform=(100, 100, 20, 200000, 0, -100)))

    def test_invalid_sizes_and_too_small_source(self):
        for kwargs in ({"chip_size": 0}, {"chip_size": True}, {"block_size_chips": 1.5}):
            with self.assertRaises(ValueError):
                self.plan(self.grid(), **kwargs)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "no complete"):
            self.plan(self.grid(200, 200))

    def test_group_atomic_stable_splits_and_seed(self):
        requests = self.plan(self.grid(4096, 4096), block_size_chips=2).requests
        first = plan_splits(requests, SimpleSplitConfig(seed=42))
        self.assertEqual(first, plan_splits(tuple(reversed(requests)), SimpleSplitConfig(seed=42)))
        groups = {}
        for item in first.assignments:
            groups.setdefault(item.split_group_key, set()).add(item.assigned_split)
        self.assertTrue(all(len(values) == 1 for values in groups.values()))
        self.assertEqual({a.assigned_split for a in first.assignments}, {"train", "val", "test"})
        self.assertNotEqual(first, plan_splits(requests, SimpleSplitConfig(seed=43)))

    def test_full_notebook_syntax_and_preview_only_execution(self):
        nb = json.loads((REPO_ROOT / "notebooks/chip_full_workflow.ipynb").read_text())
        self.assertEqual(len(nb["cells"]), len({c["id"] for c in nb["cells"]}))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("\n".join(line for line in "".join(cell["source"]).splitlines()
                                    if not line.startswith("%")))
        cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
        namespace = dict(RUN_CREATION=False, print=lambda *a: None)
        exec(cells["dataset_index"], namespace)
        exec(cells["dataset_run"], namespace)
        self.assertIsNone(namespace["batch"])

    @unittest.skipUnless(importlib.util.find_spec("osgeo"), "GDAL unavailable")
    def test_real_geographic_query_does_not_expand_target_window(self):
        from lfm.data_processing.tiling.grid_registry import default_grid_registry
        from lfm.data_processing.chip.chip_requests import validate_request_geographic_aoi
        grid = self.grid(512, 512)
        grid = TargetGrid(default_grid_registry()["23N"].crs_wkt, grid.transform,
                          grid.bounds, grid.width, grid.height)
        plan = plan_raster_chips(grid, product_id="M123", label_input=LabelInput(Path("label.gpkg")),
                                source_selectors=())
        for request in plan.requests:
            self.assertEqual((request.target_grid.width, request.target_grid.height), (256, 256))
            validate_request_geographic_aoi(request)
