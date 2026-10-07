"""Full-raster window planning and spatially grouped dataset splits."""

import ast
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from collections import Counter

from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.chip import GeographicAOI, LabelInput, SimpleSplitConfig, SourceSelector, TargetGrid
from lfm.data_processing.chip.chip_dataset import plan_raster_chips
from lfm.data_processing.chip.chip_requests import raster_bounds
from lfm.data_processing.chip.chip_splits import plan_splits


class RasterDatasetTestCase(unittest.TestCase):
    def test_inspection_requires_all_dynamic_bands_for_each_modality(self):
        nb = json.loads((REPO_ROOT / "notebooks/chip_full_workflow.ipynb").read_text())
        code = "".join(next(c["source"] for c in nb["cells"] if c["id"] == "dataset_inspect"))
        for modality, count in (("wac", 7), ("nac", 1)):
            valid = [{"invalid_count": 0} for _ in range(count)]
            for bands, expected in ((valid + [{"invalid_count": 100}], 1),
                                    (valid[:-1] + [{"invalid_count": 1}], 0),
                                    (valid[:-1], 0), ([{}] * count, 0)):
                with self.subTest(modality=modality, bands=bands), tempfile.TemporaryDirectory() as tmp:
                    plot = MagicMock()
                    result = SimpleNamespace(status="success", imagery_nodata={"bands": bands})
                    exec(code, dict(batch=SimpleNamespace(results=[result]), INSPECTION_SAMPLES=3,
                                    OUTPUT_ROOT=Path(tmp), json=json, MODALITY=modality,
                                    plot_chip_result=plot, print=lambda *a: None))
                    self.assertEqual(plot.call_count, expected)
                    if expected:
                        plot.assert_called_once_with(result, display_band_index=0)

    def test_notebook_counts_any_band_nodata_and_plots_only_fully_valid(self):
        nb = json.loads((REPO_ROOT / "notebooks/chip_full_workflow.ipynb").read_text())
        code = "".join(next(c["source"] for c in nb["cells"] if c["id"] == "dataset_inspect"))
        bands = [{"invalid_count": 0}] * 7
        valid = SimpleNamespace(status="success", imagery_nodata={"union_invalid_count": 0, "bands": bands})
        static_gap = SimpleNamespace(status="success", imagery_nodata={"union_invalid_count": 1, "bands": bands})
        unknown = SimpleNamespace(status="success", imagery_nodata=None)
        failed = SimpleNamespace(status="failed", imagery_nodata={"union_invalid_count": 0})
        for results, expected in (([static_gap, unknown, valid, failed], (3, 1, 1, 1, 2)),
                                  ([static_gap], (1, 1, 0, 0, 1)), ([], (0, 0, 0, 0, 0))):
            with tempfile.TemporaryDirectory() as tmp:
                plot = MagicMock()
                namespace = dict(batch=SimpleNamespace(results=results), INSPECTION_SAMPLES=1,
                    OUTPUT_ROOT=Path(tmp), json=json, MODALITY="wac", plot_chip_result=plot,
                    print=lambda *a: None)
                exec(code, namespace)
                summary = json.loads((Path(tmp) / "nodata_summary.json").read_text())
                self.assertEqual(tuple(summary.values()), expected)
                self.assertEqual(plot.call_count, min(1, expected[4]))
                if expected[4]:
                    plot.assert_called_once_with(static_gap, display_band_index=0)

    def test_manual_mask_rejects_partial_windows_without_shifting_ids(self):
        import numpy as np
        mask = np.ones((512, 512), dtype=bool)
        mask[0, 0] = False
        with self.assertWarnsRegex(UserWarning, "Rejected 1"):
            plan = self.plan(self.grid(512, 512), valid_mask=mask)
        self.assertEqual(plan.rejected_mask_windows, 1)
        self.assertEqual([r.sample_id for r in plan.requests],
                         ["M123_r0_c256", "M123_r256_c0", "M123_r256_c256"])
        for mask in (np.ones((512, 512)), np.ones((1, 1), dtype=bool)):
            with self.assertRaisesRegex(ValueError, "Boolean"):
                self.plan(self.grid(512, 512), valid_mask=mask)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "No complete"):
            self.plan(self.grid(512, 512), valid_mask=np.zeros((512, 512), dtype=bool))

    @unittest.skipUnless(importlib.util.find_spec("rasterio"), "rasterio unavailable")
    def test_source_mask_checks_every_band_and_nonfinite_pixels(self):
        import numpy as np
        import rasterio
        from affine import Affine
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.tif"
            pixels = np.ones((2, 512, 512), dtype="float32")
            pixels[1, 0, 0] = -99
            pixels[0, 0, 256] = np.nan
            transform = (100, 100, 0, 200000, 0, -100)
            with rasterio.open(path, "w", driver="GTiff", width=512, height=512,
                               count=2, dtype="float32", crs="EPSG:3857",
                               transform=Affine.from_gdal(*transform), nodata=-99) as ds:
                ds.write(pixels)
            grid = TargetGrid("EPSG:3857", transform, raster_bounds(transform, 512, 512), 512, 512)
            with self.assertWarnsRegex(UserWarning, "Rejected 2"):
                plan = self.plan(grid, source_raster=path)
            self.assertEqual(len(plan.requests), 2)
            with self.assertRaisesRegex(ValueError, "match the planning grid"):
                self.plan(self.grid(256, 256), source_raster=path)

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

    def test_full_notebook_syntax_and_direct_creation_with_bar(self):
        nb = json.loads((REPO_ROOT / "notebooks/chip_full_workflow.ipynb").read_text())
        self.assertEqual(len(nb["cells"]), len({c["id"] for c in nb["cells"]}))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("\n".join(line for line in "".join(cell["source"]).splitlines()
                                    if not line.startswith("%")))
        cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
        self.assertNotIn("RUN_CREATION", "".join(cells.values()))
        self.assertNotIn("CHIP_SIZE", "".join(cells.values()))
        calls = [node for node in ast.walk(ast.parse(cells["dataset_plan"]))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "plan_raster_chips"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("chip_size", {kw.arg for kw in calls[0].keywords})
        create = MagicMock(return_value=SimpleNamespace(results=[], manifest_path="manifest.json"))
        namespace = dict(create_chips=create, requests=[object(), object()], chip_config=object(),
                         notebook_index_workers=lambda: 16, OVERWRITE=True, Counter=Counter,
                         print=lambda *a: None)
        exec(cells["dataset_run"], namespace)
        self.assertEqual(create.call_args.kwargs["max_workers"], 2)
        self.assertEqual(create.call_args.kwargs["progress_mode"], "bar")
        self.assertTrue(create.call_args.kwargs["progress"])

    def test_bar_progress_counts_workers_without_stage_messages(self):
        from lfm.data_processing.chip.chip_creation import _ChipProgressReporter, ChipProgressEvent
        tqdm = MagicMock()
        with patch("lfm.data_processing.chip.chip_creation._load_tqdm", return_value=tqdm):
            reporter = _ChipProgressReporter(2, 2, enabled=True, mode="bar")
            reporter.stage(ChipProgressEvent("a", "publish", "started", 123))
            for name, status in (("a", "success"), ("b", "failed")):
                result = SimpleNamespace(request=SimpleNamespace(sample_id=name), status=status,
                                         diagnostics=[], message="test error")
                reporter.complete(result)
                reporter.complete(result)  # No duplicate increments.
            reporter.close()
        tqdm.assert_called_once()
        self.assertIn("2 workers", tqdm.call_args.kwargs["desc"])
        tqdm.write.assert_not_called()
        self.assertEqual(tqdm.return_value.update.call_count, 2)
        tqdm.return_value.set_postfix.assert_called_with({"failed": 1, "success": 1}, refresh=False)
        tqdm.return_value.close.assert_called_once()

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
