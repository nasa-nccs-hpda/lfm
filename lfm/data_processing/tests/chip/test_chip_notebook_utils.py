"""Metadata-only grid reading and reference-free notebook inspection."""

import ast
import importlib.util
import json
import os
from lfm.data_processing._paths import REPO_ROOT
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import io

from lfm.data_processing.chip import chip_notebook_utils as helpers
from lfm.data_processing.chip.chip_types import ChipPreflight, ChipResult
from lfm.data_processing.tests.chip import test_chip_types as type_fixtures


class NotebookHelperTestCase(unittest.TestCase):
    def test_explicit_zero_based_display_band_overrides_keyword(self):
        ds = MagicMock()
        ds.descriptions = ("nac-0", "vis-1")
        ds.count = 2
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value = ds
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            _, name = helpers.read_display_band("chip.tif", band_index=0)
            self.assertEqual(name, "nac-0")
            ds.read.assert_called_once_with(1, masked=True)
            for index in (-1, 2, True, 0.5):
                with self.assertRaises(ValueError):
                    helpers.read_display_band("chip.tif", band_index=index)

    def test_notebook_index_workers_respects_allocation_and_affinity(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "6"}):
            self.assertEqual(helpers.notebook_index_workers(), 6)
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": ""}), patch.object(
            helpers.os, "sched_getaffinity", return_value={0, 1, 2}, create=True
        ):
            self.assertEqual(helpers.notebook_index_workers(), 3)
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": ""}), patch.object(
            helpers.os, "sched_getaffinity", side_effect=OSError, create=True
        ), patch.object(helpers.os, "cpu_count", return_value=4):
            self.assertEqual(helpers.notebook_index_workers(), 4)
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "invalid"}):
            with self.assertRaises(ValueError):
                helpers.notebook_index_workers()

    def test_quick_example_execution_defaults(self):
        notebook = json.loads((REPO_ROOT / "notebooks/chip_example.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        config_tree = ast.parse(cells["d58d002a"])
        assigned = {node.id for node in ast.walk(config_tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.assertFalse(assigned & {"SPLIT_CONFIG", "MAX_WORKERS", "INDEX_WORKER_COUNT", "OVERWRITE"})
        create = MagicMock(return_value=SimpleNamespace(results=[SimpleNamespace(status="success", diagnostics=[])],
                                                       elapsed_seconds=0, manifest_path="manifest.json"))
        exec(cells["94a010f2"], dict(create_chips=create, request=object(), chip_config=object(),
                                   print=lambda *args: None))
        self.assertEqual(create.call_args.kwargs["max_workers"], 1)
        self.assertTrue(create.call_args.kwargs["overwrite"])
        self.assertNotIn("82828596", cells)

    def test_notebook_wac_nac_with_and_without_static(self):
        from lfm.data_processing import chip, tiling
        from lfm.data_processing.tiling.product_ids import lunar_product_id_from_raster_path

        notebook = json.loads((REPO_ROOT /
                               "notebooks/chip_example.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        setup = cells["92e30058"]
        config = cells["58269cbd"]
        for modality, source_kind in (("wac", "vis"), ("wac", "uv"), ("nac", "nac")):
            for static in (False, True):
                with self.subTest(modality=modality, source_kind=source_kind, static=static), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = root / (f"M123.prj.{source_kind}.mos.tif" if modality == "wac" else
                                     "NAC_DTM_NEWCRATER6_M1219245090_80CM.TIF")
                    source.touch()
                    label = root / "craters.gpkg"
                    label.touch()
                    static_dir = root / "static"
                    if static:
                        static_dir.mkdir()
                    reader = MagicMock(return_value=type_fixtures.ChipTypesTestCase().grid())
                    latest = MagicMock(return_value=label)
                    prepare = MagicMock(side_effect=lambda cfg, **kw: SimpleNamespace(index_path=cfg.index_path))
                    static_factory = MagicMock(side_effect=lambda **kw: tiling.TileSourceConfig(
                        name="static", selection_mode="all_intersecting",
                        band_names=tiling.STATIC_BAND_NAMES, output_nodata=tiling.STATIC_OUTPUT_NODATA, **kw))
                    namespace = dict({**vars(tiling), **vars(chip)}, MODALITY=modality, INCLUDE_STATIC=static,
                                     SOURCE_RASTER=source, LABEL_PATH=label if static else None,
                                     STATIC_DATA_DIR=static_dir, latest_crater_label_path=latest,
                                     AOI_NWSE=(1.3, 149.7, 1., 150.), OUTPUT_BASE_DIR=root / "out",
                                     notebook_index_workers=lambda: 3,
                                     datetime=datetime, sys=SimpleNamespace(stdout=io.StringIO()),
                                     print=lambda *a, **kw: None, read_source_grid=reader,
                                     lunar_product_id_from_raster_path=lunar_product_id_from_raster_path,
                                     ensure_vector_index=prepare, make_static_source=static_factory)
                    exec(compile(setup, "notebook_setup", "exec"), namespace)
                    exec(compile(config, "notebook_config", "exec"), namespace)
                    result = namespace["chip_config"]
                    self.assertIsInstance(result.split_config, chip.NoSplitConfig)
                    self.assertEqual(result.intermediate_retention, "always")
                    self.assertTrue(all(call.args[0].worker_count == 3 for call in prepare.call_args_list))
                    self.assertEqual(namespace["LABEL_PATH"], label)
                    self.assertEqual(latest.call_count, 0 if static else 1)
                    group = result.acquisition_groups[0]
                    self.assertEqual(group.tile_config.zoom_level, 5 if modality == "wac" else 11)
                    self.assertEqual(len(group.tile_config.sources), 2 if static else 1)
                    self.assertEqual(prepare.call_count, 2 if static else 1)
                    self.assertEqual(static_factory.call_count, int(static))
                    self.assertTrue(group.tile_config.sources[0].preserve_source_nodata)
                    self.assertTrue(group.tile_config.sources[0].required)
                    self.assertEqual(namespace["PRODUCT_ID"], source.name.split(".")[0])
                    reader.assert_called_once_with(source, expected_band_count=(2, 5) if modality == "wac" else 1)
                    output = result.output_modalities[0]
                    if modality == "wac":
                        self.assertEqual(output.band_names, tiling.WAC_BAND_NAMES)
                    else:
                        self.assertEqual(output.band_indices, (1,))
                        self.assertEqual(output.output_band_names, ("nac",))
                    if static:
                        self.assertEqual(group.tile_config.sources[1].band_names, tiling.STATIC_BAND_NAMES)
                        self.assertTrue(group.tile_config.sources[1].required)
                    else:
                        self.assertIsNone(namespace["STATIC_INDEX_RESOLUTION"])

    def test_source_band_count_mismatch_is_rejected(self):
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value.count = 3
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            with self.assertRaisesRegex(ValueError, "Expected 1 source bands, found 3"):
                helpers.read_source_grid("nac.tif", expected_band_count=1)

    def test_wac_grid_accepts_uv_and_vis_band_counts(self):
        dataset = MagicMock()
        dataset.width, dataset.height = 4, 3
        dataset.crs.to_wkt.return_value = "PROJCS[Moon]"
        dataset.transform.to_gdal.return_value = (10., 2., .5, 30., .25, -2.)
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value = dataset
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            for count in (2, 5):
                with self.subTest(count=count):
                    dataset.count = count
                    grid = helpers.read_source_grid("wac.tif", expected_band_count=(2, 5))
                    self.assertEqual(grid.transform, dataset.transform.to_gdal.return_value)
                    self.assertEqual((grid.width, grid.height), (4, 3))
            for count in (1, 3, 7):
                with self.subTest(count=count):
                    dataset.count = count
                    with self.assertRaisesRegex(ValueError, f"Expected 2 or 5 source bands, found {count}"):
                        helpers.read_source_grid("wac.tif", expected_band_count=(2, 5))
        dataset.read.assert_not_called()

    def test_full_workflow_accepts_uv_and_vis_source_grids(self):
        notebook = json.loads((REPO_ROOT / "notebooks/chip_full_workflow.ipynb").read_text())
        setup = "".join(next(cell["source"] for cell in notebook["cells"]
                             if cell["id"] == "dataset_plan")).split("\nplan = plan_raster_chips(")[0]
        from lfm.data_processing.tiling.product_ids import lunar_product_id_from_raster_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = root / "craters.gpkg"
            label.touch()
            for source_kind in ("uv", "vis"):
                with self.subTest(source_kind=source_kind):
                    source = root / f"M123.prj.{source_kind}.mos.tif"
                    source.touch()
                    grid = type_fixtures.ChipTypesTestCase().grid()
                    reader = MagicMock(return_value=grid)
                    namespace = dict(Path=Path, LABEL_PATH=label, SOURCE_RASTER=source,
                                     MODALITY="wac", INCLUDE_STATIC=False, OUTPUT_BASE_DIR=root / "out",
                                     datetime=datetime, read_source_grid=reader,
                                     lunar_product_id_from_raster_path=lunar_product_id_from_raster_path)
                    exec(setup, namespace)
                    reader.assert_called_once_with(source, expected_band_count=(2, 5))
                    self.assertIs(namespace["source_grid"], grid)
                    self.assertEqual(namespace["PRODUCT_ID"], "M123")

    def test_latest_export_uses_modification_time_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a_label_craters.gpkg"
            second = root / "b_label_craters.gpkg"
            for path, timestamp in ((first, 100), (second, 200), (root / "tmp123.gpkg", 300),
                                    (root / "output_index.gpkg", 400)):
                path.touch()
                os.utime(path, ns=(timestamp, timestamp))
            (root / "directory_label_craters.gpkg").mkdir()
            self.assertEqual(helpers.latest_crater_label_path(root), second)
            # An autosave updates an existing name, rather than creating a new one.
            os.utime(first, ns=(500, 500))
            self.assertEqual(helpers.latest_crater_label_path(root), first)
            os.utime(second, ns=(500, 500))
            self.assertEqual(helpers.latest_crater_label_path(root), first)

    def test_latest_export_default_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "notebooks/outputs/labels"
            labels.mkdir(parents=True)
            expected = labels / "sample_label_craters.gpkg"
            expected.touch()
            with patch.object(helpers, "REPO_ROOT", root):
                self.assertEqual(helpers.latest_crater_label_path(), expected)

    def test_latest_export_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Label directory does not exist"):
                helpers.latest_crater_label_path(Path(tmp) / "missing")
            with self.assertRaisesRegex(FileNotFoundError, "No .* exports found"):
                helpers.latest_crater_label_path(tmp)

    def test_source_grid_reads_metadata_not_pixels(self):
        dataset = MagicMock()
        dataset.width, dataset.height = 4, 3
        dataset.crs.to_wkt.return_value = "PROJCS[Moon]"
        dataset.transform.to_gdal.return_value = (10., 2., .5, 30., .25, -2.)
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value = dataset
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            grid = helpers.read_source_grid("original.tif")
        self.assertEqual((grid.width, grid.height), (4, 3))
        self.assertEqual(grid.bounds, (10., 24., 19.5, 31.))
        dataset.read.assert_not_called()
        rasterio.open.return_value.__exit__.assert_called_once()

    def test_source_without_crs_is_rejected(self):
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value = SimpleNamespace(crs=None)
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            with self.assertRaisesRegex(ValueError, "no CRS"):
                helpers.read_source_grid("original.tif")

    def test_notebook_cells_compile_and_use_aoi_path(self):
        path = REPO_ROOT / "notebooks/chip_example.ipynb"
        notebook = json.loads(path.read_text())
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            cleaned = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
            ast.parse(cleaned)
            if source.startswith("# AOI_JOBS"):
                ast.parse("\n".join(line[2:] if line.startswith("# ") else line
                                    for line in source.splitlines()))
            sources.append(cleaned)
        active = "\n".join(sources)
        self.assertIn("request = chip_request_from_aoi(", active)
        self.assertNotIn("REFERENCE_CHIP", active)
        self.assertNotIn("reference_sample_from_tiff", active)
        self.assertIn('relation="clip_to_target"', active)
        self.assertIn("if LABEL_PATH is None:", active)
        self.assertIn('LABEL_KIND = "auto"', active)
        self.assertIn("kind=LABEL_KIND", active)
        self.assertIn('LABEL_PATH.suffix.lower() == ".gpkg" else None', active)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib")),
                         "NumPy/Matplotlib unavailable")
    def test_semantic_plot_keeps_zero_class_and_reports_classes(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        req = type_fixtures.ChipTypesTestCase().request(reference_path=None)
        result = ChipResult(req, "success", ChipPreflight("passed"),
                            chip_path=Path("chip.tif"), label_path=Path("label.npy"))
        label = np.array([[0, 12, 90, 0]] * 3)
        with patch.object(helpers, "read_display_band", return_value=(np.ones((3, 4)), "vis")), \
             patch.object(helpers, "read_label", return_value=(label, None)):
            figure, axes = helpers.plot_chip_result(result, show=False)
        self.assertEqual(axes[0, 1].get_title(), "semantic classes: [0, 12, 90]")
        self.assertFalse(np.ma.getmaskarray(axes[0, 1].images[0].get_array()).any())
        rendered = axes[0, 1].images[0]
        np.testing.assert_array_equal(rendered.to_rgba(rendered.get_array())[0, 0], [0, 0, 0, 1])
        self.assertTrue(np.ma.getmaskarray(axes[0, 2].images[1].get_array())[0, 0])
        np.testing.assert_array_equal(label, [[0, 12, 90, 0]] * 3)
        plt.close(figure)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib")),
                         "NumPy/Matplotlib unavailable")
    def test_plot_layout_with_and_without_reference(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for reference in (None, Path("reference.tif")):
            with self.subTest(reference=reference):
                request = type_fixtures.ChipTypesTestCase().request(reference_path=reference)
                result = ChipResult(request, "success", ChipPreflight("passed"),
                                    chip_path=Path("chip.tif"), label_path=Path("label.npz"))
                with patch.object(helpers, "read_display_band", return_value=(np.ones((3, 4)), "vis")) as read:
                    with patch.object(helpers, "read_label", return_value=(np.ones((3, 4)), 1)):
                        figure, axes = helpers.plot_chip_result(result, show=False)
                self.assertEqual(axes.shape, (1, 3) if reference is None else (2, 2))
                self.assertEqual(read.call_count, 1 if reference is None else 2)
                plt.close(figure)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib", "rasterio")),
                     "NumPy/Matplotlib/Rasterio unavailable")
class NotebookPlotTestCase(unittest.TestCase):
    def test_instance_background_is_black_and_overlay_background_is_transparent(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for label in (np.zeros((3, 4), dtype=np.uint16),
                      np.array([[0, 1, 20, 21]] * 3, dtype=np.uint16)):
            with self.subTest(label=label.tolist()):
                result = ChipResult(type_fixtures.ChipTypesTestCase().request(), "success", ChipPreflight("passed"),
                                    chip_path="chip.tif", label_path="label.npz")
                with patch.object(helpers, "read_display_band", return_value=(np.ones((3, 4)), "vis")), \
                     patch.object(helpers, "read_label", return_value=(label, int(label.max()))):
                    figure, axes = helpers.plot_chip_result(result, show=False)
                rendered = axes[0, 1].images[0]
                rgba = rendered.to_rgba(rendered.get_array())
                np.testing.assert_array_equal(rgba[label == 0], np.tile([0, 0, 0, 1], ((label == 0).sum(), 1)))
                overlay = axes[0, 2].images[1]
                np.testing.assert_array_equal(np.ma.getmaskarray(overlay.get_array()), label == 0)
                self.assertEqual(overlay.cmap.get_bad()[-1], 0)
                if label.max():
                    self.assertFalse(np.array_equal(rgba[0, 1], rgba[0, 2]))
                    np.testing.assert_array_equal(rgba[0, 1], rgba[0, 3])
                plt.close(figure)

    def test_tiling_plot_reads_retained_cubes_masks_nodata_without_outlines(self):
        from dataclasses import replace
        import numpy as np
        import rasterio
        from rasterio.transform import Affine
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lfm.data_processing.tiling.tiling_results import TileCubeRecord

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wkt = (REPO_ROOT / "TMS/IAU_30100_2015.wkt").read_text()
            grid = type_fixtures.ChipTypesTestCase().grid(crs_wkt=wkt, width=4, height=3,
                                                        bounds=(149.7, 1.297, 149.704, 1.3))
            records = []
            for source in ("wac", "static"):
                path = root / f"{source}.tif"
                data = np.arange(24, dtype="float32").reshape(2, 3, 4)
                data[:, 0, 0] = -32768
                data[:, 0, 1] = np.inf
                with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=2, dtype="float32",
                                   crs=wkt, transform=Affine.from_gdal(*grid.transform)) as ds:
                    ds.write(data)
                records.append(TileCubeRecord(source, "42S", 5, 1, 2, "M123" if source == "wac" else None,
                                              path, ("vis", "uv") if source == "wac" else ("other", "elv"),
                                              wkt, (-32768, -32768)))
            # A later tile can be absent when the display limit excludes it.
            records.append(replace(records[0], tile_x=2, path=root / "unselected.tif"))
            result = ChipResult(type_fixtures.ChipTypesTestCase().request(target_grid=grid), "success",
                                ChipPreflight("passed"), cube_records=tuple(records))
            before = {r.path: r.path.read_bytes() for r in records[:2]}
            saved = root / "figures/tiling.png"
            with patch.object(plt, "show") as show:
                figure, axes = helpers.plot_chip_tiling(result, max_tiles=1, figure_path=saved, show=False)
            show.assert_not_called()
            self.assertTrue(saved.is_file())
            self.assertEqual(axes.shape, (1, 2))
            self.assertIn("Showing 1 of 2 tile addresses", figure._suptitle.get_text())
            for axis in axes.flat:
                image = axis.images[0]
                self.assertEqual(image.get_array().dtype, np.float64)
                self.assertEqual(np.ma.getmaskarray(image.get_array()).sum(), 2)
                self.assertTrue(np.isfinite(image.get_clim()).all())
                np.testing.assert_allclose(axis.get_xlim(), [-.5, 3.5])
                self.assertEqual(len(axis.lines), 0)
            self.assertIn("static | elv", axes[0, 1].get_title())
            self.assertEqual(axes[0, 1].images[0].get_array()[1, 0], 16)
            self.assertEqual(before, {r.path: r.path.read_bytes() for r in records[:2]})
            plt.close(figure)
            with self.assertRaisesRegex(FileNotFoundError, 'intermediate_retention="always"'):
                helpers.plot_chip_tiling(result, max_tiles=2, show=False)

    def test_step_five_plots_both_chip_and_tiling(self):
        notebook = json.loads((REPO_ROOT / "notebooks/chip_example.ipynb").read_text())
        code = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "figure, axes = plot_chip_result(" in "".join(c["source"]))
        plot_chip = MagicMock(return_value=(object(), object()))
        plot_tiles = MagicMock(return_value=(object(), object()))
        result = SimpleNamespace(prepared_label=None, status="success", cube_records=(object(),))
        with tempfile.TemporaryDirectory() as tmp:
            exec(code, dict(result=result, OUTPUT_ROOT=Path(tmp), DISPLAY_BAND_KEYWORD="vis",
                            plot_chip_result=plot_chip, plot_chip_tiling=plot_tiles, print=lambda *a: None))
        plot_chip.assert_called_once()
        plot_tiles.assert_called_once()
        self.assertIs(plot_tiles.call_args.args[0], result)
        self.assertEqual(plot_tiles.call_args.kwargs["max_tiles"], 4)


if __name__ == "__main__":
    unittest.main()
