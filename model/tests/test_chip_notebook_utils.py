"""Metadata-only grid reading and reference-free notebook inspection."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import io

from lfm.model import chip_notebook_utils as helpers
from lfm.model.chip_types import ChipPreflight, ChipResult
from lfm.model.tests import test_chip_types as type_fixtures


class NotebookHelperTestCase(unittest.TestCase):
    def test_notebook_wac_nac_with_and_without_static(self):
        from lfm import model
        from lfm.model.product_ids import lunar_product_id_from_raster_path

        notebook = json.loads((Path(__file__).resolve().parents[2] /
                               "notebooks/chip_example.ipynb").read_text())
        setup = "".join(notebook["cells"][7]["source"])
        config = "".join(notebook["cells"][8]["source"])
        for modality in ("wac", "nac"):
            for static in (False, True):
                with self.subTest(modality=modality, static=static), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = root / ("M123.prj.vis.mos.tif" if modality == "wac" else
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
                    static_factory = MagicMock(side_effect=lambda **kw: model.TileSourceConfig(
                        name="static", selection_mode="all_intersecting",
                        band_names=model.STATIC_BAND_NAMES, output_nodata=model.STATIC_OUTPUT_NODATA, **kw))
                    namespace = dict(vars(model), MODALITY=modality, INCLUDE_STATIC=static,
                                     SOURCE_RASTER=source, LABEL_PATH=label if static else None,
                                     STATIC_DATA_DIR=static_dir, latest_crater_label_path=latest,
                                     AOI_NWSE=(1.3, 149.7, 1., 150.), OUTPUT_BASE_DIR=root / "out",
                                     INDEX_WORKER_COUNT=1, SPLIT_CONFIG=model.NoSplitConfig(),
                                     datetime=datetime, sys=SimpleNamespace(stdout=io.StringIO()),
                                     print=lambda *a, **kw: None, read_source_grid=reader,
                                     lunar_product_id_from_raster_path=lunar_product_id_from_raster_path,
                                     ensure_vector_index=prepare, make_static_source=static_factory)
                    exec(compile(setup, "notebook_setup", "exec"), namespace)
                    exec(compile(config, "notebook_config", "exec"), namespace)
                    result = namespace["chip_config"]
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
                    reader.assert_called_once_with(source, expected_band_count=5 if modality == "wac" else 1)
                    output = result.output_modalities[0]
                    if modality == "wac":
                        self.assertEqual(output.band_names, model.WAC_BAND_NAMES)
                    else:
                        self.assertEqual(output.band_indices, (1,))
                        self.assertEqual(output.output_band_names, ("nac",))
                    if static:
                        self.assertEqual(group.tile_config.sources[1].band_names, model.STATIC_BAND_NAMES)
                        self.assertTrue(group.tile_config.sources[1].required)
                    else:
                        self.assertIsNone(namespace["STATIC_INDEX_RESOLUTION"])

    def test_source_band_count_mismatch_is_rejected(self):
        rasterio = MagicMock()
        rasterio.open.return_value.__enter__.return_value.count = 3
        with patch.object(helpers, "_rasterio", return_value=rasterio):
            with self.assertRaisesRegex(ValueError, "Expected 1 source bands, found 3"):
                helpers.read_source_grid("nac.tif", expected_band_count=1)

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
            with patch.object(helpers, "__file__", str(root / "model/chip_notebook_utils.py")):
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
        path = Path(__file__).resolve().parents[2] / "notebooks/chip_example.ipynb"
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


if __name__ == "__main__":
    unittest.main()
