"""Metadata-only grid reading and reference-free notebook inspection."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from lfm.model import chip_notebook_utils as helpers
from lfm.model.chip_types import ChipPreflight, ChipResult
from lfm.model.tests import test_chip_types as type_fixtures


class NotebookHelperTestCase(unittest.TestCase):
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
