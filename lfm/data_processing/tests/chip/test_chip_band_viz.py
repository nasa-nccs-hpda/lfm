"""Per-band notebook exports retain inspection selection and raster metadata."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.chip import chip_notebook_utils as helpers
from lfm.data_processing.chip.chip_types import ChipPreflight, ChipResult
from lfm.data_processing.tests.chip import test_chip_types as type_fixtures


def notebook_cells(name):
    notebook = json.loads((REPO_ROOT / "notebooks" / name).read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


class BandVizNotebookTestCase(unittest.TestCase):
    def test_copy_preserves_inputs_creation_and_chip_selection(self):
        original = notebook_cells("chip_example.ipynb")
        copied = notebook_cells("chip_per_band_viz.ipynb")
        self.assertEqual(original.keys(), copied.keys())
        for cell_id, source in original.items():
            if cell_id in ("759e81c0", "f681685c", "db0d1143") or source.startswith(("## 5.", "## Outputs")):
                continue
            self.assertEqual(source, copied[cell_id], cell_id)

        for modality in ("wac", "nac"):
            for static in (False, True):
                for status in ("success", "failed"):
                    with self.subTest(modality=modality, static=static, status=status):
                        result = SimpleNamespace(prepared_label=None, status=status, message="failed",
                                                 cube_records=(object(),))
                        chip_plot = MagicMock(return_value=(object(), object()))
                        tile_plot = MagicMock(return_value=(object(), object()))
                        save = MagicMock(return_value=())
                        namespace = dict(result=result, OUTPUT_ROOT=Path("output"), MODALITY=modality,
                                         INCLUDE_STATIC=static, DISPLAY_BAND_KEYWORD="vis",
                                         plot_chip_result=chip_plot, plot_chip_tiling=tile_plot,
                                         save_chip_band_pngs=save, print=lambda *args: None)
                        exec(copied["db0d1143"], namespace)
                        self.assertEqual(chip_plot.call_count, int(status == "success"))
                        self.assertEqual(save.call_count, int(status == "success"))
                        tile_plot.assert_called_once()
                        self.assertEqual(tile_plot.call_args.kwargs["figure_path"], Path("output/tiling_inspection.png"))
                        if status == "success":
                            chip_plot.assert_called_once_with(result, display_band_keyword="vis",
                                                              figure_path=Path("output/chip_inspection.png"))
                            self.assertIs(save.call_args.args[0], result)
                            self.assertEqual(set(save.call_args.kwargs["band_patterns"]),
                                             {modality, "static"} if static else {modality})


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib", "rasterio")),
                     "NumPy/Matplotlib/Rasterio unavailable")
class BandPngExportTestCase(unittest.TestCase):
    def test_exports_wac_nac_and_only_five_matching_static_bands_without_display(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from lfm.data_processing.tiling import WAC_BAND_NAMES, STATIC_BAND_NAMES

        code = notebook_cells("chip_per_band_viz.ipynb")["db0d1143"].split("if result.prepared_label")[0]
        for modality, dynamic in (("wac", WAC_BAND_NAMES), ("nac", ("nac",))):
            for static in (False, True):
                with self.subTest(modality=modality, static=static), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    chip_path = root / "M123_aoi_1_input_chip.tif"
                    names = (*dynamic, *(STATIC_BAND_NAMES if static else ()))
                    pixels = np.arange(len(names) * 12, dtype="float32").reshape(len(names), 3, 4)
                    pixels[:, 0, 0] = -32768
                    pixels[:, 0, 1] = np.inf
                    if static:
                        pixels[-1] = -32768  # An uncovered slope band must still export.
                    with rasterio.open(chip_path, "w", driver="GTiff", width=4, height=3,
                                       count=len(names), dtype="float32", nodata=-32768,
                                       crs="EPSG:3857", transform=from_origin(0, 300, 100, 100)) as dataset:
                        dataset.write(pixels)
                        for index, name in enumerate(names, 1):
                            qualified = "M123.prj." + name if name in WAC_BAND_NAMES else name
                            # Exercise descriptions, product prefixes, case and Name-tag fallback.
                            if index % 2:
                                dataset.set_band_description(index, qualified)
                            else:
                                dataset.update_tags(index, Name=qualified.upper())
                    before = chip_path.read_bytes()
                    result = ChipResult(type_fixtures.ChipTypesTestCase().request(reference_path=None), "success",
                                        ChipPreflight("passed"), chip_path=chip_path)
                    namespace = dict(MODALITY=modality, INCLUDE_STATIC=static)
                    exec(code, namespace)
                    figures = plt.get_fignums()
                    with patch.object(plt, "show") as show:
                        paths = helpers.save_chip_band_pngs(result, output_dir=root / "band_pngs",
                                                           band_patterns=namespace["selected_band_patterns"])
                    show.assert_not_called()
                    self.assertEqual(plt.get_fignums(), figures)
                    self.assertEqual(len(paths), len(dynamic) + (5 if static else 0))
                    self.assertEqual(len(set(paths)), len(paths))
                    self.assertEqual(set(paths), set((root / "band_pngs").glob("*.png")))
                    for path in paths:
                        self.assertTrue(path.name.startswith(chip_path.stem + "_"))
                        self.assertIn("_band_", path.name)
                        image = imread(path)
                        self.assertEqual(image.shape, (3, 4, 4))
                        self.assertEqual(image[0, 0, 3], 0)
                        self.assertEqual(image[0, 1, 3], 0)
                        if "lola_kaguya_60mpp_slp" in path.name.lower():
                            self.assertTrue((image[:, :, 3] == 0).all())
                        else:
                            self.assertEqual(image[1, 0, 3], 1)
                        if "_static_" in path.name:
                            self.assertIn("lola_kaguya", path.name.lower())
                    self.assertEqual(chip_path.read_bytes(), before)

    def test_invalid_matching_fails_before_writing_pngs(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "chip.tif"
            with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=1,
                               dtype="float32", transform=from_origin(0, 300, 100, 100)) as dataset:
                dataset.write(np.ones((1, 3, 4), dtype="float32"))
                dataset.set_band_description(1, "nac")
            result = ChipResult(type_fixtures.ChipTypesTestCase().request(), "success", ChipPreflight("passed"), chip_path=path)
            for patterns, message in (({"static": "lola_kaguya"}, "No raster band metadata"),
                                      ({"nac": "nac", "static": "nac"}, "multiple modalities"),
                                      ({"nac": ""}, "must not be empty")):
                with self.subTest(patterns=patterns), self.assertRaisesRegex(ValueError, message):
                    helpers.save_chip_band_pngs(result, output_dir=root / "band_pngs", band_patterns=patterns)
            self.assertFalse((root / "band_pngs").exists())


if __name__ == "__main__":
    unittest.main()
