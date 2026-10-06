"""Initial single-region polar chip contract; seam/pole cases stay gated."""

import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from lfm.data_processing.chip import (
    GeographicAOI, chip_grid_family, default_chip_zoom, chip_request_from_aoi,
    static_grid_reference,
)
from lfm.data_processing.chip.chip_requests import (
    UnsupportedCoverageError, geographic_query_parts, geographic_aoi_from_target_grid,
    target_grid_from_bounds, validate_request_geographic_aoi,
)
from lfm.data_processing.tiling.grid_registry import default_grid_registry, GridFamily
from lfm.data_processing._paths import REPO_ROOT


def polar_acquire(prepared, config):
    """Replace expensive tiling only; keep all raster/label/publication stages."""
    from lfm.data_processing.tests.chip.test_chip_a5_integration import synthetic_acquire
    acquired = synthetic_acquire(prepared, config)
    family = chip_grid_family(prepared.request.geographic_aoi)
    groups = tuple(replace(group, zoom_level=4, records=tuple(
        replace(record, zone=family.value.upper(), zoom_level=4) for record in group.records))
        for group in acquired.group_results)
    return replace(acquired, group_results=groups)


def polar_worker(task):
    from lfm.data_processing.chip.chip_creation import _run_prepared_request
    with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=polar_acquire):
        return _run_prepared_request(*task)


class PolarChipContractTestCase(unittest.TestCase):
    def test_hemisphere_static_grid_and_zoom_defaults(self):
        for aoi, family in ((GeographicAOI(86.1, -.1, 85.9, .1), GridFamily.LPS_N),
                            (GeographicAOI(-85.9, -.1, -86.1, .1), GridFamily.LPS_S)):
            with self.subTest(family=family):
                self.assertEqual(chip_grid_family(aoi), family)
                parts = geographic_query_parts(aoi)
                self.assertEqual(len(parts), 1)
                self.assertAlmostEqual(parts[0].upper_left_longitude, aoi.upper_left_longitude)
                self.assertAlmostEqual(parts[0].lower_right_longitude, aoi.lower_right_longitude)
                grid = static_grid_reference(aoi)
                self.assertEqual(grid.crs_wkt, default_grid_registry()[family.value.upper()].crs_wkt)
                self.assertEqual(grid.transform, (0, 100, 0, 0, 0, -100))
                self.assertEqual(default_chip_zoom(aoi, "wac"), 4)
                self.assertEqual(default_chip_zoom(aoi, "nac"), 10)
                self.assertEqual(default_chip_zoom(aoi, "static"), 4)

    def test_ltm_defaults_unchanged_and_custom_requires_override(self):
        aoi = GeographicAOI(1, 10, 0, 11)
        self.assertEqual(default_chip_zoom(aoi, "wac"), 5)
        self.assertEqual(default_chip_zoom(aoi, "nac"), 11)
        with self.assertRaises(ValueError):
            default_chip_zoom(aoi, "custom")

    def test_unsupported_seams_poles_and_polar_antimeridian(self):
        for aoi in (GeographicAOI(82.1, 10, 81.9, 11),
                    GeographicAOI(-81.9, 10, -82.1, 11),
                    GeographicAOI(90, 10, 89, 11),
                    GeographicAOI(-89, 10, -90, 11),
                    GeographicAOI(86, 179, 85, -179)):
            with self.subTest(aoi=aoi), self.assertRaises(UnsupportedCoverageError):
                geographic_query_parts(aoi)

    def test_exact_threshold_routes_by_positive_area(self):
        self.assertEqual(chip_grid_family(GeographicAOI(83, 10, 82, 11)), GridFamily.LPS_N)
        self.assertEqual(chip_grid_family(GeographicAOI(82, 10, 81, 11)), GridFamily.LTM)

    def test_acquisition_keeps_explicit_zoom_and_selectors(self):
        from lfm.data_processing.tests.chip.test_chip_acquisition import ChipAcquisitionTestCase
        from lfm.data_processing.chip.chip_acquisition import acquire_prepared_request
        from lfm.data_processing.chip import SourceSelector
        helper = ChipAcquisitionTestCase()
        with tempfile.TemporaryDirectory() as tmp:
            group = helper.group("polar", Path(tmp), 9, helper.source("nac", selection_mode="product_id"))
            config = helper.config(Path(tmp), (group,))
            req = helper.request(aoi=GeographicAOI(86, 10, 85.9, 10.1),
                                 selectors=(SourceSelector("polar", "nac", "M100"),))
            with patch("lfm.data_processing.chip.chip_acquisition.create_tiles_for_aoi", return_value=[]) as tiler:
                acquire_prepared_request(helper.prepared(req), config)
            self.assertEqual(tiler.call_args.args[0].zoom_level, 9)
            self.assertEqual(tiler.call_args.kwargs["selectors"], {"nac": "M100"})
            self.assertEqual(tiler.call_args.kwargs["ul_lat"], 86)

    def test_polar_notebook_cells_compile(self):
        notebook = json.loads((REPO_ROOT / "notebooks/chip_polar_example.ipynb").read_text())
        self.assertEqual(len({c["id"] for c in notebook["cells"]}), len(notebook["cells"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))


@unittest.skipUnless(importlib.util.find_spec("osgeo"), "GDAL unavailable")
class PolarChipGridTestCase(unittest.TestCase):
    def test_north_south_native_and_static_requests(self):
        for aoi in (GeographicAOI(86.01, -.02, 86, .02),
                    GeographicAOI(-86, -.02, -86.01, .02)):
            request = chip_request_from_aoi(sample_id="polar", split_group_key="polar",
                                            geographic_aoi=aoi, static_only=True)
            self.assertEqual(request.target_grid.transform[1], 100)
            self.assertEqual(request.target_grid.transform[5], -100)
            self.assertEqual(request.requested_aoi, aoi)
            validate_request_geographic_aoi(request)
            # Preserve an explicitly chosen 1 m lattice, not the acquisition grid.
            source = target_grid_from_bounds(crs_wkt=request.target_grid.crs_wkt,
                                             bounds=request.target_grid.bounds,
                                             width=request.target_grid.width * 100,
                                             height=request.target_grid.height * 100)
            native = chip_request_from_aoi(sample_id="native", split_group_key="polar",
                                           geographic_aoi=aoi, source_grid=source)
            self.assertEqual(native.target_grid.crs_wkt, source.crs_wkt)
            self.assertEqual(native.target_grid.transform[1], 1)
            validate_request_geographic_aoi(native)

    def test_rectangle_enclosing_pole_rejected(self):
        for name in ("LPS_N", "LPS_S"):
            grid = target_grid_from_bounds(crs_wkt=default_grid_registry()[name].crs_wkt,
                                           bounds=(499000, 499000, 501000, 501000),
                                           width=20, height=20)
            with self.assertRaises(UnsupportedCoverageError):
                geographic_aoi_from_target_grid(grid)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("osgeo", "numpy")),
                     "GDAL/NumPy unavailable")
class PolarChipPipelineTestCase(unittest.TestCase):
    def test_both_hemispheres_tiff_tasks_publish_serial_and_spawn(self):
        import numpy as np
        from osgeo import gdal
        from lfm.data_processing.chip import LabelInput, create_chips
        from lfm.data_processing.tests.chip.test_chip_creation import ChipCreationRasterTestCase
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = []
            for name, aoi in (("north", GeographicAOI(86.01, -.02, 86, .02)),
                              ("south", GeographicAOI(-86, -.02, -86.01, .02))):
                req = chip_request_from_aoi(sample_id=name, split_group_key=name,
                                            geographic_aoi=aoi, static_only=True)
                grid = req.target_grid
                path = root / f"{name}.tif"
                ds = gdal.GetDriverByName("GTiff").Create(str(path), grid.width, grid.height, 1, gdal.GDT_UInt16)
                ds.SetProjection(grid.crs_wkt)
                ds.SetGeoTransform(grid.transform)
                ds.GetRasterBand(1).WriteArray(np.full((grid.height, grid.width), 12, dtype=np.uint16))
                ds = None
                for kind in ("semantic", "raster_instance"):
                    requests.append(replace(req, sample_id=f"{name}_{kind}",
                                            label_input=LabelInput(path, kind=kind, relation="clip_to_target")))
            config = ChipCreationRasterTestCase().config(root)
            group = config.acquisition_groups[0]
            config = replace(config, acquisition_groups=(replace(group, tile_config=replace(group.tile_config, zoom_level=4)),))
            outputs = []
            for workers in (1, 2):
                with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=polar_acquire), \
                     patch("lfm.data_processing.chip.chip_creation._run_prepared_task", new=polar_worker):
                    batch = create_chips(requests, config, max_workers=workers, overwrite=workers == 2)
                self.assertEqual([r.status for r in batch.results], ["success"] * 4,
                                 [r.message for r in batch.results])
                outputs.append([(r.chip_path.read_bytes(), r.label_path.read_bytes()) for r in batch.results])
            self.assertEqual(outputs[0], outputs[1])
