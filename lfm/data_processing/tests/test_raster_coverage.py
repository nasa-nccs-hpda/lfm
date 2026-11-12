"""Coverage geometry and Jupyter-compatible spawn worker checks."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import Affine, from_bounds
from shapely.geometry import box

from lfm.data_processing.raster_coverage import (
    coverage_context, inspect_raster_coverage, inspect_raster_footprint, rank_raster_pairs,
)


class RasterCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wkt = (Path(__file__).resolve().parents[3] / "TMS/IAU_30100_2015.wkt").read_text()
        cls.aoi = (-20.9333825982, 147.9653785664, -21.8923820897, 148.9955331735)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def raster(self, name, transform, crs=None, nodata=None):
        path = Path(self.temp.name) / name
        with rasterio.open(path, "w", driver="GTiff", width=32, height=32,
                           count=1, dtype="uint8", crs=crs, transform=transform,
                           nodata=nodata) as dataset:
            dataset.write(np.zeros((1, 32, 32), dtype="uint8"))
        return path

    def inspect(self, path):
        return inspect_raster_coverage(path, aoi_nwse=self.aoi, lunar_wkt=self.wkt)

    def geographic_raster(self, name, west, south, east, north, **kwargs):
        return self.raster(name, from_bounds(west, south, east, north, 32, 32),
                           crs=self.wkt, **kwargs)

    def test_full_half_disjoint_and_nodata_margins(self):
        north, west, south, east = self.aoi
        for name, bounds, expected in (
            ("full.tif", (west, south, east, north), 100),
            ("half.tif", (west, south, (west + east) / 2, north), 50),
            ("far.tif", (-40, 30, -39, 31), 0),
        ):
            with self.subTest(name=name):
                # All pixels are NoData: footprint coverage must still count.
                result = self.inspect(self.geographic_raster(name, *bounds, nodata=0))
                self.assertIsNone(result["error"])
                self.assertAlmostEqual(result["overlap_percent"], expected, places=4)

    def test_rotated_raster_uses_pixel_perimeter_not_axis_aligned_bounds(self):
        north, west, south, east = self.aoi
        # Diamond inscribed in the AOI: its bounding box covers 100%,
        # but the raster itself covers approximately 50%.
        transform = Affine((east - west) / 64, -(east - west) / 64, (west + east) / 2,
                           -(north - south) / 64, -(north - south) / 64, north)
        result = self.inspect(self.raster("rotated.tif", transform, crs=self.wkt))
        self.assertIsNone(result["error"])
        self.assertAlmostEqual(result["overlap_percent"], 50, places=2)

    def test_projected_lunar_scene_inside_aoi(self):
        north, west, south, east = self.aoi
        crs = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=148 +k=0.999 +R=1737400 +units=m")
        project = Transformer.from_crs(CRS.from_wkt(self.wkt), crs, always_xy=True)
        x, y = project.transform((west + east) / 2, (north + south) / 2)
        path = self.raster("projected.tif", from_bounds(x - 500, y - 500, x + 500, y + 500, 32, 32), crs=crs)
        result = self.inspect(path)
        self.assertIsNone(result["error"])
        aoi_area = coverage_context(self.aoi, self.wkt)[3]
        self.assertAlmostEqual(result["overlap_percent"], 100 * 1_000_000 / (0.999**2 * aoi_area), places=4)

    def test_failures_are_not_zero_overlap(self):
        missing_crs = self.raster("missing_crs.tif", from_bounds(147, -22, 149, -20, 32, 32))
        corrupt = Path(self.temp.name) / "corrupt.tif"
        corrupt.write_text("not a raster")
        earth = self.raster("earth.tif", from_bounds(147, -22, 149, -20, 32, 32), crs="EPSG:4326")
        for path in (missing_crs, corrupt, earth):
            with self.subTest(path=path.name):
                result = self.inspect(path)
                self.assertIsNone(result["overlap_percent"])
                self.assertTrue(result["error"])

    def test_ten_worker_spawn_matches_serial(self):
        north, west, south, east = self.aoi
        paths = [self.geographic_raster(f"scene_{i}.tif", west, south,
                                       west + (east - west) * (i + 1) / 10, north)
                 for i in range(10)]
        worker = partial(inspect_raster_footprint, lunar_wkt=self.wkt)
        with ProcessPoolExecutor(max_workers=10, mp_context=multiprocessing.get_context("spawn")) as pool:
            parallel = list(pool.map(worker, paths))
        self.assertEqual(parallel, [worker(path) for path in paths])
        self.assertTrue(all(result["error"] is None for result in parallel))

    def test_pair_ranking_requires_shared_area_and_considers_all_candidates(self):
        # Five larger WAC footprints miss the NAC entirely. The sixth is the
        # only usable pair. Percentage must use that WAC's own area.
        wac = {f"large_{index}.tif": box(0, 0, 6, 10).wkb_hex for index in range(5)}
        wac["smaller.vis.tif"] = box(6, 0, 10, 10).wkb_hex
        nac = {"nac.tif": box(6, 0, 10, 10).wkb_hex}
        result = rank_raster_pairs(wac, nac, lunar_wkt=self.wkt)
        pairs = result["top_pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["wac_filename"], "smaller.vis.tif")
        self.assertAlmostEqual(pairs[0]["wac_coverage_percent"], 100)
        self.assertEqual(result["positive_pair_count"], 1)
        for index in range(5):
            self.assertEqual(result["overlap_percent_by_wac"][f"large_{index}.tif"], {"nac.tif": 0.0})

    def test_pair_rank_ties_limit_and_empty_collections(self):
        geometry = box(0, 0, 10, 10).wkb_hex
        wac = {"b.uv.tif": geometry, "a.vis.tif": geometry}
        nac = {"b.nac.tif": geometry, "a.nac.tif": geometry}
        result = rank_raster_pairs(wac, nac, lunar_wkt=self.wkt, limit=2)
        self.assertEqual([(p["wac_filename"], p["nac_filename"]) for p in result["top_pairs"]],
                         [("a.vis.tif", "a.nac.tif"), ("a.vis.tif", "b.nac.tif")])
        self.assertEqual(rank_raster_pairs({}, nac, lunar_wkt=self.wkt)["top_pairs"], [])
        self.assertEqual(rank_raster_pairs(wac, {}, lunar_wkt=self.wkt)["overlap_percent_by_wac"],
                         {"a.vis.tif": {}, "b.uv.tif": {}})

    def test_full_matrix_includes_zero_and_failed_files(self):
        geometry = box(0, 0, 1, 1).wkb_hex
        wac = {"valid.vis.tif": geometry, "broken.uv.tif": None}
        nac = {"valid.nac.tif": geometry, "far.nac.tif": box(20, 20, 21, 21).wkb_hex,
               "broken.nac.tif": None}
        result = rank_raster_pairs(wac, nac, lunar_wkt=self.wkt)
        matrix = result["overlap_percent_by_wac"]
        self.assertEqual(set(matrix), set(wac))
        self.assertTrue(all(set(row) == set(nac) for row in matrix.values()))
        self.assertEqual(sum(len(row) for row in matrix.values()), 6)
        self.assertEqual(matrix["valid.vis.tif"],
                         {"valid.nac.tif": 100.0, "far.nac.tif": 0.0, "broken.nac.tif": None})
        self.assertTrue(all(value is None for value in matrix["broken.uv.tif"].values()))
        self.assertEqual(len(result["top_pairs"]), 1)

    def test_pair_ranking_from_full_inspection_outside_original_aoi(self):
        # Deliberately far from the original poster AOI: there is no AOI gate.
        wac = self.geographic_raster("wac.vis.tif", -41, 30, -39, 31)
        nac = self.geographic_raster("nac.tif", -41, 30, -40, 31)
        worker = partial(inspect_raster_footprint, lunar_wkt=self.wkt)
        wac_result, nac_result = worker(wac), worker(nac)
        self.assertIsNone(wac_result["error"])
        self.assertIsNone(nac_result["error"])
        result = rank_raster_pairs(
            {str(wac): wac_result["footprint_wkb_hex"]},
            {str(nac): nac_result["footprint_wkb_hex"]}, lunar_wkt=self.wkt,
        )
        self.assertAlmostEqual(result["top_pairs"][0]["wac_coverage_percent"], 50, places=4)

    def test_antimeridian_overlap_is_found_once(self):
        wac = self.geographic_raster("seam.vis.tif", 179, -1, 181, 1)
        nac = self.geographic_raster("seam.nac.tif", -180, -1, -179, 1)
        worker = partial(inspect_raster_footprint, lunar_wkt=self.wkt)
        wac_result, nac_result = worker(wac), worker(nac)
        self.assertIsNone(wac_result["error"])
        self.assertIsNone(nac_result["error"])
        result = rank_raster_pairs(
            {str(wac): wac_result["footprint_wkb_hex"]},
            {str(nac): nac_result["footprint_wkb_hex"]}, lunar_wkt=self.wkt,
        )
        self.assertEqual(result["positive_pair_count"], 1)
        self.assertAlmostEqual(result["top_pairs"][0]["wac_coverage_percent"], 50, places=4)


if __name__ == "__main__":
    unittest.main()
