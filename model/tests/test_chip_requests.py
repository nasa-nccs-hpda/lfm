import json
from pathlib import Path
import tempfile
import unittest

from lfm.model.chip_requests import (
    AmbiguousLongitudeError,
    UnsupportedCoverageError,
    chip_request_from_aoi,
    discover_reference_tiffs,
    geographic_aoi_from_target_grid,
    geographic_query_parts,
    longitude_envelope,
    materialize_requests,
    normalize_sample_id,
    product_id_from_sample_id,
    reference_sample_from_tiff,
    target_grid_from_bounds,
    validate_numbered_ltm_coverage,
    validate_request_geographic_aoi,
    validate_target_grid_consistency,
)
from lfm.model.chip_types import ChipRequest, GeographicAOI, TargetGrid
from lfm.model.lunar_crs import load_lunar_geographic_wkt

try:
    from osgeo import gdal, osr

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class ChipRequestTestCase(unittest.TestCase):
    def grid(self):
        return TargetGrid(
            crs_wkt="GEOGCRS[Moon]",
            transform=(149.7, 0.001, 0.0, 1.3, 0.0, -0.001),
            bounds=(149.7, 1.0, 150.0, 1.3),
            width=300,
            height=300,
        )

    def request(self, sample_id, product="M1187363083CE"):
        return ChipRequest(
            sample_id=sample_id,
            target_grid=self.grid(),
            geographic_aoi=GeographicAOI(1.3, 149.7, 1.0, 150.0),
            split_group_key=product,
        )

    def test_sample_and_product_identity_are_distinct(self):
        name = "M1187363083CE_r12750_c1500_input_wac_chip.tif"

        sample_id = normalize_sample_id(name)

        self.assertEqual(sample_id, "M1187363083CE_r12750_c1500")
        self.assertEqual(product_id_from_sample_id(sample_id), "M1187363083CE")
        self.assertEqual(
            normalize_sample_id(
                "M1187363083CE_r12750_c1500_input_nac_static_chip.tif"
            ),
            "M1187363083CE_r12750_c1500",
        )

    def test_reference_discovery_is_sorted_and_tiff_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("b.tiff", "A.tif", "ignore.npy"):
                (root / name).touch()

            paths = discover_reference_tiffs(root)

        self.assertEqual([path.name for path in paths], ["A.tif", "b.tiff"])

    def test_request_materialization_is_deterministic_and_limits_after_sort(self):
        requests = (
            self.request("M1_r300_c0", product="M1"),
            self.request("M1_r0_c0", product="M1"),
        )

        result = materialize_requests(iter(requests), sample_limit=1)

        self.assertEqual(tuple(item.sample_id for item in result), ("M1_r0_c0",))

    def test_target_grid_derives_affine_and_rejects_inconsistent_grid(self):
        grid = target_grid_from_bounds(
            crs_wkt="GEOGCRS[Moon]",
            bounds=(149.7, 1.0, 150.0, 1.3),
            width=300,
            height=300,
        )

        self.assertEqual(grid.transform[::3], (149.7, 1.3))
        self.assertAlmostEqual(grid.transform[1], 0.001)
        self.assertAlmostEqual(grid.transform[5], -0.001)
        inconsistent = TargetGrid(
            crs_wkt=grid.crs_wkt,
            transform=grid.transform,
            bounds=(149.7, 1.0, 150.1, 1.3),
            width=grid.width,
            height=grid.height,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_target_grid_consistency(inconsistent)

    def test_antimeridian_aoi_becomes_two_non_wrapping_queries(self):
        west, east, span, crosses = longitude_envelope((179.8, -179.8))
        aoi = GeographicAOI(1.0, west, 0.8, east)

        parts = geographic_query_parts(aoi)

        self.assertAlmostEqual(span, 0.4)
        self.assertTrue(crosses)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0].upper_left_longitude, 179.8)
        self.assertEqual(parts[0].lower_right_longitude, 180.0)
        self.assertEqual(parts[1].upper_left_longitude, -180.0)
        self.assertAlmostEqual(parts[1].lower_right_longitude, -179.8)

    def test_longitude_span_of_180_degrees_is_ambiguous(self):
        with self.assertRaises(AmbiguousLongitudeError):
            longitude_envelope((0.0, 180.0))

    def test_polar_coverage_is_rejected_but_overlap_is_supported(self):
        validate_numbered_ltm_coverage(GeographicAOI(82.0, 10.0, 80.0, 11.0))
        with self.assertRaisesRegex(UnsupportedCoverageError, "numbered LTM"):
            validate_numbered_ltm_coverage(
                GeographicAOI(82.1, 10.0, 81.0, 11.0)
            )

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_explicit_geographic_aoi_produces_exact_grid(self):
        request = chip_request_from_aoi(
            sample_id="M1_r0_c0",
            crs_wkt=load_lunar_geographic_wkt(),
            bounds=(149.7, 1.0, 150.0, 1.3),
            width=300,
            height=300,
            split_group_key="M1",
        )

        self.assertEqual(request.target_grid.bounds, (149.7, 1.0, 150.0, 1.3))
        self.assertAlmostEqual(request.geographic_aoi.upper_left_latitude, 1.3)
        self.assertAlmostEqual(request.geographic_aoi.lower_right_longitude, 150.0)
        self.assertEqual(len(validate_request_geographic_aoi(request)), 1)
        inconsistent = ChipRequest(
            sample_id=request.sample_id,
            target_grid=request.target_grid,
            geographic_aoi=GeographicAOI(1.3, 149.7, 1.1, 150.0),
            split_group_key=request.split_group_key,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_request_geographic_aoi(inconsistent)

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_projected_reference_tiff_preserves_grid_and_band_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "M1_r0_c0_input_nac_chip.tif"
            zone_definition = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "TMS"
                    / "RG"
                    / "tms_LTM_42NRG.json"
                ).read_text(encoding="utf-8")
            )
            source = osr.SpatialReference()
            source.ImportFromWkt(zone_definition["crs"])
            target = osr.SpatialReference()
            target.ImportFromWkt(load_lunar_geographic_wkt())
            source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            with osr.ExceptionMgr(useExceptions=False):
                transform = osr.CoordinateTransformation(target, source)
            self.assertIsNotNone(transform)
            upper_left = transform.TransformPoint(149.7, 1.3)
            lower_right = transform.TransformPoint(150.0, 1.0)
            geotransform = (
                upper_left[0],
                (lower_right[0] - upper_left[0]) / 30,
                0.0,
                upper_left[1],
                0.0,
                (lower_right[1] - upper_left[1]) / 30,
            )
            dataset = gdal.GetDriverByName("GTiff").Create(
                str(path),
                30,
                30,
                2,
                gdal.GDT_Byte,
            )
            dataset.SetProjection(source.ExportToWkt())
            dataset.SetGeoTransform(geotransform)
            dataset.GetRasterBand(1).SetDescription("pho")
            dataset.GetRasterBand(2).SetDescription("dtm")
            dataset = None

            reference = reference_sample_from_tiff(path)

        self.assertEqual(reference.sample_id, "M1_r0_c0")
        self.assertEqual(reference.band_count, 2)
        self.assertEqual(reference.band_descriptions, ("pho", "dtm"))
        self.assertEqual(reference.target_grid.transform, geotransform)
        self.assertLess(reference.geographic_aoi.lower_right_latitude, 1.3)

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_reference_tiff_rejects_missing_crs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "M1_r0_c0_input_chip.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(
                str(path),
                4,
                3,
                1,
                gdal.GDT_Byte,
            )
            dataset.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
            dataset = None

            with self.assertRaisesRegex(ValueError, "CRS"):
                reference_sample_from_tiff(path)

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_projected_grid_envelope_crosses_ltm_zone_boundary(self):
        zone_definition = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "TMS"
                / "RG"
                / "tms_LTM_42NRG.json"
            ).read_text(encoding="utf-8")
        )
        projected = osr.SpatialReference()
        projected.ImportFromWkt(zone_definition["crs"])
        geographic = osr.SpatialReference()
        geographic.ImportFromWkt(load_lunar_geographic_wkt())
        projected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        with osr.ExceptionMgr(useExceptions=False):
            transformation = osr.CoordinateTransformation(geographic, projected)
        self.assertIsNotNone(transformation)
        points = (
            transformation.TransformPoint(155.9, 1.2),
            transformation.TransformPoint(156.1, 1.0),
        )
        xs = tuple(point[0] for point in points)
        ys = tuple(point[1] for point in points)
        grid = target_grid_from_bounds(
            crs_wkt=projected.ExportToWkt(),
            bounds=(min(xs), min(ys), max(xs), max(ys)),
            width=40,
            height=40,
        )

        aoi = geographic_aoi_from_target_grid(grid)

        self.assertLess(aoi.upper_left_longitude, 156.0)
        self.assertGreater(aoi.lower_right_longitude, 156.0)
        self.assertEqual(len(geographic_query_parts(aoi)), 1)

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_transformed_antimeridian_and_polar_boundaries(self):
        antimeridian = chip_request_from_aoi(
            sample_id="M1_r0_c0",
            crs_wkt=load_lunar_geographic_wkt(),
            bounds=(179.8, 0.8, 180.2, 1.0),
            width=40,
            height=20,
            split_group_key="site-a",
        )

        self.assertEqual(len(geographic_query_parts(antimeridian.geographic_aoi)), 2)
        with self.assertRaises(UnsupportedCoverageError):
            chip_request_from_aoi(
                sample_id="M2_r0_c0",
                crs_wkt=load_lunar_geographic_wkt(),
                bounds=(10.0, 81.9, 10.2, 82.1),
                width=20,
                height=20,
                split_group_key="site-b",
            )


if __name__ == "__main__":
    unittest.main()
