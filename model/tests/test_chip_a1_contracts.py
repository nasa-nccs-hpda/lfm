"""A1 metadata/grid tests; real lunar transformations run when GDAL is available."""

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from lfm.model.chip_creation import ChipProgressEvent, _diagnostic_document
from lfm.model.chip_labels import preflight_label
from lfm.model.chip_preflight import PreparedChipRequest
from lfm.model.chip_publication import _sample_document
from lfm.model.chip_requests import (
    _projected_to_pixel,
    _select_source_grid,
    chip_request_from_aoi,
    raster_bounds,
    static_grid_reference,
    target_grid_from_geographic_aoi,
    target_grid_from_pixel_bounds,
    validate_request_geographic_aoi,
)
from lfm.model.chip_splits import SplitAssignment
from lfm.model.chip_types import (
    ChipDiagnostic, ChipPreflight, ChipRequest, ChipResult, GeographicAOI,
    LabelInput, LabelMismatchError, LabelPreparationPlan, LabelValidationDiagnostic,
    PreparedLabelArtifact, SourceSelector, TargetGrid,
)
from lfm.model.tests.test_chip_publication import simple_config


FIXTURES = Path(__file__).resolve().parents[2] / ".agents/planning_docs/aoi_chip_contract_fixtures.json"
HAS_GDAL = importlib.util.find_spec("osgeo") is not None
DIGEST = "a" * 64


def grid(transform=(0, 100, 0, 1000, 0, -100), width=10, height=10):
    return TargetGrid("lunar test CRS", transform, raster_bounds(transform, width, height), width, height)


def request(**kwargs):
    values = dict(sample_id="M1_r0_c0", target_grid=grid(),
                  geographic_aoi=GeographicAOI(2, -9, 1, -8), split_group_key="M1")
    values.update(kwargs)
    return ChipRequest(**values)


class LabelContractTestCase(unittest.TestCase):
    def test_legacy_label_normalizes_to_exact_without_io(self):
        item = request(label_path="/missing/M1_r0_c0_label.npz", label_grid=grid())
        self.assertEqual(item.label_input, LabelInput(item.label_path, "raster_instance", grid()))
        self.assertEqual(item.label_input.relation, "exact")
        self.assertIsNone(request().label_input)

    def test_explicit_label_does_not_need_sample_name(self):
        source = LabelInput("/labels/full_scene.gpkg", relation="clip_to_target")
        item = request(label_input=source)
        self.assertEqual(item.label_path, source.path)
        self.assertEqual(source.kind, "vector_instance")
        self.assertEqual(source.layer, "craters")

    def test_legacy_typed_conflicts_are_rejected(self):
        source = LabelInput("full.npy", source_grid=grid())
        with self.assertRaisesRegex(ValueError, "label_path conflicts"):
            request(label_input=source, label_path="different.npy")
        with self.assertRaisesRegex(ValueError, "label_grid conflicts"):
            request(label_input=source, label_grid=grid(width=20))

    def test_invalid_label_inputs(self):
        for kwargs in ({"path": ""}, {"kind": "unknown"}, {"relation": "resize"},
                       {"source_grid": {}}, {"layer": "craters"},
                       {"path": "a.gpkg", "source_grid": grid()}, {"sidecar_path": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                LabelInput(**({"path": "a.npy"} | kwargs))

    def test_plan_methods_and_hash_validation(self):
        source = LabelInput("full.npz", source_grid=grid(), relation="clip_to_target")
        target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 4))
        plan = LabelPreparationPlan(source, target, "aligned_window", DIGEST, (1, 1, 2, 3))
        self.assertEqual(plan.source_window, (1, 1, 2, 3))
        for changes in ({"source_sha256": "abc"}, {"source_window": None},
                        {"source_window": (9, 9, 2, 3)}, {"source_window": (0, 0, 1, 1)},
                        {"source_window": (True, 0, 2, 3)}, {"method": "nearest_warp"},
                        {"source": replace(source, relation="exact")}, {"diagnostics": ("bad",)}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(plan, **changes)
        with self.assertRaisesRegex(ValueError, "source_grid"):
            LabelPreparationPlan(LabelInput("a.npy", relation="clip_to_target"), target, "nearest_warp", DIGEST)
        with self.assertRaisesRegex(ValueError, "Vector"):
            LabelPreparationPlan(source, target, "vector_rasterize", DIGEST)

    def test_vector_artifact_and_compact_mapping(self):
        source = LabelInput("craters.gpkg", relation="clip_to_target")
        plan = LabelPreparationPlan(source, grid(), "vector_rasterize", DIGEST)
        artifact = PreparedLabelArtifact("derived.npz", plan, "b" * 64, ((12, 1), (38, 2)))
        self.assertEqual(artifact.kind, "raster_instance")
        self.assertEqual(artifact.target_grid, grid())
        for mapping in (((12, 1), (12, 2)), ((38, 1), (12, 2)), ((12, 2),), ((0, 1),)):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                replace(artifact, instance_id_map=mapping)
        with self.assertRaises(FrozenInstanceError):
            artifact.path = Path("changed.npz")

    def test_exact_artifact_preserves_checksum(self):
        plan = LabelPreparationPlan(LabelInput("a.npy", source_grid=grid()), grid(), "exact", DIGEST)
        with self.assertRaisesRegex(ValueError, "preserve"):
            PreparedLabelArtifact("a.npy", plan, "b" * 64)
        with self.assertRaisesRegex(ValueError, "must use"):
            PreparedLabelArtifact("a.npz", plan, DIGEST)
        raster_plan = replace(plan, source=LabelInput("full.tif", source_grid=grid()))
        # Same pixel grid, but a GeoTIFF must become a training NPY artifact.
        PreparedLabelArtifact("derived.npy", raster_plan, "b" * 64)

    def test_json_and_pickle_round_trips(self):
        diagnostic = LabelValidationDiagnostic("occluded", "Retained hidden annotation", "warning")
        source = LabelInput("full.gpkg", relation="clip_to_target", source_id="scene", sidecar_path="full.json")
        plan = LabelPreparationPlan(source, grid(), "vector_rasterize", DIGEST, diagnostics=(diagnostic,))
        artifact = PreparedLabelArtifact("derived.npz", plan, "b" * 64, ((90, 1),), (diagnostic,))
        item = request(label_input=source, requested_aoi=GeographicAOI(1.8, -8.9, 1.2, -8.1),
                       source_selectors=(SourceSelector("wac_grid", "wac", "M1"),))
        for record in (grid(), source, plan, artifact, item, request(label_path="a.npy")):
            with self.subTest(record=type(record).__name__):
                self.assertEqual(type(record).from_dict(json.loads(json.dumps(record.to_dict()))), record)
                self.assertEqual(pickle.loads(pickle.dumps(record)), record)
        malformed = item.to_dict() | {"unknown": "field"}
        with self.assertRaises(TypeError):
            ChipRequest.from_dict(malformed)
        malformed = item.to_dict()
        malformed["target_grid"]["width"] = 0
        with self.assertRaises(ValueError):
            ChipRequest.from_dict(malformed)

    def test_old_request_dictionary_still_loads(self):
        original = request(label_path="a.npy", label_grid=grid())
        old = original.to_dict()
        del old["label_input"]
        del old["requested_aoi"]
        self.assertEqual(ChipRequest.from_dict(old), original)

    def test_clip_request_is_not_silently_byte_copied(self):
        from lfm.model.chip_label_planning import require_materialized_label
        item = request(label_input=LabelInput("full.npz", relation="clip_to_target"))
        prepared = PreparedChipRequest(item, SplitAssignment(item.sample_id, item.split_group_key, "unsplit", "no_split"),
                                       ChipPreflight("passed", "unsplit"))
        with self.assertRaises(LabelMismatchError) as caught:
            require_materialized_label(prepared)
        self.assertEqual(caught.exception.diagnostics[0].code, "label_materialization_required")

    def test_provenance_and_new_stages(self):
        source = LabelInput("full.gpkg", relation="clip_to_target")
        plan = LabelPreparationPlan(source, grid(), "vector_rasterize", DIGEST)
        artifact = PreparedLabelArtifact("derived.npz", plan, "b" * 64)
        item = request(label_input=source, requested_aoi=GeographicAOI(1.8, -8.9, 1.2, -8.1))
        preflight = ChipPreflight("passed", "unsplit", source.path, label_plan=plan)
        assignment = SplitAssignment(item.sample_id, item.split_group_key, "unsplit", "no_split")
        prepared = PreparedChipRequest(item, assignment, preflight)
        result = ChipResult(item, "pending", preflight, prepared_label=artifact)
        manifest = _sample_document(prepared, result)
        diagnostic = _diagnostic_document(prepared, result, None, simple_config(Path("/tmp/a1")))
        for document in (manifest, diagnostic):
            self.assertEqual(document["label_preparation_plan"], plan.to_dict())
            self.assertEqual(document["prepared_label"], artifact.to_dict())
            self.assertEqual(document["requested_aoi"], item.requested_aoi.to_dict())
            self.assertEqual(document["geographic_aoi"], item.geographic_aoi.to_dict())
            self.assertEqual(document["target_grid"], item.target_grid.to_dict())
            json.dumps(document, allow_nan=False)
        ChipDiagnostic("label_preparation", "occluded", "Retained occluded instance", "warning")
        ChipProgressEvent(item.sample_id, "label/clip", "started", worker_pid=1)
        with self.assertRaises(ValueError):
            ChipResult(replace(item, target_grid=grid(width=20)), "pending", preflight, prepared_label=artifact)


class GeographicGridContractTestCase(unittest.TestCase):
    def test_a0_window_fixtures_use_production_rounding(self):
        fixtures = json.loads(FIXTURES.read_text())
        for case in fixtures["grid_windows"]:
            with self.subTest(case=case["name"]):
                result = target_grid_from_pixel_bounds(grid(tuple(case["source_affine_gdal"])), case["pixel_bounds"])
                self.assertEqual(result.transform, tuple(case["output_affine_gdal"]))
                self.assertEqual((result.width, result.height), tuple(case["window"][2:]))

    def test_a0_static_zone_fixtures(self):
        for case in json.loads(FIXTURES.read_text())["static_grids"]:
            west, south, east, north = case["aoi_wsen"]
            result = static_grid_reference(GeographicAOI(north, west, south, east))
            self.assertIn(f"LTM_{case['zone']}", result.crs_wkt)
            self.assertEqual(result.transform, (0, 100, 0, 0, 0, -100))

    def test_a0_static_rounding_fixture(self):
        case = json.loads(FIXTURES.read_text())["static_rounding"]
        source = static_grid_reference(GeographicAOI(-8, -9, -10, -7))
        left, bottom, right, top = case["projected_bounds_xyxy"]
        result = target_grid_from_pixel_bounds(source, (left / 100, -top / 100, right / 100, -bottom / 100))
        self.assertEqual(result.bounds, tuple(case["output_bounds_xyxy"]))
        self.assertEqual((result.height, result.width), tuple(case["shape_hw"]))

    def test_wac_precedence_and_competing_lattices(self):
        with self.assertWarnsRegex(UserWarning, "WAC takes precedence"):
            selected = _select_source_grid({"nac": (grid(width=20),), "wac": (grid(),)})
        self.assertEqual(selected, grid())
        shifted = grid((100, 100, 0, 1000, 0, -100))
        self.assertEqual(_select_source_grid({"wac": (grid(), shifted)}), grid())
        with self.assertRaisesRegex(ValueError, "Competing"):
            _select_source_grid({"wac": (grid(), grid((50, 100, 0, 1000, 0, -100)))})
        with self.assertRaisesRegex(ValueError, "explicit source_grid"):
            _select_source_grid({"nac": (grid(),), "other": (grid(),)})

    def test_constructor_rejects_mixed_or_missing_grid_modes(self):
        aoi = GeographicAOI(2, -9, 1, -8)
        for kwargs in ({}, {"geographic_aoi": aoi}, {"source_grid": grid()},
                       {"geographic_aoi": aoi, "source_grid": grid(), "static_only": True},
                       {"geographic_aoi": aoi, "source_grid": grid(), "width": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chip_request_from_aoi(sample_id="M1", split_group_key="M1", **kwargs)

    def test_constructor_retains_requested_and_realized_aois(self):
        original = GeographicAOI(1.8, -8.9, 1.2, -8.1)
        realized = GeographicAOI(2, -9, 1, -8)
        with patch("lfm.model.chip_requests.target_grid_from_geographic_aoi", return_value=grid()), \
             patch("lfm.model.chip_requests.geographic_aoi_from_target_grid", return_value=realized):
            result = chip_request_from_aoi(sample_id="M1", split_group_key="M1", geographic_aoi=original,
                                           source_grid=grid(), label_input=LabelInput("full.gpkg", relation="clip_to_target"))
        self.assertEqual(result.requested_aoi, original)
        self.assertEqual(result.geographic_aoi, realized)
        self.assertIsNone(result.reference_path)

    def test_adaptive_edge_densification_captures_curvature(self):
        # Smooth geographic-to-pixel mapping with a curved edge. Real algorithm,
        # mocked coordinate transform only; maximum row is between coarse samples.
        class CurvedTransform:
            def TransformPoint(self, x, y):
                return x * 100, y * 100 + 130 * (1 - (x - 0.517) ** 2), 0

        source = grid((0, 100, 0, 0, 0, 100))
        with patch("lfm.model.chip_requests._spatial_reference"), \
             patch("lfm.model.chip_requests._create_transformation", return_value=CurvedTransform()):
            result = target_grid_from_geographic_aoi(GeographicAOI(1, 0, 0, 1), source)
        self.assertEqual(result.height, 3)
        self.assertEqual(result.width, 1)

    def test_discontinuous_transform_fails_instead_of_inventing_grid(self):
        class DiscontinuousTransform:
            def TransformPoint(self, x, y):
                return x * 100, y * 100 + (1000 if x >= 0.517 else 0), 0

        with patch("lfm.model.chip_requests._spatial_reference"), \
             patch("lfm.model.chip_requests._create_transformation", return_value=DiscontinuousTransform()):
            with self.assertRaisesRegex(ValueError, "did not converge"):
                target_grid_from_geographic_aoi(GeographicAOI(1, 0, 0, 1), grid())

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_real_antimeridian_and_rotated_native_grid(self):
        for aoi in (GeographicAOI(2.01, 179.99, 2, -179.99),
                    GeographicAOI(-8, -8.01, -8.01, -8)):
            reference = static_grid_reference(aoi)
            affine = (150000, 80, 60, 2250000, 60, -80)
            rotated = TargetGrid(reference.crs_wkt, affine, raster_bounds(affine, 100, 100), 100, 100)
            item = chip_request_from_aoi(sample_id="M1", split_group_key="M1", geographic_aoi=aoi,
                                         source_grid=rotated)
            self.assertEqual(tuple(item.target_grid.transform[i] for i in (1, 2, 4, 5)), (80, 60, 60, -80))
            origin = _projected_to_pixel(rotated, item.target_grid.transform[0], item.target_grid.transform[3])
            for coordinate in origin:
                self.assertAlmostEqual(coordinate, round(coordinate), places=7)
            validate_request_geographic_aoi(item)

    @unittest.skipUnless(HAS_GDAL, "GDAL Python bindings are unavailable")
    def test_real_lunar_aoi_native_grid_and_static_grid(self):
        aoi = GeographicAOI(-8.001, -8.01, -8.02, -8.001)
        source = static_grid_reference(aoi)
        # Different origin from tiling LTM: preserve a legacy source's offsets.
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(source.crs_wkt)
        srs.SetProjParm("false_northing", 2500000)
        source = replace(source, crs_wkt=srs.ExportToWkt())
        for kwargs in ({"source_grid": source}, {"static_only": True}):
            item = chip_request_from_aoi(sample_id="M1", split_group_key="M1", geographic_aoi=aoi,
                                         label_input=LabelInput("full.gpkg", relation="clip_to_target"), **kwargs)
            self.assertGreater(item.target_grid.width, 0)
            self.assertEqual(item.target_grid.transform[1:3], (100, 0))
            validate_request_geographic_aoi(item)
            if "source_grid" in kwargs:
                self.assertEqual(item.target_grid.crs_wkt, source.crs_wkt)
            # Independently densely sample the requested boundary to check that
            # outward-rounded output contains it (not just its four corners).
            from lfm.model.chip_requests import _create_transformation, _spatial_reference
            from lfm.model.lunar_crs import load_lunar_geographic_wkt
            transform = _create_transformation(_spatial_reference(load_lunar_geographic_wkt()),
                                               _spatial_reference(item.target_grid.crs_wkt))
            for i in range(101):
                fraction = i / 100
                x = -8.01 + 0.009 * fraction
                y = -8.02 + 0.019 * fraction
                for lon, lat in ((x, -8.001), (x, -8.02), (-8.01, y), (-8.001, y)):
                    px, py = _projected_to_pixel(item.target_grid, *transform.TransformPoint(lon, lat)[:2])
                    self.assertTrue(-1e-8 <= px <= item.target_grid.width + 1e-8)
                    self.assertTrue(-1e-8 <= py <= item.target_grid.height + 1e-8)


if __name__ == "__main__":
    unittest.main()
