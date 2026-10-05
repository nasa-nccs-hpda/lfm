"""Read-only planning tests, with real array/GDAL fixtures when available."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from lfm.model.chip_acquisition import acquire_prepared_request
from lfm.model.chip_creation import create_chip
from lfm.model.chip_label_planning import (
    _pixel_mapper, _resolve_grid, classify_label_grid, plan_label_preparation,
)
from lfm.model.chip_labels import preflight_label, resolve_label_path
from lfm.model.chip_preflight import PreparedChipRequest, preflight_chip_requests
from lfm.model.chip_requests import raster_bounds, static_grid_reference, target_grid_from_pixel_bounds
from lfm.model.chip_splits import SplitAssignment
from lfm.model.chip_types import (
    ChipPreflight, ChipRequest, GeographicAOI, LabelInput, LabelMismatchError,
    LabelPreparationPlan, TargetGrid,
)
from lfm.model.tests.test_chip_publication import simple_config


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_GDAL = importlib.util.find_spec("osgeo") is not None
FIXTURES = Path(__file__).resolve().parents[2] / ".agents/planning_docs/aoi_chip_contract_fixtures.json"


def grid(width=4, height=3, affine=(0, 100, 0, 300, 0, -100)):
    reference = static_grid_reference(GeographicAOI(2, -9, 1, -8))
    return TargetGrid(reference.crs_wkt, affine, raster_bounds(affine, width, height), width, height)


def request(path=None, source_grid=None, target=None, relation="clip_to_target", **label_kwargs):
    return ChipRequest("M1_r0_c0", target or grid(), GeographicAOI(2, -9, 1, -8), "M1",
                       label_input=None if path is None else LabelInput(path, source_grid=source_grid,
                                                                       relation=relation, **label_kwargs))


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class PlanningMetadataTestCase(unittest.TestCase):
    def test_derivative_probes_stay_inside_small_target_grids(self):
        for width, height in ((2, 1), (1, 2), (1, 1)):
            target = grid(width, height)
            source = grid(width + 2, height + 2)

            def bounded_mapper(col, row):
                self.assertTrue(0 <= col <= width, (col, row))
                self.assertTrue(0 <= row <= height, (col, row))
                return col + 1, row + 1

            for same_crs in (True, False):
                with self.subTest(width=width, height=height, same_crs=same_crs):
                    with patch("lfm.model.chip_label_planning._pixel_mapper",
                               return_value=(bounded_mapper, same_crs)):
                        relation = classify_label_grid(request(target=target), source)
                    expected = ("aligned_window", (1, 1, width, height)) if same_crs else ("nearest_warp", None)
                    self.assertEqual(relation, expected)

    def test_invalid_round_trip_is_still_rejected_with_coordinates(self):
        srs = SimpleNamespace(IsGeographic=lambda: False, Clone=lambda: srs)
        forward = SimpleNamespace(TransformPoint=lambda x, y: (x, y, 0))
        reverse = SimpleNamespace(TransformPoint=lambda x, y: (x + 1, y, 0))
        with patch("lfm.model.chip_label_planning._lunar_srs", return_value=srs), \
             patch("lfm.model.chip_label_planning._crs_is_same", return_value=False), \
             patch("lfm.model.chip_label_planning._create_transformation", side_effect=(forward, reverse)):
            mapper, _ = _pixel_mapper(request(), grid())
            with self.assertRaises(LabelMismatchError) as caught:
                mapper(0.5, 0.5)
        diagnostic = caught.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, "invalid_label_transform")
        self.assertIn("at (0.5, 0.5)", diagnostic.message)
        self.assertIn("error=0.01 pixels", diagnostic.message)
        self.assertEqual(diagnostic.expected, "(0.5, 0.5)")

    def test_affine_classification_and_pixel_space_coverage_without_gdal(self):
        srs = SimpleNamespace(IsGeographic=lambda: False)
        with patch("lfm.model.chip_label_planning._lunar_srs", return_value=srs):
            source = grid()
            self.assertEqual(classify_label_grid(request(), source), ("exact", None))
            target = target_grid_from_pixel_bounds(source, (1, 1, 3, 3))
            self.assertEqual(classify_label_grid(request(target=target), source),
                             ("aligned_window", (1, 1, 2, 2)))
            target = grid(2, 1, (50, 100, 0, 200, 0, -100))
            self.assertEqual(classify_label_grid(request(target=target), source), ("nearest_warp", None))
            target = grid(affine=(-0.01, 100, 0, 300, 0, -100))
            with self.assertRaises(LabelMismatchError) as caught:
                classify_label_grid(request(target=target), source)
            self.assertEqual(caught.exception.diagnostics[0].code, "incomplete_label_coverage")

    def test_curved_footprint_cannot_pass_using_only_its_corners(self):
        def curved(col, row):
            return col, row - 0.1 * col * (4 - col)

        with patch("lfm.model.chip_label_planning._pixel_mapper", return_value=(curved, False)):
            with self.assertRaises(LabelMismatchError) as caught:
                classify_label_grid(request(), grid())
        self.assertEqual(caught.exception.diagnostics[0].code, "incomplete_label_coverage")

    def test_explicit_path_and_file_source_bypass_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            label = root / "finished_whole_scene.npy"
            label.write_bytes(b"placeholder")
            self.assertEqual(resolve_label_path(request(label), root), label)
            self.assertEqual(resolve_label_path(request(), label), label)
            with self.assertRaises(LabelMismatchError):
                resolve_label_path(request(), root)

    def test_missing_grid_and_ambiguous_sidecars_are_typed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.npy"
            path.write_bytes(b"placeholder")
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "missing_label_grid")
            for sidecar in (path.with_suffix(".json"), path.with_suffix(".npy.json")):
                sidecar.write_text(json.dumps({"source_grid": grid().to_dict()}))
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "ambiguous_label_sidecar")
            explicit = request(path, sidecar_path=path.with_suffix(".json"))
            self.assertEqual(_resolve_grid(explicit, explicit.label_input).source_grid, grid())

    def test_source_sidecar_is_not_an_identity_gate_and_legacy_is_not_scene_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.npy"
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps({"sample_id": "other-scene", "source_grid": grid().to_dict()}))
            target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
            item = request(path, target=target)
            self.assertEqual(_resolve_grid(item, item.label_input).source_grid, grid())
            sidecar.write_text(json.dumps({"target_grid": grid().to_dict()}))
            with self.assertRaises(LabelMismatchError):
                _resolve_grid(item, item.label_input)
            sidecar.write_text(json.dumps({"target_grid": grid().to_dict(), "source_grid": grid().to_dict()}))
            with self.assertRaises(LabelMismatchError) as caught:
                _resolve_grid(item, item.label_input)
            self.assertEqual(caught.exception.diagnostics[0].code, "malformed_label_metadata")

    def test_inconsistent_embedded_or_sidecar_grid_fails(self):
        item = request("a.npy", source_grid=grid())
        with self.assertRaises(LabelMismatchError):
            _resolve_grid(item, item.label_input, grid(affine=(0.00001, 100, 0, 300, 0, -100)))

    def test_read_only_preflight_and_failed_sample_do_not_call_tiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good, bad = root / "good.npy", root / "bad.npy"
            good.write_bytes(b"mock content")
            bad.write_bytes(b"no grid")
            first = request(good, relation="exact")
            second = replace(request(bad), sample_id="M2_r0_c0", split_group_key="M2")
            before = snapshot(root)
            with patch("lfm.model.chip_preflight.validate_request_geographic_aoi"), \
                 patch("lfm.model.chip_label_planning.validate_label", return_value=()), \
                 patch("lfm.model.tiling.create_tiles_for_aoi") as tiling:
                batch = preflight_chip_requests((first, second), simple_config(root))
                with patch("lfm.model.chip_creation.acquire_prepared_request") as acquire:
                    with self.assertRaises(LabelMismatchError):
                        create_chip(batch.requests[1], simple_config(root))
                    acquire.assert_not_called()
            self.assertEqual([item.preflight.status for item in batch.requests], ["passed", "failed"])
            self.assertEqual(batch.requests[1].preflight.label_diagnostics[0].code, "missing_label_grid")
            self.assertEqual(snapshot(root), before)
            self.assertFalse((root / "dataset").exists())
            tiling.assert_not_called()

    def test_pending_materialization_is_blocked_before_tiling_or_cleanup(self):
        item = request("full.gpkg")
        plan = LabelPreparationPlan(item.label_input, item.target_grid, "vector_rasterize", "a" * 64)
        prepared = PreparedChipRequest(item, SplitAssignment(item.sample_id, "M1", "train", "explicit"),
                                       ChipPreflight("passed", "train", item.label_path, label_plan=plan))
        with tempfile.TemporaryDirectory() as temp:
            config = simple_config(Path(temp))
            with patch("lfm.model.chip_creation.acquire_prepared_request") as acquire, \
                 patch("lfm.model.chip_creation._clear_sample_intermediates") as cleanup:
                with self.assertRaises(LabelMismatchError) as caught:
                    create_chip(prepared, config, overwrite=True)
            self.assertEqual(caught.exception.diagnostics[0].code, "label_preparation_not_available")
            acquire.assert_not_called()
            cleanup.assert_not_called()
            with patch("lfm.model.chip_acquisition.derive_source_selectors") as select:
                with self.assertRaises(LabelMismatchError):
                    acquire_prepared_request(prepared, config)
            select.assert_not_called()
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_mutating_source_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.npy"
            path.write_bytes(b"before")

            def mutate(*args):
                path.write_bytes(b"after")
                return ()

            with patch("lfm.model.chip_label_planning.validate_label", side_effect=mutate):
                with self.assertRaises(LabelMismatchError) as caught:
                    plan_label_preparation(request(path, relation="exact"), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "label_source_changed")


@unittest.skipUnless(HAS_GDAL, "GDAL is unavailable")
class GridRelationTestCase(unittest.TestCase):
    def test_exact_aligned_rotated_and_nearest_relations(self):
        source = grid()
        self.assertEqual(classify_label_grid(request(target=source), source), ("exact", None))
        target = target_grid_from_pixel_bounds(source, (1, 1, 3, 3))
        self.assertEqual(classify_label_grid(request(target=target), source), ("aligned_window", (1, 1, 2, 2)))
        rotated = grid(affine=(0, 80, 60, 300, 60, -80))
        target = target_grid_from_pixel_bounds(rotated, (1, 1, 3, 3))
        self.assertEqual(classify_label_grid(request(target=target), rotated), ("aligned_window", (1, 1, 2, 2)))
        shifted = grid(2, 2, (50, 100, 0, 250, 0, -100))
        self.assertEqual(classify_label_grid(request(target=shifted), source), ("nearest_warp", None))
        finer = grid(8, 6, (0, 50, 0, 300, 0, -50))
        self.assertEqual(classify_label_grid(request(target=finer), source), ("nearest_warp", None))

    def test_partial_coverage_and_incompatible_crs_fail(self):
        partial = grid(affine=(-1, 100, 0, 300, 0, -100))
        with self.assertRaises(LabelMismatchError) as caught:
            classify_label_grid(request(target=partial), grid())
        self.assertEqual(caught.exception.diagnostics[0].code, "incomplete_label_coverage")
        from osgeo import osr
        earth = osr.SpatialReference()
        earth.ImportFromEPSG(4326)
        with self.assertRaises(LabelMismatchError) as caught:
            classify_label_grid(request(), replace(grid(), crs_wkt=earth.ExportToWkt()))
        self.assertEqual(caught.exception.diagnostics[0].code, "incompatible_label_crs")

    def test_different_crs_uses_nearest_warp(self):
        from osgeo import osr
        source = grid()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(source.crs_wkt)
        srs.SetProjParm("false_easting", 250100)
        shifted = grid(affine=(100, 100, 0, 300, 0, -100))
        shifted = replace(shifted, crs_wkt=srs.ExportToWkt())
        # An interior target avoids numerical uncertainty at coverage boundaries.
        target = grid(2, 1, (100, 100, 0, 200, 0, -100))
        self.assertEqual(classify_label_grid(request(target=target), shifted), ("nearest_warp", None))


@unittest.skipUnless(HAS_NUMPY and HAS_GDAL, "NumPy/GDAL are unavailable")
class SourcePlanningIntegrationTestCase(unittest.TestCase):
    def test_larger_semantic_and_instance_arrays_validate_in_source_shape(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = grid()
            target = target_grid_from_pixel_bounds(source, (1, 1, 3, 3))
            semantic, instance = root / "whole.npy", root / "whole.npz"
            np.save(semantic, np.arange(12, dtype=np.int16).reshape(3, 4))
            np.savez(instance, mask=np.ones((3, 4), dtype=np.uint16),
                     bboxes=np.array([[0, 0, 4, 3]]), num_craters=np.array(1))
            before = snapshot(root)
            for path in (semantic, instance):
                plan = plan_label_preparation(request(path, source, target), path)
                self.assertEqual(plan.method, "aligned_window")
                self.assertEqual(plan.source_window, (1, 1, 2, 2))
                self.assertEqual(plan.output_suffix, path.suffix)
            self.assertEqual(snapshot(root), before)
            np.savez(instance, mask=np.zeros((3, 4), dtype=np.uint8), num_craters=np.array(1))
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(instance, source, target), instance)
            self.assertEqual(caught.exception.diagnostics[0].code, "malformed_instance_label")

    def test_source_grid_sidecar_and_malformed_array(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full.npy"
            np.save(path, np.zeros((3, 4), dtype=np.uint8))
            path.with_suffix(".json").write_text(json.dumps({"source_grid": grid().to_dict(), "sample_id": "unrelated"}))
            target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
            plan = plan_label_preparation(request(path, target=target), path)
            self.assertEqual(plan.method, "aligned_window")
            np.save(path, np.zeros((3, 4), dtype=np.float32))
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path, target=target), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "invalid_label_dtype")

    def _tiff(self, path, *, bands=1, nodata_at=None, dtype=None):
        import numpy as np
        from osgeo import gdal
        source = grid()
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), 4, 3, bands, dtype or gdal.GDT_Int16)
        dataset.SetProjection(source.crs_wkt)
        dataset.SetGeoTransform(source.transform)
        for index in range(1, bands + 1):
            array = np.zeros((3, 4), dtype=np.int16)
            if nodata_at is not None:
                array[nodata_at] = -9999
            dataset.GetRasterBand(index).WriteArray(array)
            dataset.GetRasterBand(index).SetNoDataValue(-9999)
        dataset = None

    def test_geotiff_integer_band_nodata_and_no_output(self):
        from osgeo import gdal
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "full.tif"
            self._tiff(path, nodata_at=(0, 0))
            target = target_grid_from_pixel_bounds(grid(), (1, 1, 3, 3))
            before = snapshot(root)
            plan = plan_label_preparation(request(path, target=target), path)
            self.assertEqual(plan.method, "aligned_window")
            self.assertEqual(plan.output_suffix, ".npy")
            self.assertEqual(snapshot(root), before)
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "label_nodata_in_target")
            # A coarse target would not sample source pixel (0, 0), but unknown
            # coverage inside its footprint must still be rejected.
            coarse = grid(2, 1, (0, 200, 0, 300, 0, -300))
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path, target=coarse), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "label_nodata_in_target")
            for options, code in (({"bands": 2}, "malformed_label"), ({"dtype": gdal.GDT_Float32}, "invalid_label_dtype")):
                other = root / f"{code}.tif"
                self._tiff(other, **options)
                with self.assertRaises(LabelMismatchError) as caught:
                    plan_label_preparation(request(other), other)
                self.assertEqual(caught.exception.diagnostics[0].code, code)

    def _gpkg(self, path, records, *, wkt=None):
        from osgeo import ogr, osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt or grid().crs_wkt)
        dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
        layer = dataset.CreateLayer("craters", srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("crater_id", ogr.OFTInteger64))
        for instance, polygon in records:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("crater_id", instance)
            feature.SetGeometry(ogr.CreateGeometryFromWkt(polygon))
            layer.CreateFeature(feature)
            feature = None
        layer = None
        dataset = None

    def test_gpkg_arbitrary_ids_overlaps_and_empty_layer_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = json.loads(FIXTURES.read_text())
            for case in fixtures["vector_cases"]:
                records = []
                for item in case["rectangles"]:
                    x0, y0, x1, y1 = item["bounds"]
                    records.append((item["id"], f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"))
                path = root / f"{case['name']}.gpkg"
                self._gpkg(path, records)
                before = snapshot(root)
                result = preflight_label(request(path), label_source=root, assigned_split="train")
                self.assertEqual(result.status, "passed")
                self.assertEqual(result.label_plan.method, "vector_rasterize")
                self.assertEqual(result.label_plan.output_suffix, ".npz")
                self.assertEqual(snapshot(root), before)

    def test_gpkg_duplicate_ids_invalid_geometry_and_missing_layer_fail(self):
        polygon = "POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0))"
        bowtie = "POLYGON ((0 0, 100 100, 100 0, 0 100, 0 0))"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = (([(12, polygon), (12, polygon)], "invalid_instance_ids"),
                     ([(0, polygon)], "invalid_instance_ids"),
                     ([(12, bowtie)], "invalid_label_geometry"))
            for index, (records, code) in enumerate(cases):
                path = root / f"invalid_{index}.gpkg"
                self._gpkg(path, records)
                with self.assertRaises(LabelMismatchError) as caught:
                    plan_label_preparation(request(path), path)
                self.assertEqual(caught.exception.diagnostics[0].code, code)
            path = root / "empty.gpkg"
            self._gpkg(path, [])
            with self.assertRaises(LabelMismatchError) as caught:
                plan_label_preparation(request(path, layer="absent"), path)
            self.assertEqual(caught.exception.diagnostics[0].code, "missing_label_layer")


if __name__ == "__main__":
    unittest.main()
