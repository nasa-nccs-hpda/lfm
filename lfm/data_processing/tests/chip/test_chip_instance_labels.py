"""A4 synthetic label exports and archive conversion, independent of tiling."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfm.data_processing.chip.chip_instance_labels import (
    _edge, _rectangle, _write_archive, convert_crater_labels, materialize_instance_label,
)
from lfm.data_processing.chip.chip_label_planning import _hash_file, plan_label_preparation
from lfm.data_processing.chip.chip_requests import pixel_to_projected, target_grid_from_pixel_bounds
from lfm.data_processing.chip.chip_types import LabelMismatchError, LabelPreparationPlan
from lfm.data_processing.tests.chip.test_chip_label_planning import HAS_GDAL, HAS_NUMPY, FIXTURES, grid, request, snapshot


MODULE = "lfm.data_processing.chip.chip_instance_labels"


class InstanceSafetyTestCase(unittest.TestCase):
    def test_linear_edge_needs_no_densification(self):
        self.assertEqual(_edge((0, 0), (1, 1), lambda x, y: (x + 2, y - 2), False), [(2, -2)])

    def test_curved_edge_densifies_in_target_pixels(self):
        result = _edge((0, 0), (1, 0), lambda x, y: (x, x * x * 100), True)
        self.assertGreater(len(result), 20)
        self.assertEqual(result[0], (0, 0))
        for a, b in zip(result, result[1:] + [(1, 100)]):
            self.assertLessEqual(abs(((a[0] + b[0]) / 2) ** 2 * 100 - (a[1] + b[1]) / 2), 1e-4)

    def test_nonconvergent_edge_fails(self):
        with self.assertRaisesRegex(ValueError, "converge"):
            _edge((0, 0), (1, 0), lambda x, y: (x, 0 if x == 0 else 1), True)

    def test_stale_source_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            path.write_bytes(b"source")
            req = request(path, grid(), relation="exact")
            plan = LabelPreparationPlan(req.label_input, req.target_grid, "exact", "0" * 64)
            before = snapshot(Path(tmp))
            with self.assertRaises(LabelMismatchError):
                materialize_instance_label(req, plan, staging_root=Path(tmp) / "stage")
            self.assertEqual(snapshot(Path(tmp)), before)

    def test_exact_archive_can_reuse_without_numpy_or_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            path.write_bytes(b"validated by mocked planner")
            req = request(path, grid(), relation="exact")
            plan = LabelPreparationPlan(req.label_input, req.target_grid, "exact", _hash_file(path))
            with patch(MODULE + ".plan_label_preparation", return_value=plan):
                artifact = materialize_instance_label(req, plan)
            self.assertEqual(artifact.path, path)
            self.assertEqual(artifact.sha256, plan.source_sha256)

    def test_request_contract_cannot_be_replaced_by_plan(self):
        req = request(Path("labels.npz"), grid(), relation="exact")
        plan = LabelPreparationPlan(req.label_input, grid(2, 2), "exact", "0" * 64)
        with self.assertRaises(LabelMismatchError):
            materialize_instance_label(req, plan)


@unittest.skipUnless(HAS_NUMPY and HAS_GDAL, "NumPy/GDAL are unavailable")
class InstanceConversionTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import ogr, osr

        self.np, self.ogr, self.osr = np, ogr, osr
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def gpkg(self, target, features, name="labels.gpkg", source_wkt=None):
        """Coordinates are target pixel outlines; metadata mirrors crater export."""
        path = self.root / name
        dataset = self.ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
        srs = self.osr.SpatialReference()
        srs.ImportFromWkt(source_wkt or target.crs_wkt)
        layer = dataset.CreateLayer("craters", srs=srs, geom_type=self.ogr.wkbUnknown)
        for field, kind in (("crater_id", self.ogr.OFTInteger64), ("method", self.ogr.OFTString),
                            ("source", self.ogr.OFTString), ("band", self.ogr.OFTInteger),
                            ("seed_col", self.ogr.OFTReal), ("seed_row", self.ogr.OFTReal),
                            ("area_native", self.ogr.OFTReal)):
            layer.CreateField(self.ogr.FieldDefn(field, kind))

        def native(geometry):
            kind = self.ogr.GT_Flatten(geometry.GetGeometryType())
            result = self.ogr.Geometry(kind)
            if geometry.GetGeometryName() == "LINEARRING":
                result = self.ogr.Geometry(self.ogr.wkbLinearRing)
                for i in range(geometry.GetPointCount()):
                    result.AddPoint_2D(*pixel_to_projected(target.transform, *geometry.GetPoint(i)[:2]))
            else:
                for child in geometry:
                    result.AddGeometry(native(child))
            return result

        for instance, outline in features:
            feature = self.ogr.Feature(layer.GetLayerDefn())
            feature.SetField("crater_id", instance)
            feature.SetField("source", "scientist-selected-scene.tif")
            feature.SetField("method", "manual")
            feature.SetField("band", 1)
            feature.SetGeometry(native(outline))
            layer.CreateFeature(feature)
            feature = None
        layer = dataset = None
        return path

    def archive(self, mask, boxes, name="labels.npz"):
        path = self.root / name
        boxes = self.np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.np.savez(path, mask=self.np.asarray(mask, dtype=self.np.int64),
                      bboxes=boxes, num_craters=self.np.asarray(len(boxes)))
        return path

    def materialize(self, path, source=None, target=None, stage="stage", relation="clip_to_target"):
        req = request(path, source, target, relation)
        plan = plan_label_preparation(req, path)
        return materialize_instance_label(req, plan, staging_root=self.root / stage)

    def test_all_a0_vector_oracles_and_feature_order(self):
        cases = json.loads(FIXTURES.read_text())["vector_cases"]
        for case in cases:
            for reverse in (False, True):
                with self.subTest(case=case["name"], reverse=reverse):
                    height, width = case["shape_hw"]
                    target = grid(width, height)
                    features = [(r["id"], _rectangle(*r["bounds"])) for r in case["rectangles"]]
                    if reverse:
                        features.reverse()
                    path = self.gpkg(target, features, f"{case['name']}-{reverse}.gpkg")
                    before = path.read_bytes()
                    result = convert_crater_labels(path, target_grid=target)
                    self.np.testing.assert_array_equal(result.mask, case["mask"])
                    self.np.testing.assert_allclose(result.bboxes, self.np.asarray(case["bboxes"]).reshape(-1, 4))
                    self.assertEqual(result.id_mapping, tuple((old, new) for new, old in enumerate(case["ids"], 1)))
                    self.assertEqual(result.num_craters, len(case["ids"]))
                    self.assertEqual(sum(d.code == "subpixel_instance" for d in result.diagnostics), len(case["subpixel"]))
                    self.assertEqual(sum(d.code == "fully_occluded_instance" for d in result.diagnostics), len(case["occluded"]))
                    self.assertEqual(path.read_bytes(), before)

    def test_holes_multipart_and_boundary_centers(self):
        target = grid(5, 3)
        polygon = _rectangle(0, 0, 3, 3)
        hole = _rectangle(.5, .5, 2.5, 2.5)
        polygon.AddGeometry(hole.GetGeometryRef(0))
        multi = self.ogr.Geometry(self.ogr.wkbMultiPolygon)
        multi.AddGeometry(polygon)
        multi.AddGeometry(_rectangle(4, 0, 5, 1))
        result = convert_crater_labels(self.gpkg(target, [(17, multi)]), target_grid=target)
        self.np.testing.assert_array_equal(result.mask, [[1, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0]])
        self.np.testing.assert_array_equal(result.bboxes, [[0, 0, 5, 3]])

    def test_touch_only_multipart_piece_does_not_expand_box(self):
        target = grid()
        multi = self.ogr.Geometry(self.ogr.wkbMultiPolygon)
        multi.AddGeometry(_rectangle(0, 0, 1, 1))
        multi.AddGeometry(_rectangle(4, 2, 5, 3))
        result = convert_crater_labels(self.gpkg(target, [(99, multi)]), target_grid=target)
        self.np.testing.assert_array_equal(result.bboxes, [[0, 0, 1, 1]])

    def test_rotated_grid_preserves_pixel_outline(self):
        target = grid(3, 2, (1000, 80, 60, 2000, 60, -80))
        path = self.gpkg(target, [(92, _rectangle(.5, .5, 2.5, 1.5))])
        result = convert_crater_labels(path, target_grid=target)
        self.np.testing.assert_array_equal(result.mask, self.np.ones((2, 3)))
        self.np.testing.assert_allclose(result.bboxes, [[.5, .5, 2, 1]])

    def test_different_crs_vector(self):
        source = grid(3, 2, (250100, 100, 0, 1000, 0, -100))
        srs = self.osr.SpatialReference()
        srs.ImportFromWkt(source.crs_wkt)
        srs.SetProjParm("false_easting", 250100)
        source = replace(source, crs_wkt=srs.ExportToWkt())
        target = grid(3, 2, (250000, 100, 0, 1000, 0, -100))
        path = self.gpkg(source, [(9, _rectangle(.2, .2, 2.8, 1.8))])
        result = convert_crater_labels(path, target_grid=target)
        self.np.testing.assert_array_equal(result.mask, self.np.ones((2, 3)))
        self.np.testing.assert_allclose(result.bboxes, [[.2, .2, 2.6, 1.6]], atol=1e-4)

    def test_vector_from_larger_coarser_scene_clips_to_target(self):
        source = grid(4, 3, (0, 200, 0, 600, 0, -200))
        target = grid(3, 2, (200, 100, 0, 400, 0, -100))
        path = self.gpkg(source, [(51, _rectangle(0, 0, 2.25, 3))])
        result = convert_crater_labels(path, target_grid=target)
        self.np.testing.assert_array_equal(result.mask, self.np.ones((2, 3)))
        self.np.testing.assert_allclose(result.bboxes, [[0, 0, 2.5, 2]])

    def test_aligned_archive_clips_boxes_and_compacts_gaps(self):
        source = grid(4, 2, (0, 100, 0, 200, 0, -100))
        target = target_grid_from_pixel_bounds(source, (1, 0, 4, 2))
        path = self.archive([[1, 2, 3, 3], [0, 0, 3, 3]], [[0, 0, 1, 1], [1, 0, 1, 1], [2, 0, 2, 2]])
        before = path.read_bytes()
        artifact = self.materialize(path, source, target)
        self.assertEqual(artifact.instance_id_map, ((2, 1), (3, 2)))
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[1, 2, 2], [0, 2, 2]])
            self.np.testing.assert_array_equal(output["bboxes"], [[0, 0, 1, 1], [1, 0, 2, 2]])
        self.assertEqual(path.read_bytes(), before)

    def test_nearest_archive_omits_lost_support_with_warning(self):
        source = grid(4, 2, (0, 100, 0, 200, 0, -100))
        target = grid(2, 1, (0, 200, 0, 200, 0, -200))
        path = self.archive([[1, 0, 2, 2], [0, 0, 2, 2]], [[0, 0, 1, 1], [2, 0, 2, 2]])
        artifact = self.materialize(path, source, target)
        self.assertEqual(artifact.instance_id_map, ((2, 1),))
        self.assertIn("unsupported_instance", [d.code for d in artifact.diagnostics])
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[0, 1]])
            self.np.testing.assert_array_equal(output["bboxes"], [[1, 0, 1, 1]])

    def test_rotated_archive_partial_box(self):
        source = grid(4, 3, (1000, 80, 60, 2000, 60, -80))
        target = target_grid_from_pixel_bounds(source, (1, 1, 3, 3))
        path = self.archive([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 2.25, 2]])
        artifact = self.materialize(path, source, target)
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[1, 0], [0, 0]])
            self.np.testing.assert_allclose(output["bboxes"], [[0, 0, 1.25, 1]])

    def test_outside_fractional_box_drops_its_nearest_samples(self):
        source = grid(2, 1, (0, 100, 0, 100, 0, -100))
        target = grid(1, 1, (0, 25, 0, 100, 0, -100))
        path = self.archive([[1, 0]], [[.4, 0, .2, 1]])
        artifact = self.materialize(path, source, target)
        self.assertEqual(artifact.instance_id_map, ())
        self.assertIn("excluded_instance_pixels", [d.code for d in artifact.diagnostics])
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[0]])
            self.assertEqual(output["bboxes"].shape, (0, 4))

    def test_different_crs_archive_nearest_and_boxes(self):
        source = grid(2, 2, (250100, 100, 0, 1000, 0, -100))
        srs = self.osr.SpatialReference()
        srs.ImportFromWkt(source.crs_wkt)
        srs.SetProjParm("false_easting", 250100)
        source = replace(source, crs_wkt=srs.ExportToWkt())
        target = grid(4, 4, (250000, 50, 0, 1000, 0, -50))
        path = self.archive([[1, 2], [1, 2]], [[0, 0, 1, 2], [1, 0, 1, 2]])
        artifact = self.materialize(path, source, target)
        self.assertEqual(artifact.plan.method, "nearest_warp")
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[1, 1, 2, 2]] * 4)
            self.np.testing.assert_allclose(output["bboxes"], [[0, 0, 2, 4], [2, 0, 2, 4]], atol=1e-4)

    def test_raster_occlusion_survives_crop_and_boxes_do_not_shrink(self):
        source = grid(3, 2, (0, 100, 0, 200, 0, -100))
        target = target_grid_from_pixel_bounds(source, (1, 0, 3, 2))
        path = self.archive([[2, 2, 2], [2, 2, 2]], [[0, 0, 3, 2], [0, 0, 3, 2]])
        artifact = self.materialize(path, source, target)
        self.assertEqual(artifact.instance_id_map, ((1, 1), (2, 2)))
        with self.np.load(artifact.path) as output:
            self.np.testing.assert_array_equal(output["mask"], [[2, 2], [2, 2]])
            self.np.testing.assert_array_equal(output["bboxes"], [[0, 0, 2, 2], [0, 0, 2, 2]])

    def test_empty_archive_and_vector_have_canonical_shapes(self):
        source = grid()
        target = target_grid_from_pixel_bounds(source, (0, 0, 2, 2))
        paths = [self.archive(self.np.zeros((3, 4)), []), self.gpkg(target, [])]
        for i, path in enumerate(paths):
            artifact = self.materialize(path, source if i == 0 else None, target, stage=str(i))
            self.assertTrue(any(d.code == "no_crater_labels" and d.severity == "warning"
                                for d in artifact.diagnostics))
            with self.np.load(artifact.path) as output:
                self.assertEqual(output["bboxes"].shape, (0, 4))
                self.assertEqual(output["num_craters"].shape, ())
                self.assertEqual(int(output["num_craters"]), 0)
                self.assertFalse(output["mask"].any())

    def test_exact_archive_preserves_extra_arrays_and_bytes(self):
        path = self.archive([[1]], [[0, 0, 1, 1]])
        self.np.savez_compressed(path, mask=self.np.ones((1, 1), dtype=self.np.uint16),
                                 bboxes=self.np.asarray([[0, 0, 1, 1]]), num_craters=self.np.asarray(1),
                                 annotation_notes=self.np.asarray(["preserve extra content"]))
        before = path.read_bytes()
        artifact = self.materialize(path, None, grid(1, 1), relation="exact")
        self.assertEqual(artifact.path, path)
        self.assertEqual(artifact.path.read_bytes(), before)
        self.assertFalse((self.root / "stage").exists())

    def test_deterministic_archives_across_staging_roots(self):
        target = grid()
        path = self.gpkg(target, [(64, _rectangle(0, 0, 2, 2))])
        first = self.materialize(path, target=target, stage="first")
        second = self.materialize(path, target=target, stage="second")
        self.assertEqual(first.path.read_bytes(), second.path.read_bytes())
        self.assertEqual(first.sha256, _hash_file(first.path))
        self.assertEqual(first.sha256, second.sha256)

    def test_invalid_geometry_ids_and_layer_write_nothing(self):
        target = grid()
        bowtie = self.ogr.CreateGeometryFromWkt("POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))")
        for i, features in enumerate(([(0, _rectangle(0, 0, 1, 1))],
                                      [(1, _rectangle(0, 0, 1, 1)), (1, _rectangle(2, 0, 3, 1))],
                                      [(2, bowtie)])):
            path = self.gpkg(target, features, f"bad-{i}.gpkg")
            with self.assertRaises(LabelMismatchError):
                self.materialize(path, target=target)
        path = self.gpkg(target, [], "missing.gpkg")
        with self.assertRaises(LabelMismatchError):
            convert_crater_labels(path, target_grid=target, layer="missing")
        self.assertFalse((self.root / "stage").exists())

    def test_malformed_archive_rejected(self):
        path = self.archive([[3]], [[0, 0, 1, 1]])
        with self.assertRaises(LabelMismatchError):
            self.materialize(path, grid(1, 1), grid(1, 1))
        self.assertFalse((self.root / "stage").exists())

    def test_existing_artifact_and_symlink_are_not_overwritten(self):
        target = grid()
        path = self.gpkg(target, [])
        artifact = self.materialize(path, target=target)
        before = snapshot(self.root)
        with self.assertRaises(LabelMismatchError):
            self.materialize(path, target=target)
        self.assertEqual(snapshot(self.root), before)
        stage = self.root / "unsafe"
        stage.mkdir()
        (stage / "M1_r0_c0").symlink_to(artifact.path.parent.parent, target_is_directory=True)
        with self.assertRaises(LabelMismatchError):
            self.materialize(path, target=target, stage="unsafe")

    def test_corrupt_reopen_and_source_change_remove_temporary(self):
        target = grid()
        path = self.gpkg(target, [(1, _rectangle(0, 0, 2, 2))])
        req = request(path, target=target)
        plan = plan_label_preparation(req, path)

        def corrupt(destination, result):
            _write_archive(destination, replace(result, mask=self.np.zeros_like(result.mask),
                                                bboxes=self.np.empty((0, 4)), num_craters=0))

        with patch(MODULE + "._write_archive", side_effect=corrupt):
            with self.assertRaises(LabelMismatchError):
                materialize_instance_label(req, plan, staging_root=self.root / "stage")
        self.assertFalse(any((self.root / "stage").rglob("*.npz")))
        with patch(MODULE + "._verify_source", side_effect=[None, None, ValueError("source changed")]):
            with self.assertRaises(LabelMismatchError):
                materialize_instance_label(req, plan, staging_root=self.root / "stage")
        self.assertFalse(any((self.root / "stage").rglob("*.npz")))

    def test_install_race_preserves_winners_archive(self):
        target = grid()
        path = self.gpkg(target, [])

        def other_worker_wins(source, destination):
            destination.write_bytes(b"another worker's archive")
            raise FileExistsError(str(destination))

        with patch(MODULE + ".os.link", side_effect=other_worker_wins):
            with self.assertRaises(LabelMismatchError):
                self.materialize(path, target=target)
        files = list((self.root / "stage").rglob("*.npz"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), b"another worker's archive")

    def test_stale_sidecar_and_source_owned_staging_are_rejected(self):
        source = grid()
        target = target_grid_from_pixel_bounds(source, (1, 0, 3, 2))
        path = self.archive(self.np.zeros((3, 4)), [])
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps({"source_grid": source.to_dict()}))
        req = request(path, target=target)
        plan = plan_label_preparation(req, path)
        sidecar.write_text(json.dumps({"source_grid": grid(2, 2).to_dict()}))
        with self.assertRaises(LabelMismatchError):
            materialize_instance_label(req, plan, staging_root=self.root / "stage")
        self.assertFalse((self.root / "stage").exists())
        sidecar.unlink()
        owned = self.root / "stage" / req.sample_id
        owned.mkdir(parents=True)
        moved = owned / "source.npz"
        path.rename(moved)
        with self.assertRaises(LabelMismatchError):
            self.materialize(moved, source, target)
        self.assertTrue(moved.exists())


if __name__ == "__main__":
    unittest.main()
