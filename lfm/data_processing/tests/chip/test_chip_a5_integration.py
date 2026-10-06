"""Worker label preparation through real reprojection, staging and publication."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfm.data_processing.chip.chip_acquisition import AcquisitionGroupResult, ChipAcquisitionResult
from lfm.data_processing.chip.chip_creation import create_chips, _clear_sample_intermediates, _run_prepared_request
from lfm.data_processing.chip.chip_preflight import preflight_chip_requests
from lfm.data_processing.chip.chip_requests import raster_bounds
from lfm.data_processing.chip.chip_types import ChipRequest, GeographicAOI, LabelInput, TargetGrid
from lfm.data_processing.tiling.tiling_results import TileCubeRecord
from lfm.data_processing.tests.chip import test_chip_creation as creation_fixtures


HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "osgeo"))


def synthetic_acquire(prepared, config):
    """Only replace expensive tiling; all downstream raster stages are real."""
    import numpy as np
    from osgeo import gdal

    assert prepared.prepared_label is not None
    assert prepared.prepared_label.path.is_file()
    req = prepared.request
    if req.sample_id.startswith("empty"):
        return ChipAcquisitionResult(prepared, "complete")
    if req.sample_id.startswith("changed"):
        with req.label_path.open("ab") as stream:
            stream.write(b"source changed during imagery acquisition")
    target = req.target_grid
    path = config.intermediate_root / req.sample_id / "coarse" / "cube.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), target.width, target.height, 1, gdal.GDT_Float32)
    dataset.SetProjection(target.crs_wkt)
    dataset.SetGeoTransform(target.transform)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-32768.)
    band.SetDescription("elevation")
    band.SetMetadataItem("Name", "elevation")
    pixels = np.ones((target.height, target.width), dtype=np.float32)
    if req.sample_id.startswith("partial"):
        pixels[0, 0] = -32768.
    elif req.sample_id.startswith("invalid"):
        pixels[:] = -32768.
    band.WriteArray(pixels)
    band = dataset = None
    record = TileCubeRecord("static", "1N", 5, 0, 0, None, path,
                            ("elevation",), target.crs_wkt, (-32768.,))
    group = AcquisitionGroupResult(req.sample_id, "coarse", 5, path.parent,
                                   req.geographic_aoi, (req.geographic_aoi,), (), "complete",
                                   records=(record,), inventory_paths=(path,),
                                   attempted_query_parts=(req.geographic_aoi,))
    return ChipAcquisitionResult(prepared, "complete", (group,))


def synthetic_worker(task):
    # Spawn imports this module and applies its own patch, never inherits GDAL.
    with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=synthetic_acquire):
        return _run_prepared_request(*task)


class CleanupSafetyTestCase(unittest.TestCase):
    def test_cleanup_never_owns_source_index_or_label(self):
        helper = creation_fixtures.ChipCreationTestCase()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = helper.config(root, retention="never")
            prepared = helper.prepared(helper.request(1))
            sample = config.intermediate_root / prepared.request.sample_id
            sample.mkdir(parents=True)
            source = sample / "source.npy"
            source.write_bytes(b"protected")
            unsafe = replace(prepared, preflight=replace(prepared.preflight, resolved_label_path=source))
            self.assertEqual(_clear_sample_intermediates(unsafe, config).code, "intermediate_cleanup_failed")
            self.assertEqual(source.read_bytes(), b"protected")
            group = config.acquisition_groups[0]
            index = sample / "protected.gpkg"
            index.write_bytes(b"index")
            source_config = replace(group.tile_config.sources[0], index_path=index)
            config = replace(config, acquisition_groups=(replace(group, tile_config=replace(
                group.tile_config, sources=(source_config,))),))
            self.assertIsNotNone(_clear_sample_intermediates(prepared, config))
            self.assertTrue(source.is_file())

    def test_sibling_symlink_is_never_followed(self):
        helper = creation_fixtures.ChipCreationTestCase()
        with tempfile.TemporaryDirectory() as tmp:
            config = helper.config(Path(tmp), retention="never")
            prepared = helper.prepared(helper.request(1))
            other = config.intermediate_root / "other"
            other.mkdir(parents=True)
            keep = other / "keep"
            keep.write_bytes(b"keep")
            (config.intermediate_root / prepared.request.sample_id).symlink_to(other, target_is_directory=True)
            self.assertIsNotNone(_clear_sample_intermediates(prepared, config))
            self.assertTrue(keep.is_file())


@unittest.skipUnless(HAS_DEPS, "NumPy/GDAL required")
class LabelPipelineTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import ogr, osr

        self.np, self.ogr, self.osr = np, ogr, osr
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = creation_fixtures.ChipCreationRasterTestCase()
        self.helper.setUp()
        self.grid = self.helper.grid
        affine = (-1., 1., 0., 3., 0., -1.)
        self.source_grid = TargetGrid(self.grid.crs_wkt, affine, raster_bounds(affine, 3, 3), 3, 3)
        self.label = self.root / "scene.npy"
        np.save(self.label, np.arange(9, dtype=np.uint16).reshape(3, 3))

    def request(self, name="valid", label=None):
        return ChipRequest(name, self.grid, GeographicAOI(2, 0, 0, 2), "same-scene",
                           label_input=LabelInput(label or self.label, relation="clip_to_target",
                                                  source_grid=None if label and label.suffix == ".gpkg" else self.source_grid))

    def run_batch(self, requests, name="run", workers=1, **kwargs):
        config = self.helper.config(self.root / name)
        with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=synthetic_acquire), \
             patch("lfm.data_processing.chip.chip_creation._run_prepared_task", new=synthetic_worker):
            return create_chips(requests, config, max_workers=workers, **kwargs)

    def test_semantic_materializes_before_acquisition_and_preserves_provenance(self):
        before = self.label.read_bytes()
        events = []
        # Call the prepared worker to capture stage events without UI dependencies.
        config = self.helper.config(self.root / "run")
        prepared = preflight_chip_requests((self.request(),), config).requests[0]
        with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=synthetic_acquire):
            result = _run_prepared_request(prepared, config, False, events.append)
        self.assertEqual(result.status, "success", result.message)
        self.np.testing.assert_array_equal(self.np.load(result.label_path), [[4, 5], [7, 8]])
        self.assertEqual(self.label.read_bytes(), before)
        self.assertFalse(result.prepared_label.path.exists())  # retention=never
        starts = [event.stage for event in events if event.state == "started"]
        self.assertLess(starts.index("label/clip"), starts.index("tiling"))
        document = json.loads(result.diagnostic_path.read_text())
        self.assertEqual(document["prepared_label"]["plan"]["method"], "aligned_window")

    def test_serial_spawn_outputs_and_manifests_match(self):
        requests = (self.request("b"), self.request("a"))
        serial = self.run_batch(requests)
        first_manifest = serial.manifest_path.read_bytes()
        first_labels = [r.label_path.read_bytes() for r in serial.results]
        first_chips = [r.chip_path.read_bytes() for r in serial.results]
        parallel = self.run_batch(tuple(reversed(requests)), workers=2, overwrite=True)
        self.assertEqual([r.status for r in parallel.results], ["success", "success"])
        self.assertEqual(parallel.manifest_path.read_bytes(), first_manifest)
        self.assertEqual([r.label_path.read_bytes() for r in parallel.results], first_labels)
        self.assertEqual([r.chip_path.read_bytes() for r in parallel.results], first_chips)
        self.assertEqual([r.request.sample_id for r in parallel.results], ["a", "b"])

    def test_tiff_tasks_publish_canonical_pairs_in_serial_and_spawn(self):
        from osgeo import gdal

        path = self.root / "instance_or_class.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(path), 3, 3, 1, gdal.GDT_UInt16)
        ds.SetGeoTransform(self.source_grid.transform)
        ds.SetProjection(self.source_grid.crs_wkt)
        ds.GetRasterBand(1).WriteArray(self.np.array([[0, 0, 0], [0, 12, 90], [0, 12, 0]], dtype=self.np.uint16))
        ds = None
        before = path.read_bytes()
        requests = tuple(replace(self.request(kind), label_path=path, label_grid=None, label_input=LabelInput(
            path, kind=kind, relation="clip_to_target")) for kind in ("semantic", "raster_instance"))
        serial = self.run_batch(requests)
        self.assertEqual([r.status for r in serial.results], ["success", "success"])
        first_labels = [r.label_path.read_bytes() for r in serial.results]
        first_manifest = serial.manifest_path.read_bytes()
        for result in serial.results:
            if result.request.label_input.kind == "semantic":
                self.assertEqual(result.label_path.suffix, ".npy")
                self.np.testing.assert_array_equal(self.np.load(result.label_path), [[12, 90], [12, 0]])
            else:
                self.assertEqual(result.label_path.suffix, ".npz")
                with self.np.load(result.label_path) as archive:
                    self.np.testing.assert_array_equal(archive["mask"], [[1, 2], [1, 0]])
                    self.assertEqual(int(archive["num_craters"]), 2)
                document = json.loads(result.diagnostic_path.read_text())
                self.assertEqual(document["raster_instance_contract"]["box_derivation"],
                                 "final_grid_visible_pixel_support")
        parallel = self.run_batch(requests, workers=2, overwrite=True)
        self.assertEqual([r.status for r in parallel.results], ["success", "success"])
        self.assertEqual([r.label_path.read_bytes() for r in parallel.results], first_labels)
        self.assertEqual(parallel.manifest_path.read_bytes(), first_manifest)
        self.assertEqual(path.read_bytes(), before)

    def test_materialization_failure_starts_no_tiling_and_later_sample_succeeds(self):
        from lfm.data_processing.chip.chip_label_materialization import materialize_semantic_label
        from lfm.data_processing.chip.chip_labels import _label_error

        def materialize(req, plan, **kwargs):
            if req.sample_id == "a_bad":
                raise _label_error(req, code="injected_conversion_failure", message="conversion failed")
            return materialize_semantic_label(req, plan, **kwargs)

        calls = []
        def acquire(prepared, config):
            calls.append(prepared.request.sample_id)
            return synthetic_acquire(prepared, config)

        config = self.helper.config(self.root / "run")
        with patch("lfm.data_processing.chip.chip_creation.materialize_semantic_label", side_effect=materialize), \
             patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=acquire):
            batch = create_chips((self.request("a_bad"), self.request("z_good")), config)
        self.assertEqual([r.status for r in batch.results], ["failed", "success"])
        self.assertEqual(calls, ["z_good"])
        self.assertEqual(batch.results[0].diagnostics[-1].stage, "label_preparation")
        self.assertFalse((config.output_root / "chips/a_bad_input_static_chip.tif").exists())

    def test_changed_source_and_no_imagery_fail_without_pair(self):
        for name in ("changed", "empty"):
            with self.subTest(name=name):
                batch = self.run_batch((self.request(name),), name=name)
                result = batch.results[0]
                self.assertEqual(result.status, "failed", result.message)
                self.assertIsNone(result.chip_path)
                self.assertIsNone(result.label_path)
                self.assertFalse(any((self.root / name / "dataset/chips").glob("*")))

    def test_wholly_uncovered_acquired_raster_publishes_nodata_pair(self):
        from osgeo import gdal

        result = self.run_batch((self.request("invalid"),)).results[0]
        self.assertEqual(result.status, "success", result.message)
        self.assertTrue(result.label_path.is_file())
        self.assertEqual(result.imagery_nodata["union_invalid_percent"], 100.)
        self.assertIn("uncovered_imagery_bands", [d.code for d in result.diagnostics])
        dataset = gdal.Open(str(result.chip_path))
        band = dataset.GetRasterBand(1)
        self.assertEqual(band.GetNoDataValue(), -32768.)
        self.assertTrue(self.np.all(band.ReadAsArray() == -32768.))
        self.assertFalse(band.GetMaskBand().ReadAsArray().any())
        band = dataset = None

    def test_partial_nodata_warns_and_records_spatial_union(self):
        result = self.run_batch((self.request("partial"),)).results[0]
        self.assertEqual(result.status, "success", result.message)
        self.assertEqual(result.imagery_nodata["union_invalid_count"], 1)
        self.assertEqual(result.imagery_nodata["union_invalid_percent"], 25)
        self.assertEqual(result.imagery_nodata["bands"][0]["invalid_count"], 1)
        self.assertIn("partial_imagery_nodata", [d.code for d in result.diagnostics])

    def test_failed_overwrite_preserves_prior_pair_and_manifest_records_attempt(self):
        batch = self.run_batch((self.request(),))
        old = batch.results[0]
        old_bytes = (old.chip_path.read_bytes(), old.label_path.read_bytes())
        config = self.helper.config(self.root / "run")
        from lfm.data_processing.chip.chip_labels import _label_error
        with patch("lfm.data_processing.chip.chip_creation.materialize_semantic_label", side_effect=_label_error(
                self.request(), code="injected", message="failed conversion")):
            failed = create_chips((self.request(),), config, overwrite=True)
        result = failed.results[0]
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.chip_path)
        self.assertIsNotNone(result.preserved_pair)
        self.assertEqual((old.chip_path.read_bytes(), old.label_path.read_bytes()), old_bytes)
        self.assertIsNotNone(json.loads(failed.manifest_path.read_text())["samples"][0]["preserved_pair"])

    def test_geopackage_converts_and_publishes_npz_not_source(self):
        path = self.root / "scene.gpkg"
        ds = self.ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
        srs = self.osr.SpatialReference()
        srs.ImportFromWkt(self.grid.crs_wkt)
        layer = ds.CreateLayer("craters", srs=srs, geom_type=self.ogr.wkbPolygon)
        layer.CreateField(self.ogr.FieldDefn("crater_id", self.ogr.OFTInteger))
        feature = self.ogr.Feature(layer.GetLayerDefn())
        feature.SetField("crater_id", 87)
        feature.SetGeometry(self.ogr.CreateGeometryFromWkt("POLYGON ((-1 -1, 3 -1, 3 3, -1 3, -1 -1))"))
        layer.CreateFeature(feature)
        feature = layer = ds = None
        before = path.read_bytes()
        batch = self.run_batch((self.request(label=path),))
        result = batch.results[0]
        self.assertEqual(result.status, "success", result.message)
        self.assertEqual(result.label_path.suffix, ".npz")
        self.assertEqual(result.prepared_label.instance_id_map, ((87, 1),))
        with self.np.load(result.label_path) as archive:
            self.np.testing.assert_array_equal(archive["mask"], self.np.ones((2, 2)))
            self.np.testing.assert_array_equal(archive["bboxes"], [[0, 0, 2, 2]])
        self.assertEqual(path.read_bytes(), before)
        sample = json.loads(batch.manifest_path.read_text())["samples"][0]
        self.assertEqual(sample["source_label_path"], str(path))
        self.assertIsNotNone(sample["label_preparation_id"])

    def test_second_publication_link_failure_restores_prior_derived_pair(self):
        batch = self.run_batch((self.request(),))
        first = batch.results[0]
        original = (first.chip_path.read_bytes(), first.label_path.read_bytes())
        real_link = os.link

        def fail_label_link(source, destination):
            if Path(destination) == first.label_path:
                raise OSError("injected second publication failure")
            return real_link(source, destination)

        with patch("lfm.data_processing.chip.chip_publication.os.link", side_effect=fail_label_link):
            failed = self.run_batch((self.request(),), overwrite=True).results[0]
        self.assertEqual(failed.status, "failed")
        self.assertEqual((first.chip_path.read_bytes(), first.label_path.read_bytes()), original)
        self.assertIsNotNone(failed.preserved_pair)
        self.assertFalse(list(first.label_path.parent.glob(".*.publishing")))

    def test_bad_preflight_on_overwrite_keeps_old_pair_and_continues(self):
        first = self.run_batch((self.request("a_bad"),)).results[0]
        original = (first.chip_path.read_bytes(), first.label_path.read_bytes())
        self.np.save(self.label, self.np.zeros((1, 1), dtype=self.np.uint8))
        good = self.root / "good.npy"
        self.np.save(good, self.np.zeros((3, 3), dtype=self.np.uint8))
        batch = self.run_batch((self.request("a_bad"), self.request("z_good", label=good)), overwrite=True)
        self.assertEqual([r.status for r in batch.results], ["failed", "success"])
        self.assertIsNotNone(batch.results[0].preserved_pair)
        self.assertEqual((first.chip_path.read_bytes(), first.label_path.read_bytes()), original)

    def test_failure_retention_keeps_derived_label_and_source_unchanged(self):
        req = self.request("empty")
        before = self.label.read_bytes()
        config = replace(self.helper.config(self.root / "run"), intermediate_retention="on_failure")
        with patch("lfm.data_processing.chip.chip_creation.acquire_prepared_request", side_effect=synthetic_acquire):
            result = create_chips((req,), config).results[0]
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.prepared_label.path.is_file())
        self.assertEqual(self.label.read_bytes(), before)

    def test_nodata_union_accounts_for_empty_dynamic_and_wholly_empty_chips(self):
        from types import SimpleNamespace
        from lfm.data_processing.chip.chip_assembly import summarize_imagery_nodata

        reprojection = SimpleNamespace(acquisition=SimpleNamespace(prepared_request=SimpleNamespace(request=self.request())))
        mask = self.np.asarray([[[True, False], [True, True]], [[False, False], [True, True]]])
        assembled = SimpleNamespace(valid_mask=mask, pixels=self.np.ones((2, 2, 2)), target_grid=self.grid,
                                    band_names=("dynamic", "static"), required_bands=(True, False),
                                    band_origins=(("dynamic", "wac"), ("static", "static")),
                                    reprojection=reprojection)
        summary = summarize_imagery_nodata(assembled)
        self.assertEqual([b["invalid_count"] for b in summary["bands"]], [1, 2])
        self.assertEqual(summary["union_invalid_count"], 2)
        mask[0] = False
        summary = summarize_imagery_nodata(assembled)
        self.assertEqual(summary["bands"][0]["invalid_percent"], 100.)
        mask[:] = False
        summary = summarize_imagery_nodata(assembled)
        self.assertEqual(summary["union_invalid_percent"], 100.)


if __name__ == "__main__":
    unittest.main()
