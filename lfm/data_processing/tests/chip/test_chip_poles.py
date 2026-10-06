"""Full-longitude polar caps produce unmasked native-grid rectangles."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfm.data_processing.chip import (
    GeographicAOI, LabelInput, chip_grid_family, chip_request_from_aoi, default_chip_zoom,
    static_grid_reference, create_chips,
)
from lfm.data_processing.chip.chip_requests import (
    geographic_query_parts, geographic_aoi_from_target_grid, validate_request_geographic_aoi,
    target_grid_from_bounds, target_grid_from_pixel_bounds, _projected_to_pixel,
    _spatial_reference, _create_transformation, _transform_point, UnsupportedCoverageError,
)
from lfm.data_processing.tiling.grid_registry import GridFamily, default_grid_registry
from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt


def cap(north=True, boundary=89.99):
    return GeographicAOI(90, -180, boundary, 180) if north else GeographicAOI(-boundary, -180, -90, 180)


class PoleContractTestCase(unittest.TestCase):
    def test_caps_are_one_query_not_zero_width_or_two_products(self):
        for north, family in ((True, GridFamily.LPS_N), (False, GridFamily.LPS_S)):
            aoi = cap(north)
            self.assertEqual(geographic_query_parts(aoi), (aoi,))
            self.assertEqual(chip_grid_family(aoi), family)
            self.assertEqual(default_chip_zoom(aoi, 'wac'), 4)
            self.assertEqual(default_chip_zoom(aoi, 'nac'), 10)
            self.assertEqual(static_grid_reference(aoi).crs_wkt,
                             default_grid_registry()[family.value.upper()].crs_wkt)

    def test_partial_longitude_poles_and_nonpolar_full_longitudes_stay_rejected(self):
        for aoi in (GeographicAOI(90, 10, 89, 11), GeographicAOI(-89, 10, -90, 11),
                    GeographicAOI(90, -180, 81, 180), GeographicAOI(89, -180, 88, 180),
                    GeographicAOI(90, 0, 89, 360), GeographicAOI(90, -180, -90, 180)):
            with self.subTest(aoi=aoi), self.assertRaises(ValueError):
                geographic_query_parts(aoi)

    def test_acquisition_passes_full_cap_and_family_zoom_once(self):
        from lfm.data_processing.tests.chip import test_chip_acquisition as fixtures
        from lfm.data_processing.chip.chip_acquisition import acquire_prepared_request
        helper = fixtures.ChipAcquisitionTestCase()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = replace(helper.group('polar', root, 5, helper.source('wac', selection_mode='product_id')),
                            polar_zoom_level=4)
            config = helper.config(root, (group,))
            for north, zone in ((True, 'LPS_N'), (False, 'LPS_S')):
                record = helper.record(root / 'cube.tif', zone=zone, zoom=4)
                with patch('lfm.data_processing.chip.chip_acquisition.create_tiles_for_aoi',
                           return_value=[record, record]) as tiler:
                    result = acquire_prepared_request(helper.prepared(helper.request(aoi=cap(north))), config)
                self.assertEqual(result.status, 'complete')
                self.assertEqual(result.records, (record,))
                tiler.assert_called_once()
                self.assertEqual(tiler.call_args.kwargs['ul_lon'], -180)
                self.assertEqual(tiler.call_args.kwargs['lr_lon'], 180)
                self.assertEqual(tiler.call_args.args[0].zoom_level, 4)
                self.assertEqual(tiler.call_args.kwargs['selectors'], {'wac': 'M100'})


HAS_GDAL = importlib.util.find_spec('osgeo') is not None
HAS_ARRAYS = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_GDAL, 'GDAL required')
class PoleGridTestCase(unittest.TestCase):
    def test_native_rectangles_include_cap_and_corners_in_query(self):
        from lfm.data_processing.chip.chip_types import TargetGrid
        from lfm.data_processing.chip.chip_requests import raster_bounds
        for north in (True, False):
            for rotated in (False, True):
                requested = cap(north)
                lattice = static_grid_reference(requested)
                if rotated:
                    affine = (500003, 80, 60, 500007, 60, -80)
                    lattice = TargetGrid(lattice.crs_wkt, affine, raster_bounds(affine, 1, 1), 1, 1)
                req = chip_request_from_aoi(sample_id='cap', split_group_key='cap',
                                            geographic_aoi=requested, source_grid=lattice)
                grid, envelope = req.target_grid, req.geographic_aoi
                self.assertEqual(grid.crs_wkt, lattice.crs_wkt)
                self.assertEqual(tuple(grid.transform[i] for i in (1, 2, 4, 5)),
                                 tuple(lattice.transform[i] for i in (1, 2, 4, 5)))
                self.assertEqual(req.requested_aoi, requested)
                self.assertGreater(grid.width, 1)
                self.assertLess(grid.width, 20)
                if north:
                    self.assertLess(envelope.lower_right_latitude, requested.lower_right_latitude)
                else:
                    self.assertGreater(envelope.upper_left_latitude, requested.upper_left_latitude)
                forward = _create_transformation(_spatial_reference(load_lunar_geographic_wkt()),
                                                  _spatial_reference(grid.crs_wkt))
                boundary = 89.99 if north else -89.99
                for i in range(1440):
                    x, y = _transform_point(forward, -180 + i / 4, boundary)
                    col, row = _projected_to_pixel(grid, x, y)
                    self.assertTrue(-1e-8 <= col <= grid.width + 1e-8)
                    self.assertTrue(-1e-8 <= row <= grid.height + 1e-8)
                validate_request_geographic_aoi(req)
                with self.assertRaises(ValueError):
                    validate_request_geographic_aoi(replace(req, geographic_aoi=requested))

    def test_pole_on_edge_or_corner_and_rejected_large_cap(self):
        for zone in ('LPS_N', 'LPS_S'):
            crs = default_grid_registry()[zone].crs_wkt
            for bounds in ((499000, 499000, 501000, 501000),
                           (500000, 499000, 502000, 501000),
                           (500000, 500000, 502000, 502000)):
                grid = target_grid_from_bounds(crs_wkt=crs, bounds=bounds, width=20, height=20)
                self.assertEqual(len(geographic_query_parts(geographic_aoi_from_target_grid(grid))), 1)
            with self.assertRaisesRegex(UnsupportedCoverageError, 'reduce the AOI'):
                chip_request_from_aoi(sample_id='large', split_group_key='large',
                                      geographic_aoi=cap(zone == 'LPS_N', 82), static_only=True)
        with self.assertRaisesRegex(UnsupportedCoverageError, 'stereographic'):
            chip_request_from_aoi(sample_id='geographic', split_group_key='cap', geographic_aoi=cap(),
                source_grid=target_grid_from_bounds(crs_wkt=load_lunar_geographic_wkt(),
                                                   bounds=(-180, 89, 180, 90), width=360, height=10))


@unittest.skipUnless(HAS_GDAL and HAS_ARRAYS, 'GDAL/NumPy required')
class PolePipelineTestCase(unittest.TestCase):
    def test_semantic_instance_tiff_and_gpkg_keep_corners_serial_and_spawn(self):
        import numpy as np
        from osgeo import gdal
        from lfm.data_processing.tests.chip.test_chip_polar import polar_acquire, polar_worker
        from lfm.data_processing.tests.chip.test_chip_creation import ChipCreationRasterTestCase
        from lfm.data_processing.tests.chip.test_chip_instance_labels import InstanceConversionTestCase
        from lfm.data_processing.chip.chip_instance_labels import _rectangle
        helper = InstanceConversionTestCase()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests, originals = [], {}
            for north, name in ((True, 'north'), (False, 'south')):
                req = chip_request_from_aoi(sample_id=name, split_group_key=name,
                                            geographic_aoi=cap(north), static_only=True)
                grid = req.target_grid
                # Larger full-scene raster, all positive including outside cap.
                source_grid = target_grid_from_pixel_bounds(grid, (-2, -2, grid.width + 2, grid.height + 2))
                path = root / f'{name}.tif'
                ds = gdal.GetDriverByName('GTiff').Create(str(path), source_grid.width, source_grid.height, 1, gdal.GDT_UInt16)
                ds.SetProjection(source_grid.crs_wkt)
                ds.SetGeoTransform(source_grid.transform)
                ds.GetRasterBand(1).Fill(38)
                ds = None
                for kind in ('semantic', 'raster_instance'):
                    requests.append(replace(req, sample_id=name + '_' + kind,
                                            label_input=LabelInput(path, kind=kind, relation='clip_to_target')))
                # One outline enclosing pole and corners must remain ONE object.
                gpkg = helper.gpkg(grid, [(90, _rectangle(0, 0, grid.width, grid.height))], name=f'{name}.gpkg')
                requests.append(replace(req, label_input=LabelInput(gpkg, relation='clip_to_target', layer='craters')))
                for p in (path, gpkg):
                    originals[p] = p.read_bytes()
            config = ChipCreationRasterTestCase().config(root)
            config = replace(config, acquisition_groups=tuple(replace(g, polar_zoom_level=4)
                                                               for g in config.acquisition_groups))
            outputs = []
            for workers in (1, 2):
                with patch('lfm.data_processing.chip.chip_creation.acquire_prepared_request', side_effect=polar_acquire), \
                     patch('lfm.data_processing.chip.chip_creation._run_prepared_task', new=polar_worker):
                    batch = create_chips(requests, config, max_workers=workers, overwrite=workers == 2)
                self.assertEqual([r.status for r in batch.results], ['success'] * 6, [r.message for r in batch.results])
                for result in batch.results:
                    ds = gdal.Open(str(result.chip_path))
                    self.assertTrue(np.all(ds.ReadAsArray() == 1))  # Corners were NOT masked.
                    ds = None
                    if result.label_path.suffix == '.npy':
                        self.assertTrue(np.all(np.load(result.label_path) == 38))
                    else:
                        with np.load(result.label_path) as archive:
                            self.assertTrue(np.all(archive['mask'] == 1))
                            self.assertEqual(int(archive['num_craters']), 1)
                            grid = result.request.target_grid
                            np.testing.assert_allclose(archive['bboxes'], [[0, 0, grid.width, grid.height]])
                outputs.append([(r.chip_path.read_bytes(), r.label_path.read_bytes()) for r in batch.results])
            self.assertEqual(*outputs)
            self.assertTrue(all(p.read_bytes() == content for p, content in originals.items()))

    def test_real_tiling_and_publication_covers_pole_and_corners(self):
        import numpy as np
        from osgeo import gdal
        from lfm.data_processing.chip import ChipConfig, AcquisitionGroupConfig, OutputModalityConfig, NoSplitConfig
        from lfm.data_processing.tiling import TileConfig, TileSourceConfig
        from lfm.data_processing.tiling.vector_index import IndexedRaster
        for north, name in ((True, 'north'), (False, 'south')):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                req = chip_request_from_aoi(sample_id=name, split_group_key=name,
                                            geographic_aoi=cap(north), static_only=True)
                grid = req.target_grid
                path = root / 'source.tif'
                # Wide constant source prevents resampling at source edges from
                # confusing the rectangle-versus-cap coverage assertion.
                ds = gdal.GetDriverByName('GTiff').Create(str(path), 1000, 1000, 1, gdal.GDT_Float32)
                ds.SetProjection(grid.crs_wkt)
                ds.SetGeoTransform((450000, 100, 0, 550000, 0, -100))
                band = ds.GetRasterBand(1)
                band.SetDescription('elevation')
                band.SetMetadataItem('Name', 'elevation')
                band.SetNoDataValue(-32768)
                band.Fill(7)
                band = ds = None
                label = root / 'label.npy'
                np.save(label, np.ones((grid.height, grid.width), dtype=np.uint8))
                req = replace(req, label_input=LabelInput(label, source_grid=grid, relation='clip_to_target'))
                source = TileSourceConfig('static', root, root / 'index.gpkg', band_names=('elevation',))
                config = ChipConfig(root / 'out', label, (AcquisitionGroupConfig('polar', TileConfig(root / 'tiles', 4, (source,))),),
                                    (OutputModalityConfig('polar', 'static', 'static'),), split_config=NoSplitConfig())
                # Only source-index discovery is substituted. Tile discovery,
                # full-cap geometry, warps, records and publication are real.
                with patch('lfm.data_processing.tiling.configured_tiler.query_source_index_envelopes',
                           return_value=(IndexedRaster(path),)):
                    batch = create_chips([req], config, max_workers=1)
                result = batch.results[0]
                self.assertEqual(result.status, 'success', result.message)
                ds = gdal.Open(str(result.chip_path))
                np.testing.assert_allclose(ds.ReadAsArray(), 7, atol=1e-6)
                ds = None
