"""LTM/polar seam routing, zoom contracts and per-pixel compositing."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lfm.data_processing.chip import GeographicAOI, chip_grid_family, static_grid_reference
from lfm.data_processing.chip.chip_requests import geographic_query_parts
from lfm.data_processing.chip.chip_acquisition import acquire_prepared_request
from lfm.data_processing.tiling.grid_registry import GridFamily, default_grid_registry
from lfm.data_processing.tests.chip import test_chip_acquisition as fixtures


def seam_acquire(prepared, config):
    """Synthetic acquisition, real distinct LTM/LPS rasters for downstream tests."""
    from osgeo import gdal
    from lfm.data_processing.tests.chip.test_chip_a5_integration import synthetic_acquire
    base = synthetic_acquire(prepared, config)
    group = base.group_results[0]
    template = group.records[0]
    records = []
    aoi = prepared.request.geographic_aoi
    north = aoi.upper_left_latitude > 0
    for zone in (('23N', 'LPS_N') if north else ('23S', 'LPS_S')):
        crs = default_grid_registry()[zone].crs_wkt
        path = template.path.with_name(f'{zone}.tif')
        ds = gdal.Warp(str(path), str(template.path), dstSRS=crs, xRes=100, yRes=100,
                       resampleAlg='near', dstNodata=-32768.)
        if ds is None:
            raise RuntimeError('Synthetic seam cube warp failed')
        ds.GetRasterBand(1).SetDescription('elevation')
        ds.GetRasterBand(1).SetMetadataItem('Name', 'elevation')
        ds = None
        records.append(replace(template, path=path, zone=zone, crs_wkt=crs,
                               zoom_level=4 if zone.startswith('LPS') else 5))
    return replace(base, group_results=(replace(group, records=tuple(records)),))


def seam_worker(task):
    from lfm.data_processing.chip.chip_creation import _run_prepared_request
    with patch('lfm.data_processing.chip.chip_creation.acquire_prepared_request', side_effect=seam_acquire):
        return _run_prepared_request(*task)


class SeamContractsTestCase(unittest.TestCase):
    def test_family_zoom_dictionary_and_manifest_identity(self):
        from lfm.data_processing.tests.chip.test_chip_config import ChipConfigTestCase
        from lfm.data_processing.chip.chip_publication import _configuration_document, _configuration_id
        helper = ChipConfigTestCase()
        config = helper.dictionary_config({'wac': helper.source_dict('wac')}, acquisition_groups={
            'sensor_grid': {'sources': {'wac': helper.source_dict('wac')},
                            'ltm_zoom_level': 5, 'polar_zoom_level': 4}})
        group = config.acquisition_groups[0]
        self.assertEqual(group.zoom_for_family(GridFamily.LPS_N), 4)
        document = _configuration_document(config)
        altered = replace(config, acquisition_groups=(replace(group, polar_zoom_level=6),))
        self.assertNotEqual(_configuration_id(document), _configuration_id(_configuration_document(altered)))

    def test_empty_required_family_fails_without_trying_later_part(self):
        helper = fixtures.ChipAcquisitionTestCase()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = replace(helper.group('mixed', root, 5, helper.source('wac')),
                            ltm_zoom_level=5, polar_zoom_level=4)
            config = helper.config(root, (group,))
            req = helper.request(aoi=GeographicAOI(82.1, 10, 81.9, 11))
            with patch('lfm.data_processing.chip.chip_acquisition.create_tiles_for_aoi', return_value=[]) as tiler:
                result = acquire_prepared_request(helper.prepared(req), config)
            self.assertEqual(result.status, 'failed')
            self.assertEqual(tiler.call_count, 1)
            self.assertEqual(result.diagnostics[0].zoom_level, 4)

    def test_both_hemispheres_and_wrapped_seams_partition_without_gaps(self):
        for north, south, polar in ((82.1, 81.9, GridFamily.LPS_N),
                                     (-81.9, -82.1, GridFamily.LPS_S)):
            for west, east in ((10, 11), (179.9, -179.9)):
                aoi = GeographicAOI(north, west, south, east)
                parts = geographic_query_parts(aoi)
                self.assertEqual(len(parts), 2 if west == 10 else 4)
                self.assertEqual({chip_grid_family(p) for p in parts}, {GridFamily.LTM, polar})
                area = sum((p.upper_left_latitude - p.lower_right_latitude)
                           * (p.lower_right_longitude - p.upper_left_longitude) for p in parts)
                self.assertAlmostEqual(area, (north - south) * ((east - west) % 360))

    def test_static_grid_follows_center_and_scalar_zoom_stays_authoritative(self):
        for north, south, grid_id in ((82.01, 81.9, '24N'), (82.1, 81.99, 'LPS_N'),
                                      (-81.9, -82.01, '24S'), (-81.99, -82.1, 'LPS_S')):
            grid = static_grid_reference(GeographicAOI(north, 10, south, 11))
            self.assertEqual(grid.crs_wkt, default_grid_registry()[grid_id].crs_wkt)
        helper = fixtures.ChipAcquisitionTestCase()
        group = helper.group('mixed', Path('/tmp/seams'), 9, helper.source('wac'))
        self.assertEqual(group.zoom_for_family(GridFamily.LTM), 9)
        self.assertEqual(group.zoom_for_family(GridFamily.LPS_N), 9)
        group = replace(group, ltm_zoom_level=11, polar_zoom_level=10)
        self.assertEqual(group.zoom_for_family(GridFamily.LTM), 11)
        self.assertEqual(group.zoom_for_family(GridFamily.LPS_S), 10)
        for values in ({'polar_zoom_level': 16}, {'ltm_zoom_level': True}, {'ltm_zoom_level': 0}):
            with self.assertRaises(ValueError):
                replace(group, **values)

    def test_acquisition_uses_family_zooms_and_retains_earlier_records_on_failure(self):
        from lfm.data_processing.chip.chip_reprojection import build_modality_cube_mappings, ChipReprojectionError
        from lfm.data_processing.tiling.tiling_results import TileSourceError
        helper = fixtures.ChipAcquisitionTestCase()
        for north, south, polar, ltm in ((82.1, 81.9, 'LPS_N', '24N'),
                                         (-81.9, -82.1, 'LPS_S', '24S')):
            for fail in (False, True):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    group = replace(helper.group('mixed', root, 9, helper.source('wac', selection_mode='product_id')),
                                    ltm_zoom_level=5, polar_zoom_level=4)
                    config = helper.config(root, (group,))
                    req = helper.request(aoi=GeographicAOI(north, 10, south, 11))
                    calls = []

                    def acquire(cfg, **kwargs):
                        family = chip_grid_family(GeographicAOI(kwargs['ul_lat'], kwargs['ul_lon'],
                                                               kwargs['lr_lat'], kwargs['lr_lon']))
                        zone = ltm if family == GridFamily.LTM else polar
                        self.assertEqual(cfg.zoom_level, 5 if zone == ltm else 4)
                        self.assertEqual(kwargs['selectors'], {'wac': 'M100'})
                        record = helper.record(cfg.output_dir / f'{zone}.tif', zone=zone, zoom=cfg.zoom_level)
                        calls.append(record)
                        if fail and len(calls) == 2:
                            raise TileSourceError('failure', source_name='wac', zone=zone,
                                                  tile_x=0, tile_y=0, completed_records=(record,))
                        return [record]

                    with patch('lfm.data_processing.chip.chip_acquisition.create_tiles_for_aoi', side_effect=acquire):
                        result = acquire_prepared_request(helper.prepared(req), config)
                    self.assertEqual(set(result.records), set(calls))
                    self.assertEqual(result.status, 'failed' if fail else 'complete')
                    self.assertEqual(result.group_results[0].query_zoom_levels, tuple(r.zoom_level for r in calls))
                    if not fail:
                        self.assertEqual(len(build_modality_cube_mappings(result, config)[0].zone_groups), 2)
                        bad = replace(result, group_results=(replace(result.group_results[0], records=(
                            replace(calls[0], zoom_level=9), calls[1])),))
                        with self.assertRaises(ChipReprojectionError):
                            build_modality_cube_mappings(bad, config)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class SeamCompositeTestCase(unittest.TestCase):
    def test_preferred_family_fallback_per_band_and_order_independence(self):
        import numpy as np
        from lfm.data_processing.chip.chip_reprojection import (
            reproject_modality, ModalityCubeMapping, SourceZoneGroup,
        )
        from lfm.data_processing.chip import OutputModalityConfig
        helper = fixtures.ChipAcquisitionTestCase()
        req = helper.request()
        preferred = np.array([[True, True, True], [False, False, False], [True, False, True]])
        ltm = np.full((2, 3, 3), 10.)
        polar = np.full((2, 3, 3), 20.)
        lm, pm = np.ones((2, 3, 3), dtype=bool), np.ones((2, 3, 3), dtype=bool)
        pm[0, 0, 0] = False  # LTM fallback inside polar region.
        lm[0, 1, 0] = False  # Polar fallback inside LTM region.
        lm[1, 2, 2] = pm[1, 2, 2] = False  # Neither covers this band/pixel.
        for polar_zone in ('LPS_N', 'LPS_S'):
            ltm_zone = '24N' if polar_zone == 'LPS_N' else '24S'
            records = (helper.record(Path('/tmp/a'), zone=ltm_zone),
                       helper.record(Path('/tmp/b'), zone=polar_zone, zoom=4))
            groups = tuple(SourceZoneGroup('mixed', 'wac', r.zone, r.zoom_level, (r,)) for r in records)
            mapping = ModalityCubeMapping(OutputModalityConfig('mixed', 'wac', 'wac'),
                                          helper.source('wac'), records, groups)
            def warp(mapping, group, target, nodata):
                return (ltm, lm, ('a', 'b')) if group.zone == ltm_zone else (polar, pm, ('a', 'b'))
            outputs = []
            for ordered in (groups, groups[::-1]):
                with patch('lfm.data_processing.chip.chip_reprojection._libraries', return_value=(np, None, None)), \
                     patch('lfm.data_processing.chip.chip_reprojection._polar_target_pixels', return_value=preferred), \
                     patch('lfm.data_processing.chip.chip_reprojection._warp_zone_group', side_effect=warp):
                    outputs.append(reproject_modality(replace(mapping, zone_groups=ordered), req.target_grid,
                                                       output_nodata=-32768).pixels)
            np.testing.assert_array_equal(*outputs)
            expected = np.broadcast_to(np.where(preferred, 20., 10.), (2, 3, 3)).copy()
            expected[0, 0, 0], expected[0, 1, 0], expected[1, 2, 2] = 10, 20, -32768
            np.testing.assert_array_equal(outputs[0], expected)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('numpy', 'osgeo')), 'GDAL/NumPy required')
class SeamGridTestCase(unittest.TestCase):
    def test_real_ltm_and_polar_cubes_obey_latitude_precedence(self):
        import numpy as np
        from osgeo import osr
        from lfm.data_processing.tests.chip.test_chip_reprojection import ChipReprojectionRasterTestCase
        from lfm.data_processing.chip.chip_requests import target_grid_from_geographic_aoi, target_grid_from_pixel_bounds
        from lfm.data_processing.chip.chip_reprojection import reproject_modality, _polar_target_pixels
        from lfm.data_processing.chip import TargetGrid
        helper = ChipReprojectionRasterTestCase()
        helper.setUp()
        for north, south, ltm, polar in ((82.02, 81.98, '23N', 'LPS_N'),
                                        (-81.98, -82.02, '23S', 'LPS_S')):
            with tempfile.TemporaryDirectory() as tmp:
                aoi = GeographicAOI(north, -.05, south, .05)
                target = target_grid_from_geographic_aoi(aoi, static_grid_reference(aoi))
                records = []
                for zone, value in ((ltm, 10), (polar, 20)):
                    lattice = TargetGrid(default_grid_registry()[zone].crs_wkt,
                                         (0, 100, 0, 0, 0, -100), (0, -100, 100, 0), 1, 1)
                    grid = target_grid_from_geographic_aoi(aoi, lattice)
                    grid = target_grid_from_pixel_bounds(grid, (-10, -10, grid.width + 10, grid.height + 10))
                    records.append(helper.cube(Path(tmp) / f'{zone}.tif',
                                               np.full((grid.height, grid.width), value),
                                               transform=grid.transform, zone=zone, crs_wkt=grid.crs_wkt))
                mapping = helper.mapping(records)
                actual = reproject_modality(mapping, target, output_nodata=-32768)
                expected = np.where(_polar_target_pixels(target, np, osr), 20, 10)
                self.assertTrue(actual.valid_mask.all())
                np.testing.assert_allclose(actual.pixels[0], expected, atol=1e-8)

    def test_seam_labels_publish_identically_serial_and_spawn(self):
        import numpy as np
        from lfm.data_processing.chip import chip_request_from_aoi, LabelInput, create_chips
        from lfm.data_processing.tests.chip.test_chip_creation import ChipCreationRasterTestCase
        from lfm.data_processing.tests.chip.test_chip_instance_labels import InstanceConversionTestCase
        from lfm.data_processing.chip.chip_instance_labels import _rectangle
        helper = InstanceConversionTestCase()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = []
            for name, north, south in (('north', 82.02, 81.98), ('south', -81.98, -82.02)):
                req = chip_request_from_aoi(sample_id=name, split_group_key=name, static_only=True,
                                            geographic_aoi=GeographicAOI(north, -.05, south, .05))
                grid = req.target_grid
                label = root / f'{name}.npy'
                np.save(label, np.ones((grid.height, grid.width), dtype=np.uint8))
                requests.append(replace(req, sample_id=name + '_semantic', label_input=LabelInput(
                    label, source_grid=grid, relation='clip_to_target')))
                gpkg = helper.gpkg(grid, [(38, _rectangle(0, 0, grid.width, grid.height))], name=f'{name}.gpkg')
                requests.append(replace(req, label_input=LabelInput(gpkg, relation='clip_to_target', layer='craters')))
            config = ChipCreationRasterTestCase().config(root)
            config = replace(config, acquisition_groups=tuple(replace(g, ltm_zoom_level=5, polar_zoom_level=4)
                                                               for g in config.acquisition_groups))
            outputs = []
            for workers in (1, 2):
                with patch('lfm.data_processing.chip.chip_creation.acquire_prepared_request', side_effect=seam_acquire), \
                     patch('lfm.data_processing.chip.chip_creation._run_prepared_task', new=seam_worker):
                    batch = create_chips(requests, config, max_workers=workers, overwrite=workers == 2)
                self.assertEqual([r.status for r in batch.results], ['success'] * 4, [r.message for r in batch.results])
                for result in batch.results:
                    if result.label_path.suffix == '.npz':
                        with np.load(result.label_path) as archive:
                            self.assertEqual(int(archive['num_craters']), 1)  # Not duplicated at seam.
                            self.assertTrue((archive['mask'] == 1).all())
                    else:
                        self.assertTrue((np.load(result.label_path) == 1).all())
                outputs.append([(r.chip_path.read_bytes(), r.label_path.read_bytes()) for r in batch.results])
            self.assertEqual(outputs[0], outputs[1])

    def test_native_grid_preservation_and_pixel_center_threshold(self):
        import numpy as np
        from osgeo import osr
        from lfm.data_processing.chip import chip_request_from_aoi
        from lfm.data_processing.chip.chip_reprojection import _polar_target_pixels
        from lfm.data_processing.chip.chip_requests import target_grid_from_bounds
        from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt
        for north, south in ((82.02, 81.98), (-81.98, -82.02)):
            for west, east in ((-.02, .02), (179.98, -179.98)):
                aoi = GeographicAOI(north, west, south, east)
                source = static_grid_reference(aoi)
                req = chip_request_from_aoi(sample_id='seam', split_group_key='seam', geographic_aoi=aoi,
                                            source_grid=source)
                self.assertEqual(req.target_grid.crs_wkt, source.crs_wkt)
                mask = _polar_target_pixels(req.target_grid, np, osr)
                self.assertTrue(mask.any())
                self.assertTrue((~mask).any())
                self.assertEqual(len(geographic_query_parts(req.geographic_aoi)), 4 if west > 179 else 2)
        grid = target_grid_from_bounds(crs_wkt=load_lunar_geographic_wkt(),
                                       bounds=(0, 81.985, .03, 82.015), width=3, height=3)
        mask = _polar_target_pixels(grid, np, osr)
        np.testing.assert_array_equal(mask[:, 0], [True, True, False])
