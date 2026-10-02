import importlib.util
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.grid_registry import GeographicCoverage, GridFamily
from lfm.model.static_band_contract import STATIC_BAND_NAMES, STATIC_OUTPUT_NODATA
from lfm.model.tiling_config import TileSourceConfig
from lfm.model.tiling_results import TileCubeRecord, TileSourceError, tile_cube_filename
from lfm.model.tiling_workflow import (
    AutomaticTilingError,
    ProductAOIWarning,
    TileAOIQuery,
    TilePointQuery,
    TileSourceDefinition,
    create_tiles_for_query,
    default_zoom_for_modality,
    make_nac_tile_source,
    make_static_tile_source,
    make_wac_tile_source,
    resolve_tile_index_path,
)
from lfm.model.vector_index import IndexedRaster


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


class FakeTileDefinition:
    def __init__(self, grid_id, zoom_level):
        self.grid_id = grid_id
        self.zoom_level = zoom_level

    def getOverlappingTiles(self, *bounds):
        del bounds
        tile_x = 2 if self.grid_id == "LPS_N" else 1
        return [(tile_x, 3)]

    def llToTileIndex(self, lat, lon):
        del lat, lon
        tile_x = 2 if self.grid_id == "LPS_N" else 1
        return tile_x, 3

    def geographic_query_envelopes(self, tile_x, tile_y):
        del tile_x, tile_y
        return (
            GeographicCoverage(
                south=84.0,
                west=9.0,
                north=86.0,
                east=11.0,
            ),
        )


class AutomaticTilingWorkflowTestCase(unittest.TestCase):
    def setUp(self):
        self.tile_definition = mock.patch(
            "lfm.model.tiling_workflow._tile_definition_for_grid",
            side_effect=lambda grid_id, zoom: FakeTileDefinition(grid_id, zoom),
        )
        self.prepare = mock.patch(
            "lfm.model.tiling_workflow.prepare_tile_config",
            return_value=mock.Mock(indexes=()),
        )
        self.query_aoi = mock.patch(
            "lfm.model.tiling_workflow.query_source_index",
            return_value=[IndexedRaster(Path("/data/wac/M100.tif"))],
        )
        self.query_envelopes = mock.patch(
            "lfm.model.tiling_workflow.query_source_index_envelopes",
            return_value=[IndexedRaster(Path("/data/wac/M100.tif"))],
        )
        self.create = mock.patch(
            "lfm.model.tiling_workflow.create_tiles_for_index",
            side_effect=self._create_record,
        )
        self.tile_definition_mock = self.tile_definition.start()
        self.prepare_mock = self.prepare.start()
        self.query_aoi_mock = self.query_aoi.start()
        self.query_envelopes_mock = self.query_envelopes.start()
        self.create_mock = self.create.start()
        self.addCleanup(mock.patch.stopall)

    @staticmethod
    def _create_record(
        config,
        *,
        grid_id,
        tile_x,
        tile_y,
        selectors=None,
    ):
        source = config.sources[0]
        product_id = None if selectors is None else selectors[source.name]
        path = config.output_dir / tile_cube_filename(
            source_name=source.name,
            grid_id=grid_id,
            zoom_level=config.zoom_level,
            tile_x=tile_x,
            tile_y=tile_y,
            product_id=product_id,
        )
        return [
            TileCubeRecord(
                source_name=source.name,
                zone=grid_id,
                zoom_level=config.zoom_level,
                tile_x=tile_x,
                tile_y=tile_y,
                product_id=product_id,
                path=path,
                band_names=("band",),
                crs_wkt="PROJCRS[test]",
                nodata_values=(None,),
            )
        ]

    def wac(self, **kwargs):
        return make_wac_tile_source(data_dir="/data/wac", **kwargs)

    def static(self, **kwargs):
        return make_static_tile_source(data_dir="/data/static", **kwargs)

    def test_ltm_aoi_uses_wac_zoom_five_and_static_once(self):
        records = create_tiles_for_query(
            query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            static_sources=(self.static(),),
            product_ids="M100",
        )

        self.assertEqual(
            [(record.source_name, record.zoom_level) for record in records],
            [("wac", 5), ("static", 5)],
        )
        self.assertEqual(self.prepare_mock.call_count, 2)
        prepared = tuple(
            call.kwargs["sources"][0]
            for call in self.prepare_mock.call_args_list
        )
        self.assertEqual(
            [item.source.name for item in prepared],
            ["wac", "static"],
        )
        self.assertEqual(self.create_mock.call_count, 2)

    def test_north_polar_aoi_is_dynamic_only_at_wac_zoom_four(self):
        class DisabledStatic:
            def __iter__(self):
                raise AssertionError("disabled static sources must not be touched")

        records = create_tiles_for_query(
            query=TileAOIQuery(86.0, 9.0, 84.0, 11.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            static_sources=DisabledStatic(),
            product_ids="M100",
            include_static=False,
        )

        self.assertEqual(
            [(record.grid_id, record.zoom_level) for record in records],
            [("LPS_N", 4)],
        )

    def test_cross_threshold_uses_family_specific_zooms(self):
        records = create_tiles_for_query(
            query=TileAOIQuery(83.0, 149.0, 81.0, 151.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            include_static=False,
            product_ids="M100",
        )

        self.assertEqual(
            [(record.grid_id, record.zoom_level) for record in records],
            [("42N", 5), ("LPS_N", 4)],
        )
        self.assertEqual(self.query_aoi_mock.call_count, 2)

    def test_omitted_products_are_separate_and_static_is_written_once(self):
        self.query_aoi_mock.return_value = [
            IndexedRaster(Path("/data/wac/M200.tif")),
            IndexedRaster(Path("/data/wac/M100.uv.tif")),
            IndexedRaster(Path("/data/wac/M100.vis.tif")),
        ]

        records = create_tiles_for_query(
            query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            static_sources=(self.static(),),
        )

        self.assertEqual(
            [(record.source_name, record.product_id) for record in records],
            [("wac", "M100"), ("wac", "M200"), ("static", None)],
        )
        self.assertEqual(self.create_mock.call_count, 3)

    def test_missing_product_warns_and_skips_contextual_static(self):
        self.query_aoi_mock.return_value = []

        with self.assertWarnsRegex(
            ProductAOIWarning,
            "Skipping source 'wac'.*no products intersect.*42N",
        ):
            records = create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(self.wac(),),
                static_sources=(self.static(),),
            )

        self.assertEqual(records, [])
        self.create_mock.assert_not_called()

    def test_invalid_explicit_product_warns_and_other_product_continues(self):
        nac = make_nac_tile_source(data_dir="/data/nac")

        def query(source, **bounds):
            del bounds
            if source.name == "wac":
                return []
            return [IndexedRaster(Path("/data/nac/NAC100.tif"))]

        self.query_aoi_mock.side_effect = query
        with self.assertWarnsRegex(
            ProductAOIWarning,
            "product 'WAC404' does not intersect",
        ):
            records = create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(self.wac(), nac),
                include_static=False,
                product_ids={"wac": "WAC404", "nac": "NAC100"},
            )

        self.assertEqual(
            [(record.source_name, record.product_id) for record in records],
            [("nac", "NAC100")],
        )

    def test_polar_point_uses_tile_envelope_for_product_discovery(self):
        records = create_tiles_for_query(
            query=TilePointQuery(85.0, 10.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            include_static=False,
        )

        self.assertEqual(records[0].grid_id, "LPS_N")
        self.query_aoi_mock.assert_not_called()
        self.query_envelopes_mock.assert_called_once()
        envelopes = self.query_envelopes_mock.call_args.args[1]
        self.assertEqual(len(envelopes), 1)

    def test_polar_static_requires_explicit_verified_support(self):
        with self.assertRaisesRegex(AutomaticTilingError, "verified polar"):
            create_tiles_for_query(
                query=TileAOIQuery(86.0, 9.0, 84.0, 11.0),
                output_dir="/output",
                dynamic_sources=(self.wac(),),
                static_sources=(self.static(),),
                product_ids="M100",
            )

        self.prepare_mock.assert_not_called()
        self.query_aoi_mock.assert_not_called()

    def test_static_only_requires_a_family_zoom_override(self):
        with self.assertRaisesRegex(AutomaticTilingError, "zoom override"):
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                static_sources=(self.static(),),
                include_dynamic=False,
            )

        records = create_tiles_for_query(
            query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
            output_dir="/output",
            static_sources=(self.static(zoom_overrides={"ltm": 5}),),
            include_dynamic=False,
        )
        self.assertEqual(
            [(record.source_name, record.zoom_level) for record in records],
            [("static", 5)],
        )

    def test_static_only_rejects_a_product_id_before_preparation(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                static_sources=(self.static(zoom_overrides={"ltm": 5}),),
                include_dynamic=False,
                product_ids="M100",
            )
        self.prepare_mock.assert_not_called()

    def test_multiple_dynamic_modalities_require_a_product_mapping(self):
        nac = make_nac_tile_source(data_dir="/data/nac")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(self.wac(), nac),
                include_static=False,
                product_ids="M100",
            )

        def query(source, **bounds):
            del bounds
            return [IndexedRaster(source.data_dir / f"{source.name.upper()}100.tif")]

        self.query_aoi_mock.side_effect = query
        records = create_tiles_for_query(
            query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
            output_dir="/output",
            dynamic_sources=(self.wac(), nac),
            include_static=False,
            product_ids={"wac": "WAC100", "nac": "NAC100"},
        )

        self.assertEqual(
            [(record.source_name, record.zoom_level) for record in records],
            [("wac", 5), ("nac", 11)],
        )

    def test_custom_dynamic_source_requires_zoom_policy(self):
        source = TileSourceConfig(
            name="science",
            data_dir=Path("/data/science"),
            index_path=Path("/data/science/index.gpkg"),
            selection_mode="product_id",
        )
        custom = TileSourceDefinition(source, role="dynamic")
        self.assertFalse(custom.polar_supported)

        with self.assertRaisesRegex(AutomaticTilingError, "no default"):
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(custom,),
                include_static=False,
            )

    def test_antimeridian_parts_deduplicate_tile_addresses(self):
        records = create_tiles_for_query(
            query=TileAOIQuery(86.0, 179.0, 84.0, -179.0),
            output_dir="/output",
            dynamic_sources=(self.wac(),),
            include_static=False,
            product_ids="M100",
        )

        self.assertEqual(len(records), 1)
        self.create_mock.assert_called_once()
        self.assertEqual(self.query_aoi_mock.call_count, 2)

    def test_tile_failure_includes_complete_context_and_prior_records(self):
        def fail(config, **kwargs):
            raise TileSourceError(
                "synthetic failure",
                source_name=config.sources[0].name,
                zone=kwargs["grid_id"],
                tile_x=kwargs["tile_x"],
                tile_y=kwargs["tile_y"],
                product_id=kwargs["selectors"][config.sources[0].name],
            )

        self.create_mock.side_effect = fail
        with self.assertRaises(AutomaticTilingError) as raised:
            create_tiles_for_query(
                query=TileAOIQuery(86.0, 9.0, 84.0, 11.0),
                output_dir="/output",
                dynamic_sources=(self.wac(),),
                include_static=False,
                product_ids="M100",
            )

        error = raised.exception
        self.assertEqual(error.stage, "tile_generation")
        self.assertEqual(error.source_name, "wac")
        self.assertEqual(error.product_id, "M100")
        self.assertEqual(error.grid_id, "LPS_N")
        self.assertEqual(error.zoom_level, 4)
        self.assertEqual((error.tile_x, error.tile_y), (2, 3))

    def test_index_preparation_failure_identifies_source_and_stage(self):
        self.prepare_mock.side_effect = RuntimeError("synthetic index failure")

        with self.assertRaises(AutomaticTilingError) as raised:
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(self.wac(),),
                include_static=False,
            )

        self.assertEqual(raised.exception.stage, "index_preparation")
        self.assertEqual(raised.exception.source_name, "wac")
        self.query_aoi_mock.assert_not_called()

    def test_repeated_requests_have_identical_order_and_metadata(self):
        kwargs = {
            "query": TileAOIQuery(2.0, 149.0, 1.0, 151.0),
            "output_dir": "/output",
            "dynamic_sources": (self.wac(),),
            "static_sources": (self.static(),),
        }

        first = create_tiles_for_query(**kwargs)
        second = create_tiles_for_query(**kwargs)

        self.assertEqual(first, second)

    def test_filename_component_collision_is_rejected_before_preparation(self):
        first = self.wac(name="a/b")
        second = make_nac_tile_source(data_dir="/data/nac", name="a-b")

        with self.assertRaisesRegex(ValueError, "same filename component"):
            create_tiles_for_query(
                query=TileAOIQuery(2.0, 149.0, 1.0, 151.0),
                output_dir="/output",
                dynamic_sources=(first, second),
                include_static=False,
                product_ids={"a/b": "M100", "a-b": "N100"},
            )
        self.prepare_mock.assert_not_called()


class TileSourceDefinitionTestCase(unittest.TestCase):
    def test_index_resolution_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db2 = root / "db2.shp"
            db2.touch()

            self.assertEqual(
                resolve_tile_index_path(root, canonical_static=True),
                db2,
            )
            output_index = root / "output_index.shp"
            output_index.touch()
            self.assertEqual(
                resolve_tile_index_path(root, canonical_static=True),
                output_index,
            )
            explicit = root / "explicit.gpkg"
            self.assertEqual(
                resolve_tile_index_path(
                    root,
                    index_path=explicit,
                    canonical_static=True,
                ),
                explicit,
            )

    def test_builtin_zoom_and_nodata_contracts(self):
        wac = make_wac_tile_source(data_dir="/data/wac")
        nac = make_nac_tile_source(data_dir="/data/nac")
        static = make_static_tile_source(data_dir="/data/static")

        self.assertEqual(wac.configured_zoom(GridFamily.LTM), 5)
        self.assertEqual(wac.configured_zoom(GridFamily.LPS_N), 4)
        self.assertEqual(nac.configured_zoom(GridFamily.LTM), 11)
        self.assertEqual(nac.configured_zoom(GridFamily.LPS_S), 10)
        self.assertEqual(default_zoom_for_modality("wac", "lps_n"), 4)
        self.assertEqual(default_zoom_for_modality("nac", "ltm"), 11)
        self.assertIsNone(default_zoom_for_modality("static", "ltm"))
        self.assertIsNone(static.configured_zoom(GridFamily.LTM))
        self.assertEqual(static.source.band_names, STATIC_BAND_NAMES)
        self.assertEqual(static.source.output_nodata, STATIC_OUTPUT_NODATA)
        self.assertFalse(static.polar_supported)
        self.assertTrue(wac.source.preserve_source_nodata)
        self.assertTrue(nac.source.preserve_source_nodata)

    def test_builtin_presets_require_explicit_managed_cache_rebuild(self):
        default_wac = make_wac_tile_source(data_dir="/data/wac")
        managed_wac = make_wac_tile_source(
            data_dir="/data/wac",
            index_path="/cache/wac.gpkg",
            rebuild_invalid_index=True,
            index_worker_count=3,
        )

        self.assertFalse(default_wac.rebuild_invalid_index)
        self.assertTrue(managed_wac.rebuild_invalid_index)
        self.assertTrue(managed_wac.preparation().rebuild_invalid_index)
        self.assertEqual(managed_wac.index_worker_count, 3)
        self.assertEqual(managed_wac.preparation().worker_count, 3)

    def test_static_role_requires_contextual_selection(self):
        source = TileSourceConfig(
            name="bad_static",
            data_dir=Path("/data/static"),
            index_path=Path("/data/static/index.shp"),
            selection_mode="product_id",
        )
        with self.assertRaisesRegex(ValueError, "all_intersecting"):
            TileSourceDefinition(source, role="static")


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for workflow integration")
class AutomaticTilingWorkflowIntegrationTestCase(unittest.TestCase):
    def test_missing_index_is_created_before_real_ltm_tiling(self):
        import numpy as np
        from osgeo import gdal

        from lfm.model.lunar_crs import load_lunar_geographic_wkt

        gdal.UseExceptions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "wac"
            data_dir.mkdir()
            raster_path = data_dir / "M100.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(
                str(raster_path),
                128,
                128,
                1,
                gdal.GDT_Float32,
            )
            dataset.SetProjection(load_lunar_geographic_wkt())
            dataset.SetGeoTransform((149.7, 0.002, 0.0, 1.3, 0.0, -0.002))
            band = dataset.GetRasterBand(1)
            band.WriteArray(np.ones((128, 128), dtype=np.float32))
            band.SetNoDataValue(-9999.0)
            dataset = None

            source = make_wac_tile_source(
                data_dir=data_dir,
                image_glob=raster_path.name,
            )
            records = create_tiles_for_query(
                query=TileAOIQuery(1.25, 149.75, 1.1, 149.9),
                output_dir=root / "output",
                dynamic_sources=(source,),
                include_static=False,
                product_ids="M100",
            )

            self.assertTrue(source.source.index_path.is_file())
            self.assertTrue(records)
            self.assertTrue(all(record.grid_id == "42N" for record in records))
            self.assertTrue(all(record.zoom_level == 5 for record in records))
            self.assertTrue(all(record.path.is_file() for record in records))


if __name__ == "__main__":
    unittest.main()
