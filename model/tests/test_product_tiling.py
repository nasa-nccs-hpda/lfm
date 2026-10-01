from pathlib import Path
from unittest import mock
import unittest

from lfm.model.product_tiling import (
    MissingRequiredProductError,
    create_tiles_for_aoi_by_product,
    discover_products_for_aoi,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig
from lfm.model.tiling_results import TileCubeRecord, TileSourceError
from lfm.model.vector_index import IndexedRaster


BOUNDS = {
    "ul_lat": 2.0,
    "ul_lon": 10.0,
    "lr_lat": 1.0,
    "lr_lon": 11.0,
}


class ProductTilingTestCase(unittest.TestCase):
    def source(self, name="wac", **overrides):
        values = {
            "name": name,
            "data_dir": Path(f"/data/{name}"),
            "index_path": Path(f"/data/{name}/index.gpkg"),
            "selection_mode": "product_id",
        }
        values.update(overrides)
        return TileSourceConfig(**values)

    def record(
        self,
        source_name,
        *,
        product_id,
        tile_x=1,
        tile_y=2,
    ):
        return TileCubeRecord(
            source_name=source_name,
            zone="42N",
            zoom_level=5,
            tile_x=tile_x,
            tile_y=tile_y,
            product_id=product_id,
            path=Path(f"/output/{source_name}-{product_id}-{tile_x}.tif"),
            band_names=("band",),
            crs_wkt="PROJCRS[test]",
            nodata_values=(None,),
        )

    def config(self, *, dynamic_required=True):
        dynamic = self.source(required=dynamic_required)
        static = self.source(
            "static",
            selection_mode="all_intersecting",
        )
        return TileConfig(Path("/output"), 5, (dynamic, static))

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_omitted_pid_writes_each_product_and_static_once(
        self,
        query,
        create_strict,
    ):
        config = self.config()
        query.return_value = [
            IndexedRaster(Path("/data/wac/M200.ech.cog.tif")),
            IndexedRaster(Path("/data/wac/M100.prj.vis.mos.tif")),
            IndexedRaster(Path("/data/wac/M100.prj.uv.mos.tif")),
        ]

        def create_side_effect(run_config, **kwargs):
            source = run_config.sources[0]
            selectors = kwargs.get("selectors")
            if source.name == "static":
                return [
                    self.record("static", product_id=None, tile_x=2),
                    self.record("static", product_id=None, tile_x=1),
                ]
            product_id = selectors["wac"]
            if product_id == "M100":
                return [
                    self.record("wac", product_id=product_id, tile_x=2),
                    self.record("wac", product_id=product_id, tile_x=1),
                ]
            return [self.record("wac", product_id=product_id)]

        create_strict.side_effect = create_side_effect
        logger = mock.Mock()

        records = create_tiles_for_aoi_by_product(
            config,
            **BOUNDS,
            product_ids={"wac": None},
            logger=logger,
        )

        self.assertEqual(
            [
                (record.tile_x, record.source_name, record.product_id)
                for record in records
            ],
            [
                (1, "wac", "M100"),
                (1, "wac", "M200"),
                (1, "static", None),
                (2, "wac", "M100"),
                (2, "static", None),
            ],
        )
        self.assertEqual(create_strict.call_count, 3)
        static_calls = [
            call
            for call in create_strict.call_args_list
            if call.args[0].sources[0].name == "static"
        ]
        self.assertEqual(len(static_calls), 1)
        logger.info.assert_any_call(
            "Discovered %d product ID(s) for source %r: %s",
            2,
            "wac",
            "M100, M200",
        )

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_explicit_pid_runs_only_requested_product(self, query, create_strict):
        config = self.config()
        query.return_value = [
            IndexedRaster(Path("/data/wac/M100.prj.vis.mos.tif")),
            IndexedRaster(Path("/data/wac/M200.ech.cog.tif")),
        ]

        def create_side_effect(run_config, **kwargs):
            source = run_config.sources[0]
            if source.name == "static":
                return [self.record("static", product_id=None)]
            return [self.record("wac", product_id=kwargs["selectors"]["wac"])]

        create_strict.side_effect = create_side_effect

        records = create_tiles_for_aoi_by_product(
            config,
            **BOUNDS,
            product_ids={"wac": "M200"},
        )

        self.assertEqual(
            [(record.source_name, record.product_id) for record in records],
            [("wac", "M200"), ("static", None)],
        )
        dynamic_call = create_strict.call_args_list[0]
        self.assertEqual(dynamic_call.kwargs["selectors"], {"wac": "M200"})

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index", return_value=[])
    def test_no_required_product_fails_before_tiling(self, query, create_strict):
        del query
        with self.assertRaisesRegex(
            MissingRequiredProductError,
            "no products intersect",
        ):
            create_tiles_for_aoi_by_product(self.config(), **BOUNDS)

        create_strict.assert_not_called()

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_missing_explicit_required_product_fails_before_tiling(
        self,
        query,
        create_strict,
    ):
        query.return_value = [
            IndexedRaster(Path("/data/wac/M100.prj.vis.mos.tif"))
        ]

        with self.assertRaisesRegex(
            MissingRequiredProductError,
            "M200.*does not intersect",
        ):
            create_tiles_for_aoi_by_product(
                self.config(),
                **BOUNDS,
                product_ids={"wac": "M200"},
            )

        create_strict.assert_not_called()

    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_contextual_source_rejects_product_id(self, query):
        del query
        with self.assertRaisesRegex(ValueError, "product_id sources"):
            discover_products_for_aoi(
                self.config(),
                **BOUNDS,
                product_ids={"static": None},
            )

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index", return_value=[])
    def test_optional_source_without_products_still_writes_static(
        self,
        query,
        create_strict,
    ):
        del query
        static_record = self.record("static", product_id=None)
        create_strict.return_value = [static_record]

        records = create_tiles_for_aoi_by_product(
            self.config(dynamic_required=False),
            **BOUNDS,
        )

        self.assertEqual(records, [static_record])
        create_strict.assert_called_once()
        self.assertEqual(create_strict.call_args.args[0].sources[0].name, "static")

    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_filename_component_collision_is_rejected(self, query):
        def resolver(path):
            return "A/B" if path.stem == "first" else "A-B"

        source = self.source(product_id_resolver=resolver)
        config = TileConfig(Path("/output"), 5, (source,))
        query.return_value = [
            IndexedRaster(Path("/data/wac/first.tif")),
            IndexedRaster(Path("/data/wac/second.tif")),
        ]

        with self.assertRaisesRegex(ValueError, "same filename component"):
            discover_products_for_aoi(config, **BOUNDS)

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_product_specific_failure_retains_pid(self, query, create_strict):
        config = TileConfig(Path("/output"), 5, (self.source(),))
        query.return_value = [
            IndexedRaster(Path("/data/wac/M100.ech.cog.tif")),
            IndexedRaster(Path("/data/wac/M200.ech.cog.tif")),
        ]
        completed = self.record("wac", product_id="M100")

        def create_side_effect(run_config, **kwargs):
            del run_config
            product_id = kwargs["selectors"]["wac"]
            if product_id == "M100":
                return [completed]
            raise TileSourceError(
                "synthetic product failure",
                source_name="wac",
                zone="42N",
                tile_x=1,
                tile_y=2,
                product_id=product_id,
            )

        create_strict.side_effect = create_side_effect

        with self.assertRaises(TileSourceError) as raised:
            create_tiles_for_aoi_by_product(config, **BOUNDS)

        self.assertEqual(raised.exception.product_id, "M200")
        self.assertEqual(raised.exception.completed_records, (completed,))

    @mock.patch("lfm.model.product_tiling._create_tiles_for_aoi_strict")
    @mock.patch("lfm.model.product_tiling.query_source_index")
    def test_required_discovered_product_must_write_a_cube(
        self,
        query,
        create_strict,
    ):
        config = TileConfig(Path("/output"), 5, (self.source(),))
        query.return_value = [IndexedRaster(Path("/data/wac/M100.ech.cog.tif"))]
        create_strict.return_value = []

        with self.assertRaisesRegex(
            MissingRequiredProductError,
            "M100.*produced no tile cubes",
        ):
            create_tiles_for_aoi_by_product(config, **BOUNDS)


if __name__ == "__main__":
    unittest.main()
