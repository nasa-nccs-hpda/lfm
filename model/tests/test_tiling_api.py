from pathlib import Path
from unittest import mock
import unittest

from lfm.model.tiling import (
    create_tiles_for_aoi,
    create_tiles_for_index,
    create_tiles_for_point,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig


class TilingApiTestCase(unittest.TestCase):
    def config(self) -> TileConfig:
        source = TileSourceConfig(
            name="static",
            data_dir=Path("/data/static"),
            index_path=Path("/data/static/index.shp"),
        )
        return TileConfig(Path("/output"), 5, (source,))

    @mock.patch("lfm.model.tiling._tiler_cls")
    def test_tile_index_api(self, tiler_factory):
        tiler_cls = tiler_factory.return_value
        tiler_cls.return_value.run_tile_index.return_value = ["record"]
        config = self.config()

        result = create_tiles_for_index(
            config,
            tile_x=1,
            tile_y=63,
            zone="42N",
        )

        tiler_cls.assert_called_once_with(config, selectors=None)
        tiler_cls.return_value.run_tile_index.assert_called_once_with(1, 63, "42N")
        self.assertEqual(result, ["record"])

    @mock.patch("lfm.model.tiling._tiler_cls")
    def test_point_api(self, tiler_factory):
        tiler_cls = tiler_factory.return_value
        config = self.config()

        create_tiles_for_point(
            config,
            lat=1.2,
            lon=149.8,
            zone="42N",
        )

        tiler_cls.return_value.run_point.assert_called_once_with(1.2, 149.8, "42N")

    @mock.patch("lfm.model.tiling._tiler_cls")
    def test_aoi_api(self, tiler_factory):
        tiler_cls = tiler_factory.return_value
        config = self.config()

        create_tiles_for_aoi(
            config,
            ul_lat=1.3,
            ul_lon=149.7,
            lr_lat=1.1,
            lr_lon=149.9,
        )

        tiler_cls.return_value.run_aoi.assert_called_once_with(1.3, 149.7, 1.1, 149.9)


if __name__ == "__main__":
    unittest.main()
