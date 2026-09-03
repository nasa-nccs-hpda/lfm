from pathlib import Path
import importlib.util
import tempfile
from unittest import mock
import unittest


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for configured tiler tests")
class ConfiguredTilerIntegrationTestCase(unittest.TestCase):
    def record(self, path, *, tile_x, source_name="static"):
        from lfm.model.tiling_results import TileCubeRecord

        return TileCubeRecord(
            source_name=source_name,
            zone="42N",
            zoom_level=5,
            tile_x=tile_x,
            tile_y=63,
            product_id=None,
            path=path,
            band_names=("band",),
            crs_wkt="PROJCRS[test]",
            nodata_values=(-32768.0,),
        )

    def test_written_raster_agrees_with_structured_record(self):
        import numpy as np
        from osgeo import gdal, osr

        from lfm.model.configured_tiler import ConfiguredTiler
        from lfm.model.lunar_crs import load_lunar_geographic_wkt
        from lfm.model.raster_cube import WarpedBand, write_tile_cube
        from lfm.model.tiling_config import TileConfig, TileSourceConfig

        output_dir = Path(tempfile.mkdtemp())
        source = TileSourceConfig(
            name="wac",
            data_dir=Path("/data/wac"),
            index_path=Path("/data/wac/index.shp"),
            selection_mode="product_id",
        )
        config = TileConfig(output_dir, 5, (source,))
        tiler = ConfiguredTiler(config, selectors={"wac": "M100"})

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(load_lunar_geographic_wkt())

        class TileDefinition:
            tileWidth = 2
            tileHeight = 2
            cellSize = 10.0
            srs = spatial_ref

        bands = [
            WarpedBand(
                name="vis_1",
                pixels=np.asarray([[1, 2], [3, -9999]], dtype=np.float32),
                source_nodata=-9999,
                output_nodata=-9999,
            )
        ]
        path = output_dir / "cube.tif"

        record = write_tile_cube(
            path,
            bands,
            source=source,
            product_id="M100",
            zone="42N",
            zoom_level=5,
            tile_x=1,
            tile_y=63,
            tile_def=TileDefinition(),
            ulx=100.0,
            uly=200.0,
        )

        dataset = gdal.Open(str(path))
        self.assertEqual(dataset.RasterCount, len(record.band_names))
        self.assertEqual(dataset.RasterXSize, 2)
        self.assertEqual(dataset.RasterYSize, 2)
        self.assertEqual(dataset.GetRasterBand(1).GetMetadataItem("Name"), "vis_1")
        self.assertEqual(dataset.GetRasterBand(1).GetNoDataValue(), -9999)
        self.assertEqual(record.path, path)
        self.assertEqual(record.nodata_values, (-9999,))

    @mock.patch("lfm.model.configured_tiler.TmsIntersector")
    def test_aoi_error_includes_records_from_earlier_tiles(
        self,
        intersector_cls,
    ):
        from lfm.model.configured_tiler import ConfiguredTiler
        from lfm.model.tiling_config import TileConfig, TileSourceConfig
        from lfm.model.tiling_results import MissingRequiredSourceError

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            context_source = TileSourceConfig(
                name="elevation",
                data_dir=Path("/data/elevation"),
                index_path=Path("/data/elevation/index.shp"),
            )
            required_source = TileSourceConfig(
                name="static",
                data_dir=Path("/data/static"),
                index_path=Path("/data/static/index.shp"),
            )
            tiler = ConfiguredTiler(
                TileConfig(root, 5, (context_source, required_source))
            )
            first = self.record(root / "first.tif", tile_x=1)
            within_failure = self.record(
                root / "second.tif",
                tile_x=2,
                source_name="elevation",
            )
            error = MissingRequiredSourceError(
                "missing static",
                source_name="static",
                zone="42N",
                tile_x=2,
                tile_y=63,
                completed_records=(within_failure,),
            )
            tiler.run_tile_index = mock.Mock(
                side_effect=(
                    [first],
                    error,
                    [self.record(root / "third.tif", tile_x=3)],
                )
            )
            intersector_cls.return_value.getTids.return_value = [
                {"zone": "42N", "tileX": 3, "tileY": 63},
                {"zone": "42N", "tileX": 1, "tileY": 63},
                {"zone": "42N", "tileX": 2, "tileY": 63},
            ]

            with self.assertRaises(MissingRequiredSourceError) as raised:
                tiler.run_aoi(1.3, 149.7, 1.1, 149.9)

        self.assertEqual(
            raised.exception.completed_records,
            (first, within_failure),
        )
        self.assertEqual(tiler.run_tile_index.call_count, 2)


if __name__ == "__main__":
    unittest.main()
