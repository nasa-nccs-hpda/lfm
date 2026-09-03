from pathlib import Path
import importlib.util
import tempfile
import unittest


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for configured tiler tests")
class ConfiguredTilerIntegrationTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
