from pathlib import Path
import importlib.util
import unittest
from unittest import mock


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for raster cube tests")
class RasterCubeTestCase(unittest.TestCase):
    def test_explicit_multiband_nodata_preserves_independent_band_masks(self):
        import numpy as np
        from osgeo import gdal

        from lfm.model.raster_cube import warp_source_to_tile
        from lfm.model.tiling_config import TileSourceConfig

        class SourceBand:
            def __init__(self, name):
                self.name = name

            def GetMetadataItem(self, key):
                return self.name if key == "Name" else None

            def GetDescription(self):
                return self.name

            def GetNoDataValue(self):
                return -9999.0

        class SourceDataset:
            RasterCount = 2

            def __init__(self):
                self.bands = (SourceBand("first"), SourceBand("second"))

            def GetRasterBand(self, index):
                return self.bands[index - 1]

        class WarpedDataset:
            def ReadAsArray(self):
                return np.asarray(
                    [
                        [[1.0, -9999.0], [2.0, -9999.0]],
                        [[-9999.0, 3.0], [-9999.0, 4.0]],
                    ],
                    dtype=np.float32,
                )

        class TileDefinition:
            tileWidth = 2
            tileHeight = 2
            srs = object()

        source = TileSourceConfig(
            name="wac",
            data_dir=Path("/data/wac"),
            index_path=Path("/data/wac/index.shp"),
            selection_mode="product_id",
            preserve_source_nodata=True,
        )

        with (
            mock.patch(
                "lfm.model.raster_cube.gdal.Open",
                return_value=SourceDataset(),
            ),
            mock.patch(
                "lfm.model.raster_cube.gdal.Warp",
                return_value=WarpedDataset(),
            ) as warp,
        ):
            bands = warp_source_to_tile(
                source,
                [Path("/data/wac/product.tif")],
                tile_def=TileDefinition(),
                bounds=(0.0, 2.0, 2.0, 0.0),
            )

        self.assertEqual([band.name for band in bands], ["first", "second"])
        warp_kwargs = warp.call_args.kwargs
        self.assertEqual(warp_kwargs["resampleAlg"], gdal.GRA_Bilinear)
        self.assertEqual(warp_kwargs["srcNodata"], [-9999.0, -9999.0])
        self.assertEqual(warp_kwargs["dstNodata"], [-9999.0, -9999.0])
        self.assertEqual(
            warp_kwargs["warpOptions"],
            ["UNIFIED_SRC_NODATA=PARTIAL"],
        )


if __name__ == "__main__":
    unittest.main()
