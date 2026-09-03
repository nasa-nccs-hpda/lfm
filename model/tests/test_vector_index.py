from pathlib import Path
import importlib.util
import tempfile
import unittest

from lfm.model.tiling_config import TileSourceConfig
from lfm.model.vector_index import (
    query_source_index,
    resolve_indexed_raster_path,
)


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


class RasterPathResolutionTestCase(unittest.TestCase):
    def test_absolute_path_is_preserved(self):
        result = resolve_indexed_raster_path(
            Path("/data/source"),
            "/archive/raster.tif",
        )

        self.assertEqual(result, Path("/archive/raster.tif"))

    def test_relative_path_uses_data_directory(self):
        result = resolve_indexed_raster_path(
            Path("/data/source"),
            "tiles/raster.tif",
        )

        self.assertEqual(result, Path("/data/source/tiles/raster.tif"))

    def test_empty_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty raster path"):
            resolve_indexed_raster_path(Path("/data/source"), "  ")


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for vector-index tests")
class VectorIndexIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        from osgeo import ogr

        self.ogr = ogr
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "rasters"
        self.data_dir.mkdir()

    def _write_index(self, suffix: str, *, layer_name: str):
        driver_name = "ESRI Shapefile" if suffix == ".shp" else "GPKG"
        driver = self.ogr.GetDriverByName(driver_name)
        index_path = self.temp_dir / f"index{suffix}"
        dataset = driver.CreateDataSource(str(index_path))
        layer = dataset.CreateLayer(layer_name, geom_type=self.ogr.wkbPolygon)
        layer.CreateField(self.ogr.FieldDefn("raster", self.ogr.OFTString))

        for fid, (name, bounds) in enumerate(
            [
                ("inside.tif", (10.0, 0.0, 11.0, 1.0)),
                ("outside.tif", (30.0, 20.0, 31.0, 21.0)),
            ]
        ):
            min_x, min_y, max_x, max_y = bounds
            ring = self.ogr.Geometry(self.ogr.wkbLinearRing)
            ring.AddPoint(min_x, min_y)
            ring.AddPoint(max_x, min_y)
            ring.AddPoint(max_x, max_y)
            ring.AddPoint(min_x, max_y)
            ring.AddPoint(min_x, min_y)
            polygon = self.ogr.Geometry(self.ogr.wkbPolygon)
            polygon.AddGeometry(ring)
            feature = self.ogr.Feature(layer.GetLayerDefn())
            feature.SetFID(fid)
            feature.SetField("raster", name)
            feature.SetGeometry(polygon)
            layer.CreateFeature(feature)
        layer = None
        dataset = None
        return index_path

    def _query(self, index_path: Path, *, layer_name: str | None):
        source = TileSourceConfig(
            name="test",
            data_dir=self.data_dir,
            index_path=index_path,
            index_layer=layer_name,
            location_field="raster",
        )
        return query_source_index(
            source,
            ul_lat=2.0,
            ul_lon=9.0,
            lr_lat=-1.0,
            lr_lon=12.0,
        )

    def test_shapefile_query(self):
        index_path = self._write_index(".shp", layer_name="index")

        records = self._query(index_path, layer_name=None)

        self.assertEqual([record.path for record in records], [self.data_dir / "inside.tif"])

    def test_geopackage_named_layer_query(self):
        index_path = self._write_index(".gpkg", layer_name="raster_index")

        records = self._query(index_path, layer_name="raster_index")

        self.assertEqual([record.path for record in records], [self.data_dir / "inside.tif"])


if __name__ == "__main__":
    unittest.main()
