import importlib.util
import io
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.vector_index_builder import (
    StaleVectorIndexError,
    VectorIndexBuildConfig,
    VectorIndexValidationResult,
    create_vector_index,
    discover_raster_paths,
    ensure_vector_index,
)


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


class VectorIndexBuildConfigTestCase(unittest.TestCase):
    def test_accepts_shapefile_and_geopackage(self):
        shapefile = VectorIndexBuildConfig(
            data_dir=Path("/data/wac"),
            index_path=Path("/data/wac/index.shp"),
        )
        geopackage = VectorIndexBuildConfig(
            data_dir=Path("/data/nac"),
            index_path=Path("/data/nac/index.gpkg"),
            layer_name="nac",
        )

        self.assertEqual(shapefile.index_path.suffix, ".shp")
        self.assertEqual(geopackage.layer_name, "nac")

    def test_derives_default_index_path(self):
        config = VectorIndexBuildConfig(data_dir=Path("/data/wac"))

        self.assertEqual(config.index_path, Path("/data/wac/output_index.shp"))

    def test_rejects_unsupported_index_format(self):
        with self.assertRaisesRegex(ValueError, ".shp or .gpkg"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                index_path=Path("/data/wac/index.geojson"),
            )

    def test_rejects_empty_layer_name(self):
        with self.assertRaisesRegex(ValueError, "layer_name"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                layer_name="  ",
            )

    def test_discovers_rasters_in_deterministic_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "b.tif").touch()
            (data_dir / "a.tif").touch()
            (data_dir / "ignored.tiff").touch()
            (data_dir / "directory.tif").mkdir()

            result = discover_raster_paths(VectorIndexBuildConfig(data_dir))

        self.assertEqual([path.name for path in result], ["a.tif", "b.tif"])

    def test_discovery_requires_matching_rasters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = VectorIndexBuildConfig(Path(temporary_directory))
            with self.assertRaisesRegex(FileNotFoundError, "No rasters matched"):
                discover_raster_paths(config)

    def test_explicit_creation_never_overwrites_existing_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            config = VectorIndexBuildConfig(data_dir)
            config.index_path.touch()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_vector_index(config, progress=False)


class EnsureVectorIndexTestCase(unittest.TestCase):
    def result(self, config, paths):
        return VectorIndexValidationResult(
            index_path=config.index_path,
            driver_name="ESRI Shapefile",
            layer_name=config.index_path.stem,
            location_field=config.location_field,
            feature_count=len(paths),
            raster_paths=paths,
        )

    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    @mock.patch("lfm.model.vector_index_builder.create_vector_index")
    def test_creates_then_validates_missing_index(self, create, validate):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(data_dir)
            expected = self.result(config, (raster,))
            validate.return_value = expected

            def create_side_effect(
                received,
                *,
                raster_paths,
                progress,
                stdout,
            ):
                self.assertEqual(raster_paths, (raster,))
                self.assertTrue(progress)
                self.assertIs(stdout, active_stdout)
                received.index_path.touch()
                return received.index_path

            create.side_effect = create_side_effect
            active_stdout = io.StringIO()
            actual = ensure_vector_index(config, stdout=active_stdout)

        self.assertIs(actual, expected)
        create.assert_called_once()
        validate.assert_called_once_with(
            config,
            expected_raster_paths=(raster,),
        )
        output = active_stdout.getvalue()
        self.assertIn("will be created", output)
        self.assertIn("can take several minutes", output)
        self.assertIn("Created and validated", output)

    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    @mock.patch("lfm.model.vector_index_builder.create_vector_index")
    def test_reuses_existing_index_without_creation(self, create, validate):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(data_dir)
            config.index_path.touch()
            expected = self.result(config, (raster,))
            validate.return_value = expected
            stdout = io.StringIO()

            actual = ensure_vector_index(config, stdout=stdout)

        self.assertIs(actual, expected)
        create.assert_not_called()
        validate.assert_called_once_with(
            config,
            expected_raster_paths=(raster,),
        )
        self.assertIn("Reusing validated", stdout.getvalue())


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for vector-index tests")
class VectorIndexBuildIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        from osgeo import gdal

        from lfm.model.lunar_crs import load_lunar_geographic_wkt

        self.gdal = gdal
        self.gdal.UseExceptions()
        self.wkt = load_lunar_geographic_wkt()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_raster(self, name, *, x_origin):
        path = self.data_dir / name
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(path),
            2,
            2,
            1,
            self.gdal.GDT_Byte,
        )
        dataset.SetProjection(self.wkt)
        dataset.SetGeoTransform((x_origin, 0.1, 0.0, 1.0, 0.0, -0.1))
        dataset.GetRasterBand(1).Fill(1)
        dataset = None
        return path

    def test_ensure_creates_validates_and_reuses_each_supported_format(self):
        for suffix in (".shp", ".gpkg"):
            with self.subTest(suffix=suffix):
                directory = self.data_dir / suffix.removeprefix(".")
                directory.mkdir()
                original_data_dir = self.data_dir
                self.data_dir = directory
                try:
                    first = self.write_raster("a.tif", x_origin=10.0)
                    second = self.write_raster("b.tif", x_origin=11.0)
                    config = VectorIndexBuildConfig(
                        self.data_dir,
                        self.data_dir / f"index{suffix}",
                    )

                    created = ensure_vector_index(config, stdout=io.StringIO())
                    reused = ensure_vector_index(config, stdout=io.StringIO())
                finally:
                    self.data_dir = original_data_dir

                self.assertEqual(created.feature_count, 2)
                self.assertEqual(reused.feature_count, 2)
                self.assertEqual(
                    set(created.raster_paths),
                    {first, second},
                )

    def test_existing_index_rejects_changed_raster_inventory(self):
        self.write_raster("a.tif", x_origin=10.0)
        config = VectorIndexBuildConfig(self.data_dir)
        ensure_vector_index(config, stdout=io.StringIO())
        self.write_raster("new.tif", x_origin=11.0)

        with self.assertRaisesRegex(StaleVectorIndexError, "stale"):
            ensure_vector_index(config, stdout=io.StringIO())


if __name__ == "__main__":
    unittest.main()
