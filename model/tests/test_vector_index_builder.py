import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.vector_index_builder import (
    DEFAULT_RASTER_GLOBS,
    FOOTPRINT_EDGE_SAMPLES,
    StaleVectorIndexError,
    VectorIndexBuildConfig,
    VectorIndexLockError,
    VectorIndexValidationError,
    VectorIndexValidationResult,
    _canonical_geographic_footprint,
    _enclosed_geographic_pole,
    _index_lock_path,
    _perimeter_pixels,
    _raster_footprint,
    _spatial_references_equivalent,
    _unwrap_longitudes,
    create_vector_index,
    discover_raster_paths,
    ensure_vector_index,
)


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


class VectorIndexBuildConfigTestCase(unittest.TestCase):
    def test_geographic_longitude_winding_detects_enclosed_poles(self):
        class Ring:
            def __init__(self, coordinates):
                self.coordinates = coordinates

            def GetPointCount(self):
                return len(self.coordinates)

            def GetX(self, index):
                return self.coordinates[index][0]

            def GetY(self, index):
                return self.coordinates[index][1]

        class Polygon:
            def __init__(self, coordinates):
                self.ring = Ring(coordinates)

            def GetGeometryRef(self, index):
                self.asserted_index = index
                return self.ring

            def GetEnvelope(self):
                longitudes, latitudes = zip(*self.ring.coordinates, strict=True)
                return (
                    min(longitudes),
                    max(longitudes),
                    min(latitudes),
                    max(latitudes),
                )

        north = Polygon(
            (
                (0.0, 80.0),
                (90.0, 80.0),
                (179.0, 80.0),
                (-90.0, 80.0),
                (0.0, 80.0),
            )
        )
        south = Polygon(
            (
                (0.0, -80.0),
                (-90.0, -80.0),
                (-179.0, -80.0),
                (90.0, -80.0),
                (0.0, -80.0),
            )
        )
        seam_only = Polygon(
            (
                (170.0, 10.0),
                (-170.0, 10.0),
                (-170.0, 20.0),
                (170.0, 20.0),
                (170.0, 10.0),
            )
        )

        self.assertEqual(_enclosed_geographic_pole(north), 90.0)
        self.assertEqual(_enclosed_geographic_pole(south), -90.0)
        self.assertIsNone(_enclosed_geographic_pole(seam_only))

    def test_longitude_unwrapping_preserves_continuous_seam_footprint(self):
        self.assertEqual(
            _unwrap_longitudes(
                (
                    (170.0, 10.0),
                    (-170.0, 10.0),
                    (-170.0, 20.0),
                    (170.0, 20.0),
                    (170.0, 10.0),
                )
            ),
            (
                (170.0, 10.0),
                (190.0, 10.0),
                (190.0, 20.0),
                (170.0, 20.0),
                (170.0, 10.0),
            ),
        )

    def test_perimeter_pixels_densifies_every_edge_without_duplicate_corners(self):
        perimeter = _perimeter_pixels(10.0, 20.0, samples_per_edge=3)

        self.assertEqual(
            perimeter,
            (
                (0.0, 0.0),
                (5.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (10.0, 20.0),
                (5.0, 20.0),
                (0.0, 20.0),
                (0.0, 10.0),
            ),
        )

    def test_perimeter_pixels_rejects_invalid_sampling(self):
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            _perimeter_pixels(10.0, 20.0, samples_per_edge=True)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            _perimeter_pixels(10.0, 20.0, samples_per_edge=1)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _perimeter_pixels(0.0, 20.0)

    def test_shapefile_crs_fallback_uses_gdal_38_wkt_node_api(self):
        class GeographicSrs:
            def IsSame(self, other):
                return False

            def IsGeographic(self):
                return True

            def GetSemiMajor(self):
                return 1_737_400.0

            def GetSemiMinor(self):
                return 1_737_400.0

            def GetInvFlattening(self):
                return 0.0

            def GetAttrValue(self, node, child):
                self.assertion = (node, child)
                return "0"

            def GetAngularUnits(self):
                return 0.0174532925199433

        actual = GeographicSrs()
        expected = GeographicSrs()

        self.assertTrue(
            _spatial_references_equivalent(
                actual,
                expected,
                index_suffix=".shp",
            )
        )
        self.assertEqual(actual.assertion, ("PRIMEM", 1))
        self.assertEqual(expected.assertion, ("PRIMEM", 1))

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

    def test_invalid_rebuild_is_explicit_and_geopackage_only(self):
        config = VectorIndexBuildConfig(
            data_dir=Path("/data/wac"),
            index_path=Path("/cache/wac.gpkg"),
            rebuild_invalid_index=True,
        )

        self.assertTrue(config.rebuild_invalid_index)
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                index_path=Path("/cache/wac.gpkg"),
                rebuild_invalid_index="yes",
            )
        with self.assertRaisesRegex(ValueError, "application-owned GeoPackage"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                index_path=Path("/data/wac/output_index.shp"),
                rebuild_invalid_index=True,
            )

    def test_derives_default_index_path(self):
        config = VectorIndexBuildConfig(data_dir=Path("/data/wac"))

        self.assertEqual(config.index_path, Path("/data/wac/output_index.shp"))
        self.assertEqual(config.raster_globs, DEFAULT_RASTER_GLOBS)

    def test_legacy_image_glob_remains_a_single_pattern_override(self):
        config = VectorIndexBuildConfig(
            data_dir=Path("/data/wac"),
            image_glob="*.cog.tif",
        )

        self.assertEqual(config.image_glob, "*.cog.tif")
        self.assertEqual(config.raster_globs, ("*.cog.tif",))

    def test_rejects_ambiguous_single_and_multiple_glob_overrides(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            VectorIndexBuildConfig(
                data_dir=Path("/data/wac"),
                image_glob="*.tif",
                image_globs=("*.nc", "*.vrt"),
            )

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
            (data_dir / "c.tiff").touch()
            (data_dir / "d.nc").touch()
            (data_dir / "e.vrt").touch()
            (data_dir / "ignored.img").touch()
            (data_dir / "directory.tif").mkdir()

            result = discover_raster_paths(VectorIndexBuildConfig(data_dir))

        self.assertEqual(
            [path.name for path in result],
            ["a.tif", "b.tif", "c.tiff", "d.nc", "e.vrt"],
        )

    def test_discovery_deduplicates_overlapping_patterns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(
                data_dir,
                image_globs=("*.tif", "a.*"),
            )

            result = discover_raster_paths(config)

        self.assertEqual(result, (raster,))

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

    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    @mock.patch("lfm.model.vector_index_builder._create_vector_index_with_ogr")
    def test_creation_validates_staging_then_publishes_all_sidecars(
        self,
        write_index,
        validate,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(data_dir)

            def write_side_effect(received, paths, *, progress, stdout):
                self.assertEqual(paths, (raster,))
                self.assertFalse(progress)
                self.assertIsInstance(stdout, io.StringIO)
                for suffix in (".shp", ".shx", ".dbf", ".prj"):
                    received.index_path.with_suffix(suffix).touch()
                return received.index_path

            write_index.side_effect = write_side_effect
            output = io.StringIO()
            result = create_vector_index(
                config,
                progress=False,
                stdout=output,
            )

            self.assertEqual(result, config.index_path)
            for suffix in (".shp", ".shx", ".dbf", ".prj"):
                self.assertTrue(config.index_path.with_suffix(suffix).is_file())
            self.assertFalse(_index_lock_path(config.index_path).exists())
            self.assertEqual(
                list(data_dir.glob(f".{config.index_path.name}.staging-*")),
                [],
            )
            staged_config = validate.call_args.args[0]
            self.assertNotEqual(staged_config.index_path, config.index_path)
            validate.assert_called_once_with(
                staged_config,
                expected_raster_paths=(raster,),
            )

    @mock.patch("lfm.model.vector_index_builder._create_vector_index_with_ogr")
    def test_creation_failure_cleans_staging_and_lock(self, write_index):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(data_dir)

            def write_side_effect(received, paths, *, progress, stdout):
                del paths, progress, stdout
                received.index_path.touch()
                raise RuntimeError("synthetic creation failure")

            write_index.side_effect = write_side_effect
            with self.assertRaisesRegex(RuntimeError, "synthetic creation failure"):
                create_vector_index(config, progress=False)

            self.assertFalse(config.index_path.exists())
            self.assertFalse(_index_lock_path(config.index_path).exists())
            self.assertEqual(
                list(data_dir.glob(f".{config.index_path.name}.staging-*")),
                [],
            )

    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    @mock.patch("lfm.model.vector_index_builder._create_vector_index_with_ogr")
    def test_staging_validation_failure_publishes_nothing(
        self,
        write_index,
        validate,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            raster = data_dir / "a.tif"
            raster.touch()
            config = VectorIndexBuildConfig(data_dir)

            def write_side_effect(received, paths, *, progress, stdout):
                del paths, progress, stdout
                for suffix in (".shp", ".shx", ".dbf"):
                    received.index_path.with_suffix(suffix).touch()
                return received.index_path

            write_index.side_effect = write_side_effect
            validate.side_effect = RuntimeError("synthetic validation failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic validation failure"):
                create_vector_index(config, progress=False)

            for suffix in (".shp", ".shx", ".dbf"):
                self.assertFalse(config.index_path.with_suffix(suffix).exists())
            self.assertFalse(_index_lock_path(config.index_path).exists())
            self.assertEqual(
                list(data_dir.glob(f".{config.index_path.name}.staging-*")),
                [],
            )

    def test_concurrent_creation_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "a.tif").touch()
            config = VectorIndexBuildConfig(data_dir)
            lock_path = _index_lock_path(config.index_path)
            lock_path.touch()

            with self.assertRaisesRegex(VectorIndexLockError, "already in progress"):
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
            logger = mock.Mock()
            actual = ensure_vector_index(
                config,
                logger=logger,
                stdout=active_stdout,
            )

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
        logged = [call.args[0] for call in logger.info.call_args_list]
        self.assertTrue(any("Found 1 raster" in message for message in logged))
        self.assertTrue(any("will be created" in message for message in logged))
        self.assertTrue(any("Created and validated" in message for message in logged))

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

    @mock.patch("lfm.model.vector_index_builder._rebuild_invalid_geopackage")
    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    def test_managed_cache_rebuilds_after_validation_failure(
        self,
        validate,
        rebuild,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raster = root / "a.tif"
            raster.touch()
            index_path = root / "cache.gpkg"
            index_path.touch()
            config = VectorIndexBuildConfig(
                root,
                index_path=index_path,
                rebuild_invalid_index=True,
            )
            expected = self.result(config, (raster,))
            validate.side_effect = VectorIndexValidationError("bad geometry")
            rebuild.return_value = (expected, True)
            stdout = io.StringIO()

            actual = ensure_vector_index(config, stdout=stdout)

        self.assertIs(actual, expected)
        rebuild.assert_called_once_with(
            config,
            raster_paths=(raster,),
            stdout=stdout,
        )
        self.assertIn("invalid or stale", stdout.getvalue())
        self.assertIn("Rebuilt raster index", stdout.getvalue())

    @mock.patch("lfm.model.vector_index_builder._rebuild_invalid_geopackage")
    @mock.patch("lfm.model.vector_index_builder.validate_vector_index")
    def test_unmanaged_invalid_index_is_never_rebuilt(self, validate, rebuild):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "a.tif").touch()
            config = VectorIndexBuildConfig(root)
            config.index_path.touch()
            validate.side_effect = VectorIndexValidationError("bad geometry")

            with self.assertRaisesRegex(VectorIndexValidationError, "bad geometry"):
                ensure_vector_index(config, stdout=io.StringIO())

        rebuild.assert_not_called()


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

    def test_antimeridian_footprint_is_split_without_global_false_coverage(self):
        from osgeo import ogr

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for longitude, latitude in (
            (170.0, 10.0),
            (-170.0, 10.0),
            (-170.0, 20.0),
            (170.0, 20.0),
            (170.0, 10.0),
        ):
            ring.AddPoint_2D(longitude, latitude)
        wrapped = ogr.Geometry(ogr.wkbPolygon)
        wrapped.AddGeometry(ring)

        footprint = _canonical_geographic_footprint(wrapped, ogr=ogr)

        self.assertEqual(footprint.GetGeometryName().upper(), "MULTIPOLYGON")
        self.assertEqual(footprint.GetGeometryCount(), 2)
        self.assertTrue(footprint.IsValid())
        for longitude, expected in ((-175.0, True), (0.0, False), (175.0, True)):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint_2D(longitude, 15.0)
            self.assertEqual(footprint.Intersects(point), expected)

    def test_global_geographic_raster_uses_valid_full_longitude_band(self):
        from osgeo import ogr, osr

        raster_path = self.data_dir / "global.tif"
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(raster_path),
            360,
            180,
            1,
            self.gdal.GDT_Byte,
        )
        dataset.SetProjection(self.wkt)
        dataset.SetGeoTransform((-180.0, 1.0, 0.0, 90.0, 0.0, -1.0))
        dataset.GetRasterBand(1).Fill(1)
        dataset = None

        output_srs = osr.SpatialReference()
        self.assertEqual(output_srs.ImportFromWkt(self.wkt), 0)
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        footprint = _raster_footprint(
            raster_path,
            output_srs=output_srs,
            gdal=self.gdal,
            ogr=ogr,
            osr=osr,
        )

        self.assertTrue(footprint.IsValid())
        self.assertEqual(footprint.GetEnvelope(), (-180.0, 180.0, -90.0, 90.0))
        for longitude in (-179.0, 0.0, 179.0):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint_2D(longitude, 0.0)
            self.assertTrue(footprint.Intersects(point))

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

                    creation_stdout = io.StringIO()
                    created = ensure_vector_index(config, stdout=creation_stdout)
                    index_artifacts = (
                        (config.index_path,)
                        if suffix == ".gpkg"
                        else tuple(
                            sorted(
                                path
                                for path in self.data_dir.glob("index.*")
                                if path.is_file()
                            )
                        )
                    )
                    before_reuse = {
                        path: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in index_artifacts
                    }
                    reused = ensure_vector_index(config, stdout=io.StringIO())
                    after_reuse = {
                        path: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in index_artifacts
                    }
                finally:
                    self.data_dir = original_data_dir

                self.assertEqual(created.feature_count, 2)
                self.assertEqual(reused.feature_count, 2)
                self.assertEqual(
                    set(created.raster_paths),
                    {first, second},
                )
                self.assertIn("Building raster index", creation_stdout.getvalue())
                self.assertEqual(before_reuse, after_reuse)

    def test_managed_geopackage_atomically_replaces_invalid_cache(self):
        raster = self.write_raster("a.tif", x_origin=10.0)
        index_path = self.data_dir / "managed.gpkg"
        index_path.write_bytes(b"not a GeoPackage")
        config = VectorIndexBuildConfig(
            self.data_dir,
            index_path=index_path,
            rebuild_invalid_index=True,
        )
        stdout = io.StringIO()

        result = ensure_vector_index(config, stdout=stdout)
        reused = ensure_vector_index(config, stdout=io.StringIO())

        self.assertEqual(result, reused)
        self.assertEqual(result.raster_paths, (raster,))
        self.assertIn("invalid or stale", stdout.getvalue())
        self.assertIn("Rebuilt raster index", stdout.getvalue())
        self.assertEqual(
            list(self.data_dir.glob(".managed.gpkg.replacement-*")),
            [],
        )

    def test_existing_index_rejects_changed_raster_inventory(self):
        self.write_raster("a.tif", x_origin=10.0)
        config = VectorIndexBuildConfig(self.data_dir)
        ensure_vector_index(config, stdout=io.StringIO())
        self.write_raster("new.tif", x_origin=11.0)

        with self.assertRaisesRegex(StaleVectorIndexError, "stale"):
            ensure_vector_index(config, stdout=io.StringIO())

    def test_vrt_input_is_created_validated_and_reused(self):
        component_dir = self.data_dir / "components"
        component_dir.mkdir()
        original_data_dir = self.data_dir
        self.data_dir = component_dir
        try:
            source = self.write_raster("source.tif", x_origin=10.0)
        finally:
            self.data_dir = original_data_dir
        vrt_path = self.data_dir / "mosaic.vrt"
        translated = self.gdal.Translate(str(vrt_path), str(source), format="VRT")
        self.assertIsNotNone(translated)
        translated = None
        config = VectorIndexBuildConfig(
            self.data_dir,
            self.data_dir / "vrt_index.gpkg",
            image_globs=("*.vrt",),
        )

        created = ensure_vector_index(config, stdout=io.StringIO())
        reused = ensure_vector_index(config, stdout=io.StringIO())

        self.assertEqual(created.raster_paths, (vrt_path,))
        self.assertEqual(created, reused)

    def test_netcdf_input_is_created_validated_and_reused(self):
        netcdf_driver = self.gdal.GetDriverByName("netCDF")
        if netcdf_driver is None:
            self.skipTest("GDAL netCDF driver is unavailable")
        component_dir = self.data_dir / "components"
        component_dir.mkdir()
        original_data_dir = self.data_dir
        self.data_dir = component_dir
        try:
            source = self.write_raster("source.tif", x_origin=10.0)
        finally:
            self.data_dir = original_data_dir
        source_dataset = self.gdal.Open(str(source), self.gdal.GA_ReadOnly)
        netcdf_path = self.data_dir / "source.nc"
        copied = netcdf_driver.CreateCopy(str(netcdf_path), source_dataset)
        self.assertIsNotNone(copied)
        copied = None
        source_dataset = None
        config = VectorIndexBuildConfig(
            self.data_dir,
            self.data_dir / "netcdf_index.gpkg",
            image_globs=("*.nc",),
        )

        created = ensure_vector_index(config, stdout=io.StringIO())
        reused = ensure_vector_index(config, stdout=io.StringIO())

        self.assertEqual(created.raster_paths, (netcdf_path,))
        self.assertEqual(created, reused)

    def test_explicitly_archived_stale_index_can_be_rebuilt(self):
        first = self.write_raster("a.tif", x_origin=10.0)
        config = VectorIndexBuildConfig(
            self.data_dir,
            self.data_dir / "output_index.gpkg",
        )
        initial = ensure_vector_index(config, stdout=io.StringIO())
        second = self.write_raster("b.tif", x_origin=11.0)
        with self.assertRaises(StaleVectorIndexError):
            ensure_vector_index(config, stdout=io.StringIO())

        archived_path = self.data_dir / "archived_stale_index.gpkg"
        config.index_path.rename(archived_path)
        rebuilt = ensure_vector_index(config, stdout=io.StringIO())

        self.assertTrue(archived_path.is_file())
        self.assertTrue(config.index_path.is_file())
        self.assertEqual(initial.feature_count, 1)
        self.assertEqual(rebuilt.feature_count, 2)
        self.assertEqual(set(rebuilt.raster_paths), {first, second})

    def test_polar_raster_footprint_preserves_curved_densified_edge(self):
        from osgeo import ogr, osr

        polar_definition_path = (
            Path(__file__).resolve().parents[2] / "TMS" / "RG" / "tms_LPS_NRG.json"
        )
        polar_definition = json.loads(
            polar_definition_path.read_text(encoding="utf-8")
        )
        raster_path = self.data_dir / "polar.tif"
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(raster_path),
            100,
            100,
            1,
            self.gdal.GDT_Byte,
        )
        dataset.SetProjection(polar_definition["crs"])
        dataset.SetGeoTransform((550_000.0, 1_000.0, 0.0, 650_000.0, 0.0, -1_000.0))
        dataset.GetRasterBand(1).Fill(1)
        dataset = None

        output_srs = osr.SpatialReference()
        self.assertEqual(output_srs.ImportFromWkt(self.wkt), 0)
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        footprint = _raster_footprint(
            raster_path,
            output_srs=output_srs,
            gdal=self.gdal,
            ogr=ogr,
            osr=osr,
        )
        dense_reference = _raster_footprint(
            raster_path,
            output_srs=output_srs,
            gdal=self.gdal,
            ogr=ogr,
            osr=osr,
            samples_per_edge=201,
        )
        ring = footprint.GetGeometryRef(0)
        dense_ring = dense_reference.GetGeometryRef(0)

        self.assertEqual(
            ring.GetPointCount(),
            4 * FOOTPRINT_EDGE_SAMPLES - 3,
        )
        self.assertEqual(dense_ring.GetPointCount(), 4 * 201 - 3)
        relative_area_difference = (
            abs(footprint.GetArea() - dense_reference.GetArea())
            / dense_reference.GetArea()
        )
        self.assertLessEqual(relative_area_difference, 0.001)
        first = ring.GetPoint(0)
        midpoint = ring.GetPoint((FOOTPRINT_EDGE_SAMPLES - 1) // 2)
        last = ring.GetPoint(FOOTPRINT_EDGE_SAMPLES - 1)
        linear_midpoint_latitude = (first[1] + last[1]) / 2.0
        self.assertGreater(
            abs(midpoint[1] - linear_midpoint_latitude),
            1e-6,
        )

    def test_pole_containing_rasters_use_valid_full_longitude_caps(self):
        from osgeo import ogr, osr

        output_srs = osr.SpatialReference()
        self.assertEqual(output_srs.ImportFromWkt(self.wkt), 0)
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        for grid_id, pole_latitude in (("LPS_N", 90.0), ("LPS_S", -90.0)):
            with self.subTest(grid_id=grid_id):
                definition_path = (
                    Path(__file__).resolve().parents[2]
                    / "TMS"
                    / "RG"
                    / f"tms_{grid_id}RG.json"
                )
                definition = json.loads(definition_path.read_text(encoding="utf-8"))
                raster_path = self.data_dir / f"{grid_id}.tif"
                dataset = self.gdal.GetDriverByName("GTiff").Create(
                    str(raster_path),
                    100,
                    100,
                    1,
                    self.gdal.GDT_Byte,
                )
                dataset.SetProjection(definition["crs"])
                dataset.SetGeoTransform(
                    (400_000.0, 2_000.0, 0.0, 600_000.0, 0.0, -2_000.0)
                )
                dataset.GetRasterBand(1).Fill(1)
                dataset = None

                footprint = _raster_footprint(
                    raster_path,
                    output_srs=output_srs,
                    gdal=self.gdal,
                    ogr=ogr,
                    osr=osr,
                )
                minimum_x, maximum_x, minimum_y, maximum_y = (
                    footprint.GetEnvelope()
                )

                self.assertTrue(footprint.IsValid())
                self.assertEqual((minimum_x, maximum_x), (-180.0, 180.0))
                if pole_latitude > 0.0:
                    self.assertEqual(maximum_y, 90.0)
                    self.assertLess(minimum_y, 90.0)
                    query_latitude = (minimum_y + 90.0) / 2.0
                else:
                    self.assertEqual(minimum_y, -90.0)
                    self.assertGreater(maximum_y, -90.0)
                    query_latitude = (maximum_y - 90.0) / 2.0
                for query_longitude in (-179.0, 0.0, 179.0):
                    query_point = ogr.Geometry(ogr.wkbPoint)
                    query_point.AddPoint_2D(query_longitude, query_latitude)
                    self.assertTrue(footprint.Intersects(query_point))


if __name__ == "__main__":
    unittest.main()
