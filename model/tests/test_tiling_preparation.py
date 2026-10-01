import importlib.util
import io
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.lunar_crs import load_lunar_geographic_wkt
from lfm.model.tiling_config import TileSourceConfig
from lfm.model.tiling_preparation import (
    TileSourcePreparation,
    prepare_tile_config,
)
from lfm.model.vector_index_builder import VectorIndexValidationResult


HAS_OSGEO = importlib.util.find_spec("osgeo") is not None


class TilePreparationTestCase(unittest.TestCase):
    def source(self, name: str, *, index_layer: str | None = None):
        data_dir = Path(f"/data/{name}")
        return TileSourceConfig(
            name=name,
            data_dir=data_dir,
            index_path=data_dir / "output_index.gpkg",
            index_layer=index_layer,
            location_field="raster_path",
        )

    def validation_result(self, source: TileSourceConfig):
        return VectorIndexValidationResult(
            index_path=source.index_path,
            driver_name="GPKG",
            layer_name=source.index_layer or source.index_path.stem,
            location_field=source.location_field,
            feature_count=1,
            raster_paths=(source.data_dir / "a.tif",),
        )

    def test_preparation_derives_builder_contract_from_source(self):
        source = self.source("wac", index_layer="wac_index")
        preparation = TileSourcePreparation(
            source,
            image_glob="*.cog.tif",
        )

        build = preparation.index_config()

        self.assertEqual(build.data_dir, source.data_dir)
        self.assertEqual(build.index_path, source.index_path)
        self.assertEqual(build.layer_name, source.index_layer)
        self.assertEqual(build.location_field, source.location_field)
        self.assertEqual(build.image_glob, "*.cog.tif")

    @mock.patch("lfm.model.tiling_preparation.ensure_vector_index")
    def test_enabled_sources_are_prepared_before_config_assembly(self, ensure):
        wac = self.source("wac")
        static = self.source("static")
        expected_indexes = (
            self.validation_result(wac),
            self.validation_result(static),
        )
        ensure.side_effect = expected_indexes
        stdout = io.StringIO()

        result = prepare_tile_config(
            output_dir="/output",
            zoom_level=5,
            sources=(
                TileSourcePreparation(wac),
                TileSourcePreparation(static),
            ),
            stdout=stdout,
        )

        self.assertEqual(result.config.sources, (wac, static))
        self.assertEqual(result.indexes, expected_indexes)
        self.assertEqual(ensure.call_count, 2)
        for call, source in zip(ensure.call_args_list, (wac, static)):
            build = call.args[0]
            self.assertEqual(build.index_path, source.index_path)
            self.assertIs(call.kwargs["stdout"], stdout)

    @mock.patch("lfm.model.tiling_preparation.ensure_vector_index")
    def test_disabled_source_is_not_prepared_or_returned(self, ensure):
        wac = self.source("wac")
        static = self.source("static")
        expected = self.validation_result(wac)
        ensure.return_value = expected

        result = prepare_tile_config(
            output_dir="/output",
            zoom_level=5,
            sources=(
                TileSourcePreparation(wac),
                TileSourcePreparation(static, enabled=False),
            ),
        )

        self.assertEqual(result.config.sources, (wac,))
        self.assertEqual(result.indexes, (expected,))
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[0].index_path, wac.index_path)

    @mock.patch("lfm.model.tiling_preparation.ensure_vector_index")
    def test_no_enabled_sources_fails_before_index_preparation(self, ensure):
        with self.assertRaisesRegex(ValueError, "At least one"):
            prepare_tile_config(
                output_dir="/output",
                zoom_level=5,
                sources=(
                    TileSourcePreparation(self.source("static"), enabled=False),
                ),
            )

        ensure.assert_not_called()

    @mock.patch("lfm.model.tiling_preparation.ensure_vector_index")
    def test_duplicate_enabled_names_fail_before_index_preparation(self, ensure):
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_tile_config(
                output_dir="/output",
                zoom_level=5,
                sources=(
                    TileSourcePreparation(self.source("wac")),
                    TileSourcePreparation(self.source("wac")),
                ),
            )

        ensure.assert_not_called()


@unittest.skipUnless(HAS_OSGEO, "GDAL/OGR is required for preparation integration")
class TilePreparationIntegrationTestCase(unittest.TestCase):
    def test_missing_enabled_index_is_created_and_disabled_source_is_untouched(self):
        from osgeo import gdal

        gdal.UseExceptions()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enabled_dir = root / "enabled"
            enabled_dir.mkdir()
            raster_path = enabled_dir / "a.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(
                str(raster_path),
                2,
                2,
                1,
                gdal.GDT_Byte,
            )
            dataset.SetProjection(load_lunar_geographic_wkt())
            dataset.SetGeoTransform((10.0, 0.1, 0.0, 1.0, 0.0, -0.1))
            dataset.GetRasterBand(1).Fill(1)
            dataset = None

            enabled = TileSourceConfig(
                name="wac",
                data_dir=enabled_dir,
                index_path=enabled_dir / "output_index.gpkg",
            )
            disabled_dir = root / "disabled_missing"
            disabled = TileSourceConfig(
                name="static",
                data_dir=disabled_dir,
                index_path=disabled_dir / "output_index.gpkg",
            )

            result = prepare_tile_config(
                output_dir=root / "output",
                zoom_level=5,
                sources=(
                    TileSourcePreparation(enabled),
                    TileSourcePreparation(disabled, enabled=False),
                ),
                stdout=io.StringIO(),
            )

            self.assertEqual(result.config.sources, (enabled,))
            self.assertEqual(len(result.indexes), 1)
            self.assertTrue(enabled.index_path.is_file())
            self.assertFalse(disabled.index_path.exists())
            self.assertFalse(disabled_dir.exists())


if __name__ == "__main__":
    unittest.main()
