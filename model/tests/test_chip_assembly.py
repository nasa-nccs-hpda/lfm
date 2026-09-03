from pathlib import Path
import importlib.util
import tempfile
import unittest

from lfm.model.chip_acquisition import ChipAcquisitionResult
from lfm.model.chip_assembly import (
    ChipAssemblyError,
    assemble_chip,
    staged_chip_path,
    write_model_ready_chip,
)
from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitPercentages,
)
from lfm.model.chip_preflight import PreparedChipRequest
from lfm.model.chip_reprojection import (
    ChipReprojectionResult,
    ReprojectedModality,
)
from lfm.model.chip_splits import SplitAssignment
from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    GeographicAOI,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig


HAS_GDAL_NUMPY = (
    importlib.util.find_spec("osgeo") is not None
    and importlib.util.find_spec("numpy") is not None
)


class ChipAssemblyPathTestCase(unittest.TestCase):
    def test_staged_path_uses_sample_id_and_terminal_suffix(self):
        root = Path("/tmp/chip-assembly-path")
        source = TileSourceConfig(
            "wac",
            root / "wac",
            root / "wac/index.gpkg",
        )
        config = ChipConfig(
            output_root=root / "output",
            label_source=root / "labels",
            acquisition_groups=(
                AcquisitionGroupConfig(
                    "coarse",
                    TileConfig(root / "tiles", 5, (source,)),
                ),
            ),
            output_modalities=(
                OutputModalityConfig("coarse", "wac", "wac"),
            ),
            split_config=SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
        )

        path = staged_chip_path(config, "M123_r40_c80")

        self.assertEqual(
            path,
            root
            / "output/.intermediate/M123_r40_c80/assembled"
            / "M123_r40_c80_input_wac_chip.tif",
        )

    def test_staged_path_rejects_unsafe_sample_id(self):
        root = Path("/tmp/chip-assembly-path")
        source = TileSourceConfig(
            "wac",
            root / "wac",
            root / "wac/index.gpkg",
        )
        config = ChipConfig(
            output_root=root / "output",
            label_source=root / "labels",
            acquisition_groups=(
                AcquisitionGroupConfig(
                    "coarse",
                    TileConfig(root / "tiles", 5, (source,)),
                ),
            ),
            output_modalities=(
                OutputModalityConfig("coarse", "wac", "wac"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "dataset-compatible"):
            staged_chip_path(config, "../unsafe")


@unittest.skipUnless(HAS_GDAL_NUMPY, "GDAL and NumPy are required")
class ChipAssemblyRasterTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import osr

        self.np = np
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(4326)
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.grid = TargetGrid(
            crs_wkt=spatial_reference.ExportToWkt(),
            transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            bounds=(0.0, 0.0, 3.0, 2.0),
            width=3,
            height=2,
        )

    def config(
        self,
        root,
        modalities,
        *,
        optional=(),
        output_dtype="float32",
        output_suffix=None,
    ):
        source_names = tuple(dict.fromkeys(item.source_name for item in modalities))
        sources = tuple(
            TileSourceConfig(
                name,
                root / name,
                root / name / "index.gpkg",
                required=name not in optional,
            )
            for name in source_names
        )
        return ChipConfig(
            output_root=root / "output",
            label_source=root / "labels",
            acquisition_groups=(
                AcquisitionGroupConfig(
                    "coarse",
                    TileConfig(root / "tiles", 5, sources),
                ),
            ),
            output_modalities=tuple(modalities),
            split_config=SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
            output_dtype=output_dtype,
            output_suffix=output_suffix,
        )

    def reprojection(
        self,
        config,
        specifications,
        *,
        preflight_status="passed",
        assigned_split="train",
    ):
        request = ChipRequest(
            sample_id="M123_r40_c80",
            target_grid=self.grid,
            geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 3.0),
            split_group_key="M123",
        )
        prepared = PreparedChipRequest(
            request=request,
            assignment=SplitAssignment(
                sample_id=request.sample_id,
                split_group_key=request.split_group_key,
                assigned_split=assigned_split,
                source=(
                    "automatic_percentage"
                    if assigned_split is not None
                    else "unassigned"
                ),
            ),
            preflight=ChipPreflight(
                status=preflight_status,
                assigned_split=assigned_split,
                resolved_label_path=(
                    Path("/labels/M123_r40_c80_label.npy")
                    if preflight_status == "passed"
                    else None
                ),
            ),
        )
        acquisition = ChipAcquisitionResult(
            prepared_request=prepared,
            status="complete",
        )
        results = []
        for modality, specification in zip(
            config.output_modalities, specifications, strict=True
        ):
            names, pixels, mask, status = specification
            results.append(
                ReprojectedModality(
                    acquisition_group=modality.acquisition_group,
                    source_name=modality.source_name,
                    alias=modality.alias,
                    resampling=modality.resampling,
                    status=status,
                    target_grid=self.grid,
                    band_names=tuple(names),
                    pixels=pixels,
                    valid_mask=mask,
                    nodata_values=tuple(-32768.0 for _ in names),
                    zone_groups=(),
                )
            )
        return ChipReprojectionResult(
            acquisition=acquisition,
            target_grid=self.grid,
            modalities=tuple(results),
        )

    def values(self, constants):
        return self.np.stack(
            tuple(
                self.np.full((self.grid.height, self.grid.width), value)
                for value in constants
            )
        )

    def test_selects_bands_and_concatenates_in_config_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modalities = (
                OutputModalityConfig(
                    "coarse",
                    "static",
                    "static",
                    band_names=("VIS", "elevation"),
                ),
                OutputModalityConfig(
                    "coarse",
                    "wac",
                    "wac",
                    band_indices=(2, 1),
                ),
            )
            config = self.config(root, modalities)
            static = self.values((10.0, 20.0, 30.0))
            wac = self.values((40.0, 50.0))
            result = self.reprojection(
                config,
                (
                    (("VIS", "unused", "elevation"), static, static > 0, "complete"),
                    (("UV", "VIS"), wac, wac > 0, "complete"),
                ),
            )

            assembled = assemble_chip(result, config)

            self.assertEqual(
                assembled.band_names,
                ("static_VIS", "elevation", "wac_VIS", "UV"),
            )
            self.assertEqual(
                assembled.band_origins,
                (
                    ("static", "static"),
                    ("static", "static"),
                    ("wac", "wac"),
                    ("wac", "wac"),
                ),
            )
            self.assertEqual(
                tuple(float(item) for item in assembled.pixels[:, 0, 0]),
                (10.0, 30.0, 50.0, 40.0),
            )

    def test_legacy_wac_vis_then_uv_order_is_configuration_driven(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig(
                "coarse",
                "wac",
                "wac",
                band_names=("VIS", "UV"),
            )
            config = self.config(root, (modality,))
            values = self.values((1.0, 2.0))
            result = self.reprojection(
                config,
                ((('UV', 'VIS'), values, values > 0, "complete"),),
            )

            assembled = assemble_chip(result, config)

            self.assertEqual(assembled.band_names, ("VIS", "UV"))
            self.assertEqual(
                tuple(float(item) for item in assembled.pixels[:, 0, 0]),
                (2.0, 1.0),
            )

    def test_missing_optional_modality_uses_nodata_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig(
                "coarse",
                "context",
                "context",
                output_band_names=("context_a", "context_b"),
            )
            config = self.config(root, (modality,), optional=("context",))
            result = self.reprojection(
                config,
                (((), None, None, "missing_optional"),),
            )

            assembled = assemble_chip(result, config)

            self.assertEqual(assembled.band_names, ("context_a", "context_b"))
            self.assertFalse(assembled.valid_mask.any())
            self.assertTrue((assembled.pixels == config.common_nodata).all())
            self.assertEqual(assembled.required_bands, (False, False))

    def test_ineligible_preflight_cannot_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig("coarse", "wac", "wac")
            config = self.config(root, (modality,))
            values = self.values((1.0,))
            result = self.reprojection(
                config,
                ((('VIS',), values, values > 0, "complete"),),
                preflight_status="failed",
                assigned_split=None,
            )

            with self.assertRaises(ChipAssemblyError) as context:
                assemble_chip(result, config)

            self.assertEqual(context.exception.code, "ineligible_preflight")
            self.assertFalse(config.intermediate_root.exists())

    def test_writes_and_reopens_complete_raster_contract(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig(
                "coarse",
                "wac",
                "wac",
                output_band_names=("VIS", "UV"),
            )
            config = self.config(root, (modality,))
            values = self.values((1.25, 2.5))
            mask = self.np.ones(values.shape, dtype=bool)
            mask[0, 0, 1] = False
            values[0, 0, 1] = -9999.0
            result = self.reprojection(
                config,
                ((('source_vis', 'source_uv'), values, mask, "complete"),),
            )
            assembled = assemble_chip(result, config)

            written = write_model_ready_chip(assembled, config)

            self.assertEqual(
                written.path,
                staged_chip_path(config, assembled.sample_id),
            )
            self.assertEqual(written.validation.band_names, ("VIS", "UV"))
            self.assertEqual(written.validation.valid_pixel_counts, (5, 6))
            self.assertEqual(written.validation.compression, "LZW")
            dataset = gdal.Open(str(written.path), gdal.GA_ReadOnly)
            self.assertEqual(dataset.GetRasterBand(1).ReadAsArray()[0, 1], -32768.0)
            self.assertEqual(dataset.GetRasterBand(1).GetDescription(), "VIS")
            dataset = None

    def test_range_checked_downcast_leaves_no_chip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig("coarse", "wac", "wac")
            config = self.config(root, (modality,), output_dtype="float32")
            values = self.values((self.np.finfo(self.np.float64).max,))
            result = self.reprojection(
                config,
                ((('VIS',), values, self.np.ones(values.shape, bool), "complete"),),
            )
            assembled = assemble_chip(result, config)
            output = root / "overflow.tif"

            with self.assertRaises(ChipAssemblyError) as context:
                write_model_ready_chip(assembled, config, output_path=output)

            self.assertEqual(context.exception.code, "floating_cast_out_of_range")
            self.assertFalse(output.exists())

    def test_integer_cast_rejects_fractional_valid_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig("coarse", "wac", "wac")
            config = self.config(root, (modality,), output_dtype="int16")
            values = self.values((1.5,))
            result = self.reprojection(
                config,
                ((('VIS',), values, values > 0, "complete"),),
            )
            assembled = assemble_chip(result, config)

            with self.assertRaises(ChipAssemblyError) as context:
                write_model_ready_chip(assembled, config)

            self.assertEqual(context.exception.code, "integer_cast_not_lossless")
            self.assertFalse(staged_chip_path(config, assembled.sample_id).exists())

    def test_empty_required_band_fails_reopen_validation_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modality = OutputModalityConfig("coarse", "wac", "wac")
            config = self.config(root, (modality,))
            values = self.values((-32768.0,))
            mask = self.np.zeros(values.shape, dtype=bool)
            result = self.reprojection(
                config,
                ((('VIS',), values, mask, "complete"),),
            )
            assembled = assemble_chip(result, config)
            output = staged_chip_path(config, assembled.sample_id)

            with self.assertRaises(ChipAssemblyError) as context:
                write_model_ready_chip(assembled, config)

            self.assertEqual(context.exception.code, "empty_required_band")
            self.assertFalse(output.exists())
            self.assertEqual(tuple(output.parent.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
