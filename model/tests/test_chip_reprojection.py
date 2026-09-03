from pathlib import Path
import importlib.util
import json
import tempfile
from unittest import mock
import unittest

from lfm.model.chip_acquisition import (
    AcquisitionGroupResult,
    ChipAcquisitionResult,
)
from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitPercentages,
)
from lfm.model.chip_preflight import PreparedChipRequest
from lfm.model.chip_requests import chip_request_from_aoi
from lfm.model.chip_reprojection import (
    ChipReprojectionError,
    ModalityCubeMapping,
    SourceZoneGroup,
    build_modality_cube_mappings,
    reproject_modality,
)
from lfm.model.chip_splits import SplitAssignment
from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    GeographicAOI,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig
from lfm.model.tiling_results import TileCubeRecord
from lfm.model.lunar_crs import load_lunar_geographic_wkt


HAS_GDAL_NUMPY = (
    importlib.util.find_spec("osgeo") is not None
    and importlib.util.find_spec("numpy") is not None
)


def target_grid(crs_wkt="GEOGCRS[Moon]"):
    return TargetGrid(
        crs_wkt=crs_wkt,
        transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        bounds=(0.0, 0.0, 4.0, 2.0),
        width=4,
        height=2,
    )


def record(path, source_name, zone, zoom=5, tile_x=1, crs_wkt="GEOGCRS[Moon]"):
    return TileCubeRecord(
        source_name=source_name,
        zone=zone,
        zoom_level=zoom,
        tile_x=tile_x,
        tile_y=63,
        product_id=None,
        path=path,
        band_names=("band",),
        crs_wkt=crs_wkt,
        nodata_values=(-9999.0,),
    )


class ChipReprojectionMappingTestCase(unittest.TestCase):
    def test_output_modality_order_and_zone_grouping_are_structured(self):
        root = Path("/tmp/chip-reprojection-mapping")
        coarse_source = TileSourceConfig(
            "static",
            root / "static",
            root / "static/index.gpkg",
        )
        fine_source = TileSourceConfig(
            "nac",
            root / "nac",
            root / "nac/index.gpkg",
            selection_mode="product_id",
        )
        coarse = AcquisitionGroupConfig(
            "coarse",
            TileConfig(root / "unused-coarse", 5, (coarse_source,)),
        )
        fine = AcquisitionGroupConfig(
            "fine",
            TileConfig(root / "unused-fine", 11, (fine_source,)),
        )
        config = ChipConfig(
            output_root=root / "output",
            label_source=root / "labels",
            acquisition_groups=(coarse, fine),
            output_modalities=(
                OutputModalityConfig("fine", "nac", "nac"),
                OutputModalityConfig("coarse", "static", "static"),
            ),
            split_config=SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
        )
        request = ChipRequest(
            sample_id="M1_r0_c0",
            target_grid=target_grid(),
            geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 4.0),
            split_group_key="M1",
        )
        prepared = PreparedChipRequest(
            request=request,
            assignment=SplitAssignment(
                request.sample_id,
                request.split_group_key,
                "train",
                "automatic_percentage",
            ),
            preflight=ChipPreflight(status="passed", assigned_split="train"),
        )
        coarse_records = (
            record(root / "42.tif", "static", "42N"),
            record(root / "43.tif", "static", "43N", tile_x=2),
        )
        fine_records = (
            record(root / "nac.tif", "nac", "42N", zoom=11),
        )
        acquisition = ChipAcquisitionResult(
            prepared_request=prepared,
            status="complete",
            group_results=(
                AcquisitionGroupResult(
                    sample_id=request.sample_id,
                    acquisition_group="coarse",
                    zoom_level=5,
                    output_dir=root / "coarse",
                    logical_aoi=request.geographic_aoi,
                    query_parts=(request.geographic_aoi,),
                    selectors=(),
                    status="complete",
                    records=coarse_records,
                ),
                AcquisitionGroupResult(
                    sample_id=request.sample_id,
                    acquisition_group="fine",
                    zoom_level=11,
                    output_dir=root / "fine",
                    logical_aoi=request.geographic_aoi,
                    query_parts=(request.geographic_aoi,),
                    selectors=(),
                    status="complete",
                    records=fine_records,
                ),
            ),
        )

        mappings = build_modality_cube_mappings(acquisition, config)

        self.assertEqual([item.modality.alias for item in mappings], ["nac", "static"])
        self.assertEqual([item.zone_groups[0].zoom_level for item in mappings], [11, 5])
        self.assertEqual(
            {item.zone for item in mappings[1].zone_groups},
            {"42N", "43N"},
        )

    def test_missing_optional_modality_is_explicit(self):
        source = TileSourceConfig(
            "optional",
            Path("/data/optional"),
            Path("/data/optional/index.gpkg"),
            band_names=("context",),
            required=False,
        )
        mapping = ModalityCubeMapping(
            modality=OutputModalityConfig(
                "context_grid",
                "optional",
                "context",
            ),
            source=source,
            records=(),
            zone_groups=(),
        )

        result = reproject_modality(
            mapping,
            target_grid(),
            output_nodata=-32768.0,
        )

        self.assertEqual(result.status, "missing_optional")
        self.assertEqual(result.band_names, ("context",))
        self.assertEqual(result.nodata_values, (-32768.0,))
        self.assertIsNone(result.pixels)


@unittest.skipUnless(HAS_GDAL_NUMPY, "GDAL and NumPy are required")
class ChipReprojectionRasterTestCase(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from osgeo import gdal, osr

        self.np = np
        self.gdal = gdal
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(4326)
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.crs_wkt = spatial_reference.ExportToWkt()

    def source(self, *, required=True):
        return TileSourceConfig(
            "science",
            Path("/data/science"),
            Path("/data/science/index.gpkg"),
            required=required,
        )

    def grid(self, *, transform=None, bounds=None, width=4, height=2):
        return TargetGrid(
            crs_wkt=self.crs_wkt,
            transform=transform or (0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            bounds=bounds or (0.0, 0.0, 4.0, 2.0),
            width=width,
            height=height,
        )

    def cube(
        self,
        path,
        pixels,
        *,
        transform,
        zone="42N",
        tile_x=1,
        nodata=-9999.0,
        band_names=None,
        crs_wkt=None,
    ):
        values = self.np.asarray(pixels, dtype=self.np.float32)
        if values.ndim == 2:
            values = values[self.np.newaxis, :, :]
        names = tuple(
            band_names
            or (f"band_{index}" for index in range(values.shape[0]))
        )
        dataset = self.gdal.GetDriverByName("GTiff").Create(
            str(path),
            values.shape[2],
            values.shape[1],
            values.shape[0],
            self.gdal.GDT_Float32,
        )
        cube_crs = crs_wkt or self.crs_wkt
        dataset.SetProjection(cube_crs)
        dataset.SetGeoTransform(transform)
        for index, (name, band_pixels) in enumerate(
            zip(names, values, strict=True),
            start=1,
        ):
            band = dataset.GetRasterBand(index)
            band.WriteArray(band_pixels)
            band.SetMetadataItem("Name", name)
            band.SetDescription(name)
            band.SetNoDataValue(nodata)
        dataset = None
        return TileCubeRecord(
            source_name="science",
            zone=zone,
            zoom_level=5,
            tile_x=tile_x,
            tile_y=63,
            product_id=None,
            path=path,
            band_names=names,
            crs_wkt=cube_crs,
            nodata_values=tuple(nodata for _ in names),
        )

    def mapping(self, records, *, resampling="bilinear", required=True):
        source = self.source(required=required)
        grouped = {}
        for item in records:
            grouped.setdefault((item.zone, item.zoom_level), []).append(item)
        zone_groups = tuple(
            SourceZoneGroup(
                acquisition_group="science_grid",
                source_name="science",
                zone=key[0],
                zoom_level=key[1],
                records=tuple(grouped[key]),
            )
            for key in sorted(grouped)
        )
        return ModalityCubeMapping(
            modality=OutputModalityConfig(
                "science_grid",
                "science",
                "science",
                resampling=resampling,
            ),
            source=source,
            records=tuple(records),
            zone_groups=zone_groups,
        )

    def test_continuous_and_categorical_resampling_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.cube(
                root / "source.tif",
                [[0.0, 100.0], [0.0, 100.0]],
                transform=(0.0, 2.0, 0.0, 2.0, 0.0, -1.0),
            )
            grid = self.grid(
                transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
            )

            continuous = reproject_modality(
                self.mapping((source,), resampling="bilinear"),
                grid,
                output_nodata=-32768.0,
            )
            categorical = reproject_modality(
                self.mapping((source,), resampling="nearest"),
                grid,
                output_nodata=-32768.0,
            )

        continuous_values = continuous.pixels[continuous.valid_mask]
        categorical_values = categorical.pixels[categorical.valid_mask]
        self.assertTrue(
            self.np.any((continuous_values > 0) & (continuous_values < 100))
        )
        self.assertTrue(set(self.np.unique(categorical_values)).issubset({0.0, 100.0}))

    def test_declared_nodata_cannot_contaminate_bilinear_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self.cube(
                Path(temporary_directory) / "nodata.tif",
                [[1.0, -9999.0], [1.0, -9999.0]],
                transform=(0.0, 2.0, 0.0, 2.0, 0.0, -1.0),
            )
            result = reproject_modality(
                self.mapping((source,)),
                self.grid(),
                output_nodata=-32768.0,
            )

        valid_values = result.pixels[result.valid_mask]
        self.assertTrue(self.np.allclose(valid_values, 1.0))
        self.assertTrue(self.np.all(result.pixels[~result.valid_mask] == -32768.0))

    def test_distinct_cube_nodata_values_normalize_before_mosaic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = self.cube(
                root / "left.tif",
                [[1.0, -9999.0], [1.0, -9999.0]],
                transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                tile_x=1,
                nodata=-9999.0,
            )
            right = self.cube(
                root / "right.tif",
                [[2.0, -1234.0], [2.0, -1234.0]],
                transform=(2.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                tile_x=2,
                nodata=-1234.0,
            )

            result = reproject_modality(
                self.mapping((left, right)),
                self.grid(),
                output_nodata=-32768.0,
            )

        valid_values = set(self.np.unique(result.pixels[result.valid_mask]))
        self.assertEqual(valid_values, {1.0, 2.0})
        self.assertTrue(self.np.all(result.pixels[~result.valid_mask] == -32768.0))

    def test_multiple_adjacent_tiles_merge_before_one_zone_warp(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = self.cube(
                root / "left.tif",
                self.np.ones((2, 2)),
                transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                tile_x=1,
            )
            right = self.cube(
                root / "right.tif",
                self.np.full((2, 2), 2.0),
                transform=(2.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                tile_x=2,
            )

            result = reproject_modality(
                self.mapping((left, right), resampling="nearest"),
                self.grid(),
                output_nodata=-32768.0,
            )

        self.assertEqual(len(result.zone_groups), 1)
        self.np.testing.assert_array_equal(
            result.pixels[0],
            self.np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]]),
        )

    def test_zone_groups_composite_independently_on_exact_target(self):
        import lfm.model.chip_reprojection as chip_reprojection

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = self.cube(
                root / "42N.tif",
                self.np.ones((2, 2)),
                transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                zone="42N",
                tile_x=1,
            )
            right = self.cube(
                root / "43N.tif",
                self.np.full((2, 2), 2.0),
                transform=(2.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                zone="43N",
                tile_x=1,
            )
            with mock.patch(
                "lfm.model.chip_reprojection._warp_zone_group",
                wraps=chip_reprojection._warp_zone_group,
            ) as warp_zone:
                result = reproject_modality(
                    self.mapping((left, right), resampling="nearest"),
                    self.grid(),
                    output_nodata=-32768.0,
                )

        self.assertEqual(warp_zone.call_count, 2)
        self.np.testing.assert_array_equal(
            result.pixels[0],
            self.np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]]),
        )

    def test_lunar_target_crossing_ltm_zone_boundary_composites_both_zones(self):
        from osgeo import osr

        repository_root = Path(__file__).resolve().parents[2]
        geographic_wkt = load_lunar_geographic_wkt()
        request = chip_request_from_aoi(
            sample_id="M1_r0_c0",
            crs_wkt=geographic_wkt,
            bounds=(155.9, 1.0, 156.1, 1.2),
            width=20,
            height=20,
            split_group_key="zone-boundary",
        )

        def projected_grid(zone):
            definition = json.loads(
                (
                    repository_root
                    / "TMS"
                    / "RG"
                    / f"tms_LTM_{zone}RG.json"
                ).read_text(encoding="utf-8")
            )
            geographic = osr.SpatialReference()
            geographic.ImportFromWkt(geographic_wkt)
            projected = osr.SpatialReference()
            projected.ImportFromWkt(definition["crs"])
            geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            projected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transformation = osr.CoordinateTransformation(geographic, projected)
            west, east = (155.9, 156.0) if zone == "42N" else (156.0, 156.1)
            points = tuple(
                transformation.TransformPoint(longitude, latitude)
                for longitude in (west, east)
                for latitude in (1.0, 1.2)
            )
            xs = tuple(point[0] for point in points)
            ys = tuple(point[1] for point in points)
            width = 20
            height = 20
            transform = (
                min(xs),
                (max(xs) - min(xs)) / width,
                0.0,
                max(ys),
                0.0,
                -(max(ys) - min(ys)) / height,
            )
            return projected.ExportToWkt(), transform, width, height

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for zone, value in (("42N", 1.0), ("43N", 2.0)):
                zone_wkt, transform, width, height = projected_grid(zone)
                records.append(
                    self.cube(
                        root / f"{zone}.tif",
                        self.np.full((height, width), value),
                        transform=transform,
                        zone=zone,
                        tile_x=1,
                        crs_wkt=zone_wkt,
                    )
                )

            result = reproject_modality(
                self.mapping(tuple(records), resampling="nearest"),
                request.target_grid,
                output_nodata=-32768.0,
            )

        self.assertEqual(len(result.zone_groups), 2)
        self.assertEqual(
            set(self.np.unique(result.pixels[result.valid_mask])),
            {1.0, 2.0},
        )
        self.assertEqual(result.target_grid, request.target_grid)

    def test_reopened_nodata_must_match_structured_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            actual = self.cube(
                Path(temporary_directory) / "source.tif",
                self.np.ones((2, 2)),
                transform=(0.0, 2.0, 0.0, 2.0, 0.0, -1.0),
            )
            inconsistent = TileCubeRecord(
                source_name=actual.source_name,
                zone=actual.zone,
                zoom_level=actual.zoom_level,
                tile_x=actual.tile_x,
                tile_y=actual.tile_y,
                product_id=actual.product_id,
                path=actual.path,
                band_names=actual.band_names,
                crs_wkt=actual.crs_wkt,
                nodata_values=(-1234.0,),
            )

            with self.assertRaises(ChipReprojectionError) as raised:
                reproject_modality(
                    self.mapping((inconsistent,)),
                    self.grid(),
                    output_nodata=-32768.0,
                )

        self.assertEqual(raised.exception.code, "cube_nodata_mismatch")

    def test_rotated_target_transform_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self.cube(
                Path(temporary_directory) / "source.tif",
                self.np.stack(
                    (
                        self.np.ones((4, 4)),
                        self.np.full((4, 4), 2.0),
                    )
                ),
                transform=(0.0, 0.6, 0.0, 2.4, 0.0, -0.6),
                band_names=("first", "second"),
            )
            grid = self.grid(
                transform=(0.0, 0.5, 0.1, 2.0, 0.1, -0.5),
                bounds=(0.0, 0.0, 2.4, 2.4),
                width=4,
                height=4,
            )

            result = reproject_modality(
                self.mapping((source,), resampling="nearest"),
                grid,
                output_nodata=-32768.0,
            )

        self.assertEqual(result.target_grid.transform, grid.transform)
        self.assertEqual(result.pixels.shape, (2, 4, 4))
        self.assertEqual(result.band_names, ("first", "second"))


if __name__ == "__main__":
    unittest.main()
