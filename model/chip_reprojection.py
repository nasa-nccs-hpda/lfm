"""Merge acquired cubes and reproject modalities onto exact target grids."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

from .chip_acquisition import ChipAcquisitionResult
from .chip_config import ChipConfig, OutputModalityConfig
from .chip_requests import raster_bounds, validate_target_grid_consistency
from .chip_types import TargetGrid
from .tiling_config import TileSourceConfig
from .tiling_policy import band_nodata_values
from .tiling_results import TileCubeRecord


ModalityReprojectionStatus = Literal["complete", "missing_optional"]


class ChipReprojectionError(RuntimeError):
    """One acquired modality could not be mapped to the target grid."""

    def __init__(
        self,
        message: str,
        *,
        acquisition_group: str,
        source_name: str,
        code: str = "reprojection_error",
    ) -> None:
        super().__init__(message)
        self.acquisition_group = acquisition_group
        self.source_name = source_name
        self.code = code


@dataclass(frozen=True)
class SourceZoneGroup:
    """Cubes safe to mosaic together in one source CRS and zoom grid."""

    acquisition_group: str
    source_name: str
    zone: str
    zoom_level: int
    records: tuple[TileCubeRecord, ...]


@dataclass(frozen=True)
class ModalityCubeMapping:
    """One configured output modality and its structured acquired cubes."""

    modality: OutputModalityConfig
    source: TileSourceConfig
    records: tuple[TileCubeRecord, ...]
    zone_groups: tuple[SourceZoneGroup, ...]


@dataclass(frozen=True)
class ReprojectedModality:
    """One source modality represented on the authoritative target grid."""

    acquisition_group: str
    source_name: str
    alias: str
    resampling: str
    status: ModalityReprojectionStatus
    target_grid: TargetGrid
    band_names: tuple[str, ...]
    pixels: Any | None
    valid_mask: Any | None
    nodata_values: tuple[float, ...]
    zone_groups: tuple[SourceZoneGroup, ...]


@dataclass(frozen=True)
class ChipReprojectionResult:
    """Ordered modality arrays produced from one successful acquisition."""

    acquisition: ChipAcquisitionResult
    target_grid: TargetGrid
    modalities: tuple[ReprojectedModality, ...]


@dataclass
class _OpenedCube:
    record: TileCubeRecord
    dataset: Any
    nodata_values: tuple[float | None, ...]


def _libraries():
    try:
        import numpy as np
        from osgeo import gdal, osr
    except ImportError as exc:
        raise RuntimeError(
            "Chip reprojection requires NumPy and the GDAL Python bindings."
        ) from exc
    gdal.UseExceptions()
    return np, gdal, osr


def _error(
    mapping: ModalityCubeMapping,
    message: str,
    *,
    code: str = "reprojection_error",
) -> ChipReprojectionError:
    return ChipReprojectionError(
        message,
        acquisition_group=mapping.modality.acquisition_group,
        source_name=mapping.modality.source_name,
        code=code,
    )


def _record_sort_key(record: TileCubeRecord) -> tuple[object, ...]:
    return (
        record.zone.casefold(),
        record.zoom_level,
        record.tile_y,
        record.tile_x,
        str(record.path),
    )


def _zone_groups(
    acquisition_group: str,
    source_name: str,
    records: Sequence[TileCubeRecord],
) -> tuple[SourceZoneGroup, ...]:
    grouped: dict[tuple[str, int], list[TileCubeRecord]] = {}
    display_zones: dict[tuple[str, int], str] = {}
    for record in records:
        key = (record.zone.casefold(), record.zoom_level)
        grouped.setdefault(key, []).append(record)
        display_zones.setdefault(key, record.zone)
    return tuple(
        SourceZoneGroup(
            acquisition_group=acquisition_group,
            source_name=source_name,
            zone=display_zones[key],
            zoom_level=key[1],
            records=tuple(sorted(grouped[key], key=_record_sort_key)),
        )
        for key in sorted(grouped)
    )


def build_modality_cube_mappings(
    acquisition: ChipAcquisitionResult,
    config: ChipConfig,
) -> tuple[ModalityCubeMapping, ...]:
    """Map configured output modalities to records without parsing filenames."""
    if not isinstance(acquisition, ChipAcquisitionResult):
        raise TypeError("acquisition must be a ChipAcquisitionResult.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if acquisition.status != "complete":
        raise ValueError("Only complete acquisitions may be reprojected.")
    group_results = {
        result.acquisition_group.casefold(): result
        for result in acquisition.group_results
    }
    mappings: list[ModalityCubeMapping] = []
    for modality in config.output_modalities:
        group = config.acquisition_group(modality.acquisition_group)
        group_result = group_results.get(group.name.casefold())
        if group_result is None:
            raise ChipReprojectionError(
                f"Complete acquisition is missing group {group.name!r}.",
                acquisition_group=group.name,
                source_name=modality.source_name,
                code="missing_acquisition_group",
            )
        source = group.tile_config.source(modality.source_name)
        records = tuple(
            sorted(
                (
                    record
                    for record in group_result.records
                    if record.source_name.casefold() == source.name.casefold()
                ),
                key=_record_sort_key,
            )
        )
        invalid_zooms = sorted(
            {
                record.zoom_level
                for record in records
                if record.zoom_level != group.tile_config.zoom_level
            }
        )
        if invalid_zooms:
            raise ChipReprojectionError(
                f"Source {source.name!r} returned unexpected zoom levels "
                f"{invalid_zooms}; expected {group.tile_config.zoom_level}.",
                acquisition_group=group.name,
                source_name=source.name,
                code="unexpected_zoom",
            )
        mappings.append(
            ModalityCubeMapping(
                modality=modality,
                source=source,
                records=records,
                zone_groups=_zone_groups(group.name, source.name, records),
            )
        )
    return tuple(mappings)


def _same_nodata(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return first == second


def _spatial_reference(wkt: str, osr):
    spatial_reference = osr.SpatialReference()
    if spatial_reference.ImportFromWkt(wkt) != 0:
        raise ValueError("Invalid CRS WKT.")
    spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_reference


def _same_crs(first: str, second: str, osr) -> bool:
    try:
        return bool(
            _spatial_reference(first, osr).IsSame(
                _spatial_reference(second, osr)
            )
        )
    except (RuntimeError, ValueError):
        return False


def _opened_band_names(dataset: Any) -> tuple[str, ...]:
    result: list[str] = []
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        name = band.GetMetadataItem("Name") or band.GetDescription()
        result.append(str(name).strip() if name else "")
    return tuple(result)


def _validate_and_open_zone_group(
    mapping: ModalityCubeMapping,
    zone_group: SourceZoneGroup,
    gdal,
    osr,
) -> list[_OpenedCube]:
    opened: list[_OpenedCube] = []
    expected_names: tuple[str, ...] | None = None
    expected_crs: str | None = None
    try:
        for record in zone_group.records:
            if not record.path.is_file():
                raise _error(
                    mapping,
                    f"Acquired cube does not exist: {record.path}",
                    code="missing_cube_file",
                )
            dataset = gdal.Open(str(record.path), gdal.GA_ReadOnly)
            if dataset is None:
                raise _error(
                    mapping,
                    f"Could not open acquired cube: {record.path}",
                    code="unreadable_cube",
                )
            opened.append(
                _OpenedCube(
                    record=record,
                    dataset=dataset,
                    nodata_values=tuple(
                        dataset.GetRasterBand(index).GetNoDataValue()
                        for index in range(1, dataset.RasterCount + 1)
                    ),
                )
            )
            if dataset.RasterCount != len(record.band_names):
                raise _error(
                    mapping,
                    f"Cube {record.path} has {dataset.RasterCount} bands but its "
                    f"record declares {len(record.band_names)}.",
                    code="cube_band_count_mismatch",
                )
            names = _opened_band_names(dataset)
            if any(not name for name in names) or names != record.band_names:
                raise _error(
                    mapping,
                    f"Cube {record.path} band metadata {names!r} does not match "
                    f"its structured record {record.band_names!r}.",
                    code="cube_band_metadata_mismatch",
                )
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise _error(
                    mapping,
                    f"Cubes in {zone_group.zone} do not share band metadata.",
                    code="inconsistent_cube_bands",
                )
            reopened_nodata = opened[-1].nodata_values
            if any(
                not _same_nodata(record_value, reopened_value)
                for record_value, reopened_value in zip(
                    record.nodata_values,
                    reopened_nodata,
                    strict=True,
                )
            ):
                raise _error(
                    mapping,
                    f"Cube {record.path} reopened NoData values "
                    f"{reopened_nodata!r} do not match its structured record "
                    f"{record.nodata_values!r}.",
                    code="cube_nodata_mismatch",
                )
            configured_nodata = tuple(
                band_nodata_values(
                    mapping.source,
                    band_name=name,
                    metadata_source_nodata=reopened_value,
                )[1]
                for name, reopened_value in zip(
                    names,
                    reopened_nodata,
                    strict=True,
                )
            )
            if any(
                not _same_nodata(configured_value, reopened_value)
                for configured_value, reopened_value in zip(
                    configured_nodata,
                    reopened_nodata,
                    strict=True,
                )
            ):
                raise _error(
                    mapping,
                    f"Cube {record.path} NoData values {reopened_nodata!r} do "
                    f"not satisfy configured output policy "
                    f"{configured_nodata!r}.",
                    code="cube_nodata_policy_mismatch",
                )
            projection = dataset.GetProjectionRef()
            if not projection or not _same_crs(projection, record.crs_wkt, osr):
                raise _error(
                    mapping,
                    f"Cube {record.path} CRS does not match its structured record.",
                    code="cube_crs_mismatch",
                )
            if expected_crs is None:
                expected_crs = projection
            elif not _same_crs(projection, expected_crs, osr):
                raise _error(
                    mapping,
                    f"Cubes grouped in {zone_group.zone} do not share one CRS.",
                    code="inconsistent_zone_crs",
                )
            transform = tuple(float(value) for value in dataset.GetGeoTransform())
            determinant = transform[1] * transform[5] - transform[2] * transform[4]
            if (
                not all(math.isfinite(value) for value in transform)
                or math.isclose(determinant, 0.0, abs_tol=1e-15)
                or dataset.RasterXSize < 1
                or dataset.RasterYSize < 1
            ):
                raise _error(
                    mapping,
                    f"Cube {record.path} has invalid raster grid metadata.",
                    code="invalid_cube_grid",
                )
    except Exception:
        opened.clear()
        raise
    return opened


def _normalized_band_dataset(
    opened: _OpenedCube,
    band_index: int,
    nodata: float,
    np,
    gdal,
):
    source_band = opened.dataset.GetRasterBand(band_index)
    pixels = np.asarray(source_band.ReadAsArray(), dtype=np.float64)
    expected_shape = (
        opened.dataset.RasterYSize,
        opened.dataset.RasterXSize,
    )
    if pixels.shape != expected_shape:
        raise RuntimeError(
            f"Could not read band {band_index} from {opened.record.path}."
        )
    valid = np.isfinite(pixels)
    source_nodata = opened.nodata_values[band_index - 1]
    if source_nodata is not None:
        if math.isnan(source_nodata):
            valid &= ~np.isnan(pixels)
        else:
            valid &= pixels != source_nodata
    if not source_band.GetMaskFlags() & gdal.GMF_ALL_VALID:
        mask = np.asarray(source_band.GetMaskBand().ReadAsArray())
        valid &= mask != 0
    normalized = np.where(valid, pixels, nodata)
    dataset = gdal.GetDriverByName("MEM").Create(
        "",
        opened.dataset.RasterXSize,
        opened.dataset.RasterYSize,
        1,
        gdal.GDT_Float64,
    )
    if dataset is None:
        raise RuntimeError("Could not create an in-memory normalized cube band.")
    dataset.SetProjection(opened.dataset.GetProjectionRef())
    dataset.SetGeoTransform(opened.dataset.GetGeoTransform())
    band = dataset.GetRasterBand(1)
    band.WriteArray(normalized)
    band.SetNoDataValue(nodata)
    return dataset


def _target_dataset(target_grid: TargetGrid, nodata: float, gdal):
    dataset = gdal.GetDriverByName("MEM").Create(
        "",
        target_grid.width,
        target_grid.height,
        1,
        gdal.GDT_Float64,
    )
    if dataset is None:
        raise RuntimeError("Could not create an in-memory target raster.")
    dataset.SetProjection(target_grid.crs_wkt)
    dataset.SetGeoTransform(target_grid.transform)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.Fill(nodata)
    return dataset


def _validate_target_dataset(dataset: Any, target_grid: TargetGrid, osr) -> None:
    if (
        dataset.RasterXSize != target_grid.width
        or dataset.RasterYSize != target_grid.height
    ):
        raise RuntimeError("Warped modality dimensions do not match the target grid.")
    transform = tuple(float(value) for value in dataset.GetGeoTransform())
    scale = max(*(abs(value) for value in (*transform, *target_grid.transform)), 1.0)
    tolerance = scale * 1e-10
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        for actual, expected in zip(transform, target_grid.transform, strict=True)
    ):
        raise RuntimeError("Warped modality affine does not match the target grid.")
    if not _same_crs(dataset.GetProjectionRef(), target_grid.crs_wkt, osr):
        raise RuntimeError("Warped modality CRS does not match the target grid.")
    derived_bounds = raster_bounds(transform, target_grid.width, target_grid.height)
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        for actual, expected in zip(
            derived_bounds,
            target_grid.bounds,
            strict=True,
        )
    ):
        raise RuntimeError("Warped modality bounds do not match the target grid.")


def _warp_zone_group(
    mapping: ModalityCubeMapping,
    zone_group: SourceZoneGroup,
    target_grid: TargetGrid,
    nodata: float,
):
    np, gdal, osr = _libraries()
    opened = _validate_and_open_zone_group(mapping, zone_group, gdal, osr)
    try:
        band_names = opened[0].record.band_names
        arrays: list[Any] = []
        masks: list[Any] = []
        for band_index in range(1, len(band_names) + 1):
            normalized = [
                _normalized_band_dataset(
                    item,
                    band_index,
                    nodata,
                    np,
                    gdal,
                )
                for item in opened
            ]
            vrt = gdal.BuildVRT(
                "",
                normalized,
                options=gdal.BuildVRTOptions(
                    resolution="highest",
                    srcNodata=nodata,
                    VRTNodata=nodata,
                ),
            )
            if vrt is None:
                raise _error(
                    mapping,
                    f"Could not mosaic {zone_group.zone} band {band_index}.",
                    code="mosaic_failed",
                )
            destination = _target_dataset(target_grid, nodata, gdal)
            warp_result = gdal.Warp(
                destination,
                vrt,
                options=gdal.WarpOptions(
                    dstSRS=target_grid.crs_wkt,
                    resampleAlg=mapping.modality.resampling,
                    srcNodata=nodata,
                    dstNodata=nodata,
                    multithread=False,
                    errorThreshold=0.0,
                    warpOptions=["INIT_DEST=NO_DATA", "UNIFIED_SRC_NODATA=YES"],
                ),
            )
            if warp_result is None or (
                isinstance(warp_result, (bool, int)) and not warp_result
            ):
                raise _error(
                    mapping,
                    f"Could not reproject {zone_group.zone} band {band_index}.",
                    code="warp_failed",
                )
            destination.FlushCache()
            _validate_target_dataset(destination, target_grid, osr)
            pixels = np.asarray(destination.GetRasterBand(1).ReadAsArray())
            valid = np.isfinite(pixels) & (pixels != nodata)
            arrays.append(pixels)
            masks.append(valid)
            warp_result = None
            destination = None
            vrt = None
            normalized.clear()
        return np.stack(arrays), np.stack(masks), band_names
    finally:
        opened.clear()


def reproject_modality(
    mapping: ModalityCubeMapping,
    target_grid: TargetGrid,
    *,
    output_nodata: float,
) -> ReprojectedModality:
    """Mosaic zone groups separately and composite them on one target grid."""
    validate_target_grid_consistency(target_grid)
    if not math.isfinite(float(output_nodata)):
        raise ValueError("output_nodata must be finite.")
    if not mapping.records:
        if mapping.source.required:
            raise _error(
                mapping,
                f"Required source {mapping.source.name!r} has no acquired cubes.",
                code="missing_required_modality",
            )
        missing_band_names = tuple(
            mapping.source.band_names or mapping.modality.band_names or ()
        )
        return ReprojectedModality(
            acquisition_group=mapping.modality.acquisition_group,
            source_name=mapping.modality.source_name,
            alias=mapping.modality.alias,
            resampling=mapping.modality.resampling,
            status="missing_optional",
            target_grid=target_grid,
            band_names=missing_band_names,
            pixels=None,
            valid_mask=None,
            nodata_values=tuple(
                float(output_nodata) for _ in missing_band_names
            ),
            zone_groups=(),
        )

    np, _, _ = _libraries()
    output = None
    output_mask = None
    band_names: tuple[str, ...] | None = None
    for zone_group in mapping.zone_groups:
        try:
            zone_pixels, zone_mask, zone_names = _warp_zone_group(
                mapping,
                zone_group,
                target_grid,
                float(output_nodata),
            )
        except ChipReprojectionError:
            raise
        except Exception as exc:
            raise _error(
                mapping,
                f"Could not reproject LTM{zone_group.zone}: {exc}",
            ) from exc
        if band_names is None:
            band_names = zone_names
            output = np.full(
                zone_pixels.shape,
                float(output_nodata),
                dtype=np.float64,
            )
            output_mask = np.zeros(zone_mask.shape, dtype=bool)
        elif zone_names != band_names:
            raise _error(
                mapping,
                "LTM zone groups do not share one band contract.",
                code="inconsistent_zone_bands",
            )
        np.copyto(output, zone_pixels, where=zone_mask)
        output_mask |= zone_mask

    if output is None or output_mask is None or band_names is None:
        raise _error(mapping, "No zone groups were available for reprojection.")
    expected_shape = (len(band_names), target_grid.height, target_grid.width)
    if output.shape != expected_shape or output_mask.shape != expected_shape:
        raise _error(
            mapping,
            f"Reprojected array shape does not match {expected_shape}.",
            code="target_shape_mismatch",
        )
    return ReprojectedModality(
        acquisition_group=mapping.modality.acquisition_group,
        source_name=mapping.modality.source_name,
        alias=mapping.modality.alias,
        resampling=mapping.modality.resampling,
        status="complete",
        target_grid=target_grid,
        band_names=band_names,
        pixels=output,
        valid_mask=output_mask,
        nodata_values=tuple(float(output_nodata) for _ in band_names),
        zone_groups=mapping.zone_groups,
    )


def reproject_acquisition(
    acquisition: ChipAcquisitionResult,
    config: ChipConfig,
) -> ChipReprojectionResult:
    """Reproject every configured output modality in configuration order."""
    target_grid = acquisition.prepared_request.request.target_grid
    mappings = build_modality_cube_mappings(acquisition, config)
    modalities = tuple(
        reproject_modality(
            mapping,
            target_grid,
            output_nodata=config.common_nodata,
        )
        for mapping in mappings
    )
    return ChipReprojectionResult(
        acquisition=acquisition,
        target_grid=target_grid,
        modalities=modalities,
    )


__all__ = [
    "ChipReprojectionError",
    "ChipReprojectionResult",
    "ModalityCubeMapping",
    "ModalityReprojectionStatus",
    "ReprojectedModality",
    "SourceZoneGroup",
    "build_modality_cube_mappings",
    "reproject_acquisition",
    "reproject_modality",
]
