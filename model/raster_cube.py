"""GDAL raster warping and writing for configured LTM source cubes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from .TmsTileDef import TmsTileDef
from .tiling_config import TileSourceConfig
from .tiling_policy import band_nodata_values
from .tiling_results import TileCubeRecord


gdal.UseExceptions()

@dataclass(frozen=True)
class WarpedBand:
    name: str
    pixels: np.ndarray
    source_nodata: float | None
    output_nodata: float | None


def _gdal_nodata_argument(values: list[float | None]):
    populated = [value for value in values if value is not None]
    if not populated:
        return None
    if len(values) == 1:
        return populated[0]
    return values


def _same_nodata_value(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if np.isnan(first) or np.isnan(second):
        return bool(np.isnan(first) and np.isnan(second))
    return first == second


def _band_name(dataset, path: Path, band_index: int) -> str:
    band = dataset.GetRasterBand(band_index)
    metadata_name = band.GetMetadataItem("Name")
    description = band.GetDescription()
    if metadata_name:
        return metadata_name
    if description:
        return description
    if dataset.RasterCount == 1:
        return path.stem
    return f"{path.stem}-{band_index - 1}"


def _has_valid_pixels(
    pixels: np.ndarray,
    *,
    source_nodata: float | None,
    output_nodata: float | None,
) -> bool:
    valid = np.isfinite(pixels)
    if source_nodata is not None:
        valid &= pixels != source_nodata
    if output_nodata is not None:
        valid &= pixels != output_nodata
    return bool(np.any(valid))


def _select_bands(
    source: TileSourceConfig,
    bands: list[WarpedBand],
) -> list[WarpedBand]:
    if source.band_names is not None:
        by_name: dict[str, WarpedBand] = {}
        duplicates: set[str] = set()
        for band in bands:
            if band.name in by_name:
                duplicates.add(band.name)
            by_name[band.name] = band
        requested_duplicates = duplicates.intersection(source.band_names)
        if requested_duplicates:
            raise ValueError(
                f"Source {source.name!r} has duplicate requested band names: "
                f"{sorted(requested_duplicates)}"
            )
        missing = [name for name in source.band_names if name not in by_name]
        if missing:
            raise ValueError(
                f"Source {source.name!r} is missing configured bands: {missing}"
            )
        return [by_name[name] for name in source.band_names]
    if source.band_indices is not None:
        missing = [index for index in source.band_indices if index > len(bands)]
        if missing:
            raise IndexError(
                f"Source {source.name!r} has {len(bands)} available band(s), "
                f"but requested 1-based indices {missing}."
            )
        return [bands[index - 1] for index in source.band_indices]
    return bands


def warp_source_to_tile(
    source: TileSourceConfig,
    raster_paths: list[Path],
    *,
    tile_def: TmsTileDef,
    bounds: tuple[float, float, float, float],
) -> list[WarpedBand]:
    """Warp all selected rasters for one source onto one LTM tile grid."""
    ulx, uly, lrx, lry = bounds
    result: list[WarpedBand] = []
    for path in raster_paths:
        dataset = gdal.Open(str(path), gdalconst.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Could not open indexed raster: {path}")

        names = [
            _band_name(dataset, path, index)
            for index in range(1, dataset.RasterCount + 1)
        ]
        metadata_source_nodata = [
            dataset.GetRasterBand(index).GetNoDataValue()
            for index in range(1, dataset.RasterCount + 1)
        ]
        nodata_pairs = [
            band_nodata_values(
                source,
                band_name=name,
                metadata_source_nodata=metadata_nodata,
            )
            for name, metadata_nodata in zip(
                names,
                metadata_source_nodata,
                strict=True,
            )
        ]
        source_nodata = [pair[0] for pair in nodata_pairs]
        output_nodata = [pair[1] for pair in nodata_pairs]
        warp_kwargs = {
            "outputBounds": [ulx, lry, lrx, uly],
            "dstSRS": tile_def.srs,
            "width": tile_def.tileWidth,
            "height": tile_def.tileHeight,
            "format": "MEM",
            "resampleAlg": gdal.GRA_Bilinear,
        }
        source_arg = _gdal_nodata_argument(source_nodata)
        output_arg = _gdal_nodata_argument(output_nodata)
        # Let GDAL consume native band metadata directly when the configured
        # contract does not alter it. Besides avoiding redundant arguments,
        # this preserves intrinsic validity masks and matches the legacy path.
        uses_intrinsic_nodata = all(
            _same_nodata_value(source_value, metadata_value)
            and _same_nodata_value(output_value, metadata_value)
            for (source_value, output_value), metadata_value in zip(
                nodata_pairs,
                metadata_source_nodata,
                strict=True,
            )
        )
        if source_arg is not None and not uses_intrinsic_nodata:
            warp_kwargs["srcNodata"] = source_arg
            # Passing srcNodata explicitly changes GDAL's default multi-band
            # behavior to UNIFIED_SRC_NODATA=YES.  Lunar modalities such as
            # WAC have independent valid footprints in each band, so treating
            # a pixel as valid whenever any band is valid lets sentinel-scale
            # values contaminate bilinear interpolation in the other bands.
            # PARTIAL retains independent per-band NoData masks while also
            # marking a source pixel globally transparent when every band is
            # NoData.  This matches GDAL's intrinsic-NoData behavior used by
            # the legacy tiler.
            warp_kwargs["warpOptions"] = ["UNIFIED_SRC_NODATA=PARTIAL"]
        if output_arg is not None and not uses_intrinsic_nodata:
            warp_kwargs["dstNodata"] = output_arg
        warped = gdal.Warp("", dataset, **warp_kwargs)
        if warped is None:
            raise RuntimeError(f"Could not warp indexed raster: {path}")
        array = np.asarray(warped.ReadAsArray())
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        for index, name in enumerate(names):
            pixels = array[index]
            source_value, output_value = nodata_pairs[index]
            if not _has_valid_pixels(
                pixels,
                source_nodata=source_value,
                output_nodata=output_value,
            ):
                continue
            if output_value is not None:
                invalid = ~np.isfinite(pixels)
                if source_value is not None:
                    invalid |= pixels == source_value
                pixels = np.where(invalid, output_value, pixels)
            result.append(
                WarpedBand(
                    name=name,
                    pixels=pixels,
                    source_nodata=source_value,
                    output_nodata=output_value,
                )
            )
        warped = None
        dataset = None
    return _select_bands(source, result)


def write_tile_cube(
    path: Path,
    bands: list[WarpedBand],
    *,
    source: TileSourceConfig,
    product_id: str | None,
    zone: str,
    zoom_level: int,
    tile_x: int,
    tile_y: int,
    tile_def: TmsTileDef,
    ulx: float,
    uly: float,
) -> TileCubeRecord:
    """Write warped bands and return metadata matching the output GeoTIFF."""
    if not bands:
        raise ValueError(f"Cannot write an empty cube: {path}")
    dtype = np.result_type(*(band.pixels.dtype for band in bands))
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        tile_def.tileWidth,
        tile_def.tileHeight,
        len(bands),
        gdal_dtype,
        options=["BIGTIFF=YES", "TILED=YES", "COMPRESS=LZW"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output cube: {path}")
    dataset.SetSpatialRef(tile_def.srs)
    dataset.SetGeoTransform([ulx, tile_def.cellSize, 0, uly, 0, -tile_def.cellSize])
    for index, warped_band in enumerate(bands, start=1):
        band = dataset.GetRasterBand(index)
        band.WriteArray(warped_band.pixels.astype(dtype, copy=False))
        band.SetMetadataItem("Name", warped_band.name)
        band.SetDescription(warped_band.name)
        if warped_band.output_nodata is not None:
            band.SetNoDataValue(float(warped_band.output_nodata))
    dataset = None
    path.chmod(0o664)
    return TileCubeRecord(
        source_name=source.name,
        zone=zone,
        zoom_level=zoom_level,
        tile_x=tile_x,
        tile_y=tile_y,
        product_id=product_id,
        path=path,
        band_names=tuple(band.name for band in bands),
        crs_wkt=tile_def.srs.ExportToWkt(),
        nodata_values=tuple(band.output_nodata for band in bands),
    )


__all__ = ["WarpedBand", "warp_source_to_tile", "write_tile_cube"]
