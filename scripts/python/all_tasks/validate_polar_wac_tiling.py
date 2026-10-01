#!/usr/bin/env python3
"""Create and validate one dynamic-only north-polar WAC tile cube."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import (
    TileSourceConfig,
    TileSourcePreparation,
    compose_tile_sources,
    create_tiles_for_point,
    lunar_product_id_from_raster_path,
    prepare_tile_config,
)


DEFAULT_SOURCE_PATH = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/"
    "LROC/WAC/wac_glob_morf_mos/"
    "WAC_GLOBAL_P900N0000_100M.eqc.iau2.LPS_N.vrt"
)
DEFAULT_GRID_ID = "LPS_N"
DEFAULT_ZOOM_LEVEL = 4
DEFAULT_LATITUDE = 86.0
DEFAULT_LONGITUDE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--plot",
        type=Path,
        help=(
            "Comparison PNG path. Defaults to "
            "<work-dir>/polar_wac_source_tile_comparison.png."
        ),
    )
    parser.add_argument("--grid-id", default=DEFAULT_GRID_ID)
    parser.add_argument("--zoom-level", type=int, default=DEFAULT_ZOOM_LEVEL)
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE)
    return parser.parse_args()


def _same_nodata(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=0.0)


def _valid_pixels(pixels, nodata):
    import numpy as np

    valid = np.isfinite(pixels)
    if nodata is not None:
        if math.isnan(nodata):
            valid &= ~np.isnan(pixels)
        else:
            valid &= pixels != nodata
    return valid


def _dataset_extent(dataset) -> tuple[float, float, float, float]:
    transform = tuple(float(value) for value in dataset.GetGeoTransform())
    if not math.isclose(transform[2], 0.0, abs_tol=1e-12) or not math.isclose(
        transform[4], 0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "The polar WAC comparison expects north-up rasters, got "
            f"{transform!r}."
        )
    left = transform[0]
    top = transform[3]
    right = left + dataset.RasterXSize * transform[1]
    bottom = top + dataset.RasterYSize * transform[5]
    return left, right, bottom, top


def create_comparison_plot(
    *,
    source_path: Path,
    record,
    grid_id: str,
    zoom_level: int,
    plot_path: Path,
) -> dict[str, object]:
    """Plot the native VRT window, written cube, and independent warp error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal, osr

    from lfm.model.grid_tile_def import tile_definition_for_grid
    from lfm.model.lunar_crs import raster_crs_equivalent

    gdal.UseExceptions()
    tile_def = tile_definition_for_grid(grid_id, zoom_level)
    ulx, uly, lrx, lry = tile_def.getTileBbox(record.tile_x, record.tile_y)
    tile_extent = (ulx, lrx, lry, uly)

    source = gdal.Open(str(source_path), gdal.GA_ReadOnly)
    if source is None:
        raise RuntimeError(f"Could not open polar WAC source VRT: {source_path}")
    source_srs = source.GetSpatialRef()
    if source_srs is None:
        raise AssertionError(f"Polar WAC source has no CRS: {source_path}")
    source_srs = source_srs.Clone()
    expected_srs = tile_def.srs.Clone()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    expected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not raster_crs_equivalent(source_srs, expected_srs):
        raise AssertionError(
            "The polar WAC source CRS does not match the LPS_N tile grid."
        )

    native_window = gdal.Translate(
        "",
        source,
        format="MEM",
        bandList=[1],
        projWin=[ulx, uly, lrx, lry],
    )
    if native_window is None:
        raise RuntimeError("Could not read the native polar WAC source window.")
    native_band = native_window.GetRasterBand(1)
    native_nodata = native_band.GetNoDataValue()
    native_pixels = np.asarray(native_band.ReadAsArray(), dtype=np.float64)
    native_valid = _valid_pixels(native_pixels, native_nodata)
    if not np.any(native_valid):
        raise AssertionError("The native polar WAC source window has no valid pixels.")
    native_extent = _dataset_extent(native_window)

    reference = gdal.Warp(
        "",
        source,
        format="MEM",
        outputBounds=[ulx, lry, lrx, uly],
        dstSRS=tile_def.srs,
        width=tile_def.tileWidth,
        height=tile_def.tileHeight,
        resampleAlg=gdal.GRA_Bilinear,
    )
    if reference is None:
        raise RuntimeError("Could not create the independent GDAL reference warp.")
    reference_band = reference.GetRasterBand(1)
    reference_nodata = reference_band.GetNoDataValue()
    reference_pixels = np.asarray(reference_band.ReadAsArray(), dtype=np.float64)
    reference_valid = _valid_pixels(reference_pixels, reference_nodata)

    output = gdal.Open(str(record.path), gdal.GA_ReadOnly)
    if output is None:
        raise RuntimeError(f"Could not reopen written polar cube: {record.path}")
    output_band = output.GetRasterBand(1)
    output_nodata = output_band.GetNoDataValue()
    output_pixels = np.asarray(output_band.ReadAsArray(), dtype=np.float64)
    output_valid = _valid_pixels(output_pixels, output_nodata)
    if output_pixels.shape != reference_pixels.shape:
        raise AssertionError(
            "Reference and written tile shapes differ: "
            f"{reference_pixels.shape!r} != {output_pixels.shape!r}."
        )

    common_valid = reference_valid & output_valid
    if not np.any(common_valid):
        raise AssertionError(
            "The reference warp and written tile share no valid pixels."
        )
    mask_mismatch_count = int(np.count_nonzero(reference_valid != output_valid))
    signed_difference = output_pixels[common_valid] - reference_pixels[common_valid]
    absolute_difference = np.abs(signed_difference)
    difference_image = np.full(output_pixels.shape, np.nan, dtype=np.float64)
    difference_image[common_valid] = absolute_difference

    display_values = np.concatenate(
        (native_pixels[native_valid], output_pixels[output_valid])
    )
    display_min, display_max = (
        float(value) for value in np.percentile(display_values, [2.0, 98.0])
    )
    if not display_max > display_min:
        display_min = float(np.min(display_values))
        display_max = float(np.max(display_values))
    if not display_max > display_min:
        display_max = display_min + max(abs(display_min) * 1e-6, 1e-12)
    difference_max = float(np.percentile(absolute_difference, 99.5))
    if difference_max <= 0.0:
        difference_max = max((display_max - display_min) * 1e-7, 1e-12)

    native_display = np.where(native_valid, native_pixels, np.nan)
    output_display = np.where(output_valid, output_pixels, np.nan)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        constrained_layout=True,
    )
    source_image = axes[0].imshow(
        native_display,
        cmap="gray",
        vmin=display_min,
        vmax=display_max,
        extent=native_extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[0].set_title(
        "Native LPS_N VRT window\n"
        f"{native_window.RasterXSize} × {native_window.RasterYSize} pixels"
    )
    axes[1].imshow(
        output_display,
        cmap="gray",
        vmin=display_min,
        vmax=display_max,
        extent=tile_extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[1].set_title(
        "Written bilinear tile\n"
        f"{output.RasterXSize} × {output.RasterYSize} pixels"
    )
    difference_plot = axes[2].imshow(
        difference_image,
        cmap="magma",
        vmin=0.0,
        vmax=difference_max,
        extent=tile_extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[2].set_title(
        "|written − independent warp|\n"
        f"max={float(np.max(absolute_difference)):.3g}, "
        f"RMSE={float(np.sqrt(np.mean(signed_difference**2))):.3g}"
    )
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(ulx, lrx)
        axis.set_ylim(lry, uly)
        axis.set_xlabel("LPS_N easting (m)")
        axis.set_ylabel("LPS_N northing (m)")
        axis.ticklabel_format(style="plain", useOffset=False)
    figure.colorbar(
        source_image,
        ax=axes[:2],
        label="WAC pixel value (shared 2nd–98th percentile stretch)",
        shrink=0.84,
        pad=0.02,
    )
    figure.colorbar(
        difference_plot,
        ax=axes[2],
        label="Absolute pixel difference",
        shrink=0.84,
        pad=0.02,
    )
    figure.suptitle(
        f"{record.product_id} — {grid_id} zoom {zoom_level} "
        f"tile ({record.tile_x}, {record.tile_y})"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    plot_path.chmod(0o664)

    native_transform = tuple(
        float(value) for value in native_window.GetGeoTransform()
    )
    comparison = {
        "plot_path": str(plot_path),
        "plot_size_bytes": plot_path.stat().st_size,
        "native_window_dimensions": [
            native_window.RasterXSize,
            native_window.RasterYSize,
        ],
        "native_window_geotransform": list(native_transform),
        "native_window_extent": list(native_extent),
        "native_valid_pixel_count": int(np.count_nonzero(native_valid)),
        "reference_valid_pixel_count": int(np.count_nonzero(reference_valid)),
        "written_valid_pixel_count": int(np.count_nonzero(output_valid)),
        "common_valid_pixel_count": int(np.count_nonzero(common_valid)),
        "mask_mismatch_count": mask_mismatch_count,
        "difference_mean_absolute": float(np.mean(absolute_difference)),
        "difference_rmse": float(np.sqrt(np.mean(signed_difference**2))),
        "difference_max_absolute": float(np.max(absolute_difference)),
        "difference_p99_5_absolute": float(
            np.percentile(absolute_difference, 99.5)
        ),
        "display_percentiles": {
            "p02": display_min,
            "p98": display_max,
        },
    }
    output = None
    reference = None
    native_window = None
    source = None
    return comparison


def inspect_cube(record, *, grid_id: str, zoom_level: int) -> dict[str, object]:
    import numpy as np
    from osgeo import gdal, osr

    from lfm.model.grid_tile_def import tile_definition_for_grid
    from lfm.model.lunar_crs import raster_crs_equivalent

    gdal.UseExceptions()
    path = record.path
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not reopen written polar cube: {path}")
    if (dataset.RasterXSize, dataset.RasterYSize) != (512, 512):
        raise AssertionError(
            f"Unexpected polar cube dimensions for {path}: "
            f"{dataset.RasterXSize}x{dataset.RasterYSize}"
        )
    if dataset.RasterCount != 1:
        raise AssertionError(
            f"Expected one WAC mosaic band in {path}, got {dataset.RasterCount}."
        )

    tile_def = tile_definition_for_grid(grid_id, zoom_level)
    ulx, uly, _, _ = tile_def.getTileBbox(record.tile_x, record.tile_y)
    expected_transform = (
        ulx,
        tile_def.cellSize,
        0.0,
        uly,
        0.0,
        -tile_def.cellSize,
    )
    actual_transform = tuple(float(value) for value in dataset.GetGeoTransform())
    if not np.allclose(
        actual_transform,
        expected_transform,
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError(
            f"Polar cube geotransform mismatch: {actual_transform!r} != "
            f"{expected_transform!r}."
        )

    written_srs = dataset.GetSpatialRef()
    if written_srs is None:
        raise AssertionError(f"Written polar cube has no CRS: {path}")
    written_srs = written_srs.Clone()
    expected_srs = tile_def.srs.Clone()
    written_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    expected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not raster_crs_equivalent(written_srs, expected_srs):
        raise AssertionError(
            "Written polar cube CRS does not match the LPS_N tile grid.\n"
            f"Written: {written_srs.ExportToWkt()}\n"
            f"Expected: {expected_srs.ExportToWkt()}"
        )

    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if not _same_nodata(nodata, record.nodata_values[0]):
        raise AssertionError(
            f"Reopened NoData {nodata!r} does not match record "
            f"{record.nodata_values[0]!r}."
        )
    pixels = np.asarray(band.ReadAsArray(), dtype=np.float64)
    valid = np.isfinite(pixels)
    if nodata is not None:
        valid &= pixels != nodata
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise AssertionError(f"Polar WAC cube contains no valid pixels: {path}")

    image_structure = dataset.GetMetadata("IMAGE_STRUCTURE")
    if image_structure.get("COMPRESSION") != "LZW":
        raise AssertionError(
            f"Expected LZW compression, got {image_structure!r} for {path}."
        )
    block_width, block_height = band.GetBlockSize()
    if block_width <= 0 or block_height <= 0:
        raise AssertionError(f"Invalid GeoTIFF block size for {path}.")

    details = {
        "path": str(path),
        "source_name": record.source_name,
        "product_id": record.product_id,
        "grid_id": record.grid_id,
        "zoom_level": record.zoom_level,
        "tile_x": record.tile_x,
        "tile_y": record.tile_y,
        "band_names": list(record.band_names),
        "nodata_values": list(record.nodata_values),
        "dimensions": [dataset.RasterXSize, dataset.RasterYSize],
        "geotransform": list(actual_transform),
        "cell_size": tile_def.cellSize,
        "compression": image_structure["COMPRESSION"],
        "block_size": [block_width, block_height],
        "valid_pixel_count": valid_count,
        "valid_min": float(np.min(pixels[valid])),
        "valid_max": float(np.max(pixels[valid])),
        "size_bytes": path.stat().st_size,
        "group_writable": bool(path.stat().st_mode & 0o020),
    }
    if not details["group_writable"]:
        raise AssertionError(f"Written cube is not group-writable: {path}")
    dataset = None
    return details


def main() -> None:
    args = parse_args()
    source_path = args.source_path.resolve()
    plot_path = (
        args.work_dir / "polar_wac_source_tile_comparison.png"
        if args.plot is None
        else args.plot
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Polar WAC VRT does not exist: {source_path}")
    if source_path.suffix.casefold() != ".vrt":
        raise ValueError(f"Expected the selected polar WAC VRT: {source_path}")
    if args.work_dir.exists():
        raise FileExistsError(
            f"Polar WAC validation directory already exists: {args.work_dir}"
        )
    if not math.isfinite(args.lat) or not math.isfinite(args.lon):
        raise ValueError("Test latitude and longitude must be finite.")
    if args.zoom_level < 1:
        raise ValueError("--zoom-level must be positive.")

    args.work_dir.mkdir(parents=True)
    output_dir = args.work_dir / "cubes"
    index_path = args.work_dir / "wac_polar_index.gpkg"
    product_id = lunar_product_id_from_raster_path(source_path)
    source = TileSourceConfig(
        name="wac",
        data_dir=source_path.parent,
        index_path=index_path,
        location_field="location",
        selection_mode="product_id",
        band_indices=(1,),
        resampling="bilinear",
        preserve_source_nodata=True,
        required=True,
    )
    enabled_sources = compose_tile_sources(
        dynamic_sources=(source,),
        include_dynamic=True,
        include_static=False,
    )
    if enabled_sources != (source,):
        raise AssertionError("Dynamic-only source composition changed unexpectedly.")

    print(f"Source VRT: {source_path}", flush=True)
    print(f"Product ID: {product_id}", flush=True)
    print(
        f"Point: ({args.lat}, {args.lon}); grid: {args.grid_id}; "
        f"zoom: {args.zoom_level}",
        flush=True,
    )
    print(f"Work directory: {args.work_dir}", flush=True)
    print("Preparing an isolated one-feature raster index...", flush=True)
    started = perf_counter()
    prepared = prepare_tile_config(
        output_dir=output_dir,
        zoom_level=args.zoom_level,
        sources=tuple(
            TileSourcePreparation(item, image_glob=source_path.name)
            for item in enabled_sources
        ),
        stdout=sys.stdout,
    )
    preparation_seconds = perf_counter() - started
    if len(prepared.indexes) != 1:
        raise AssertionError("Expected exactly one prepared WAC raster index.")
    index_result = prepared.indexes[0]
    if index_result.feature_count != 1:
        raise AssertionError(
            f"Expected one indexed VRT, got {index_result.feature_count}."
        )
    if tuple(path.resolve() for path in index_result.raster_paths) != (source_path,):
        raise AssertionError(
            "The isolated index contains a raster other than the selected VRT: "
            f"{index_result.raster_paths!r}."
        )

    print("Creating one dynamic-only north-polar WAC tile...", flush=True)
    tiling_started = perf_counter()
    records = create_tiles_for_point(
        prepared.config,
        lat=args.lat,
        lon=args.lon,
        grid_id=args.grid_id,
        selectors={source.name: product_id},
    )
    tiling_seconds = perf_counter() - tiling_started
    if len(records) != 1:
        raise AssertionError(
            f"Expected one point tile cube, got {len(records)}: {records!r}"
        )
    record = records[0]
    if (
        record.source_name != source.name
        or record.product_id != product_id
        or record.grid_id != args.grid_id
        or record.zoom_level != args.zoom_level
    ):
        raise AssertionError(f"Unexpected structured tile record: {record!r}")
    cube = inspect_cube(
        record,
        grid_id=args.grid_id,
        zoom_level=args.zoom_level,
    )
    print("Creating native-source, written-tile, and difference plot...", flush=True)
    comparison_started = perf_counter()
    comparison = create_comparison_plot(
        source_path=source_path,
        record=record,
        grid_id=args.grid_id,
        zoom_level=args.zoom_level,
        plot_path=plot_path,
    )
    comparison_seconds = perf_counter() - comparison_started

    report = {
        "status": "passed",
        "source": {
            "path": str(source_path),
            "product_id": product_id,
            "selection_mode": source.selection_mode,
            "band_indices": list(source.band_indices or ()),
            "resampling": source.resampling,
            "preserve_source_nodata": source.preserve_source_nodata,
        },
        "source_modes": {
            "include_dynamic": True,
            "include_static": False,
            "enabled_sources": [item.name for item in enabled_sources],
        },
        "query": {
            "kind": "point",
            "lat": args.lat,
            "lon": args.lon,
            "grid_id": args.grid_id,
            "zoom_level": args.zoom_level,
        },
        "index": {
            "path": str(index_result.index_path),
            "driver_name": index_result.driver_name,
            "layer_name": index_result.layer_name,
            "feature_count": index_result.feature_count,
            "raster_paths": [str(path) for path in index_result.raster_paths],
        },
        "cube": cube,
        "comparison": comparison,
        "timing_seconds": {
            "preparation": preparation_seconds,
            "tiling": tiling_seconds,
            "comparison_plot": comparison_seconds,
            "total": preparation_seconds + tiling_seconds + comparison_seconds,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nComparison plot: {plot_path}")
    print(f"\nPolar WAC tiling validation passed. Report: {args.report}")


if __name__ == "__main__":
    main()
