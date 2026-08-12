#!/usr/bin/env python
"""
Create filtered static datacubes for NoData/-inf diagnostics.

This intentionally avoids the WAC/dynamic cube path. It queries the same
staticLinks tile index used by Pipeline, clips only the selected static rasters
to the AOI's TMS tiles, writes filtered StaticCube GeoTIFFs, and records
invalid-value counts before and after writing.

Default selected rasters:
  - GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif
  - GlobeNoPolesDeltaS1_v2.iau.tif
  - any static source whose filename contains "lola_kaguya"

Examples
--------
Run from the lfm repo root on Explore:

    python scripts/python/run_filtered_static_cube_diagnostics.py \
        --aoi-tif notebooks/qgis_nodata_samples/CPR_25_75pct_nodata.tif \
        --aoi-tif notebooks/qgis_nodata_samples/CPR_scattered_nodata.tif \
        --tile-db /explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites/output_index.shp \
        --out-dir /explore/nobackup/people/$USER/static_cube_debug

Or use explicit geographic bounds:

    python scripts/python/run_filtered_static_cube_diagnostics.py \
        --bounds 1.0 149.0 0.5 149.5 \
        --tile-db /explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites/output_index.shp \
        --out-dir ./static_cube_debug
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import gdal, gdal_array, gdalconst, osr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model.Pipeline import Pipeline
from lfm.model.TmsIntersector import TmsIntersector
from lfm.model.TmsTileDef import TmsTileDef

gdal.UseExceptions()


GLOBAL_MINIRF_NAMES = {
    "globlenopolesdeltacpr_v2-offsetto49d.iau.tif",
    "globenopolesdeltacpr_v2-offsetto49d.iau.tif",
    "globenopolesdeltas1_v2.iau.tif",
}

DEFAULT_OUTPUT_NODATA = -32768.0


@dataclass
class RasterStats:
    nodata: int
    nan: int
    neginf: int
    posinf: int
    finite_valid_min: float | None
    finite_valid_max: float | None
    all_nodata: bool


@dataclass
class BandRecord:
    name: str
    source_path: str
    source_nodata: float | None
    clipped_stats: RasterStats
    written_stats: RasterStats | None = None


def is_selected_static(path: Path) -> bool:
    name = path.name.lower()
    return name in GLOBAL_MINIRF_NAMES or "lola_kaguya" in name


def scalar_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def count_stats(array: np.ndarray, nodata: float | None) -> RasterStats:
    data = np.asarray(array)
    finite = np.isfinite(data)
    nan = np.isnan(data)
    neginf = np.isneginf(data)
    posinf = np.isposinf(data)

    if nodata is None:
        nodata_mask = np.zeros(data.shape, dtype=bool)
    else:
        nodata_mask = data == nodata

    valid = finite & ~nodata_mask
    if np.any(valid):
        valid_values = data[valid]
        finite_valid_min = float(valid_values.min())
        finite_valid_max = float(valid_values.max())
    else:
        finite_valid_min = None
        finite_valid_max = None

    return RasterStats(
        nodata=int(np.count_nonzero(nodata_mask)),
        nan=int(np.count_nonzero(nan)),
        neginf=int(np.count_nonzero(neginf)),
        posinf=int(np.count_nonzero(posinf)),
        finite_valid_min=finite_valid_min,
        finite_valid_max=finite_valid_max,
        all_nodata=bool(nodata is not None and np.all(nodata_mask)),
    )


def dataset_bounds_latlon(path: Path) -> tuple[float, float, float, float]:
    """Return bounds as (ul_lat, ul_lon, lr_lat, lr_lon)."""
    ds = gdal.Open(str(path), gdalconst.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open AOI raster: {path}")

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    src_srs = osr.SpatialReference()
    src_wkt = ds.GetProjection()
    if not src_wkt:
        raise RuntimeError(f"AOI raster has no CRS/projection: {path}")
    src_srs.ImportFromWkt(src_wkt)

    moon_srs = osr.SpatialReference()
    moon_srs.ImportFromWkt(Pipeline.MOON_SRS)

    # Use x/y order consistently: x = lon/easting, y = lat/northing.
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    moon_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, moon_srs)

    def px_to_xy(col: float, row: float) -> tuple[float, float]:
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        return x, y

    lon_lat = []
    for col, row in ((0, 0), (width, 0), (width, height), (0, height)):
        x, y = px_to_xy(col, row)
        lon, lat, _ = transform.TransformPoint(x, y)
        lon_lat.append((lon, lat))

    lons = [item[0] for item in lon_lat]
    lats = [item[1] for item in lon_lat]

    return max(lats), min(lons), min(lats), max(lons)


def query_filtered_static_paths(
    pipeline: Pipeline,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
) -> list[Path]:
    layer = pipeline._query(
        ul_lat,
        ul_lon,
        lr_lat,
        lr_lon,
        Pipeline.STATIC_FILE_DB,
    )

    paths = []
    for feature in layer:
        path = Path(feature["location"])
        if is_selected_static(path):
            paths.append(path)

    return sorted(paths, key=lambda p: p.name.lower())


def warp_static_source(
    source_path: Path,
    ulx: float,
    uly: float,
    lrx: float,
    lry: float,
    dst_srs: osr.SpatialReference,
    width: int,
    height: int,
    resample_alg: int,
    safe_nodata: bool,
    output_nodata: float,
) -> tuple[gdal.Dataset, float | None]:
    src_ds = gdal.Open(str(source_path), gdalconst.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Could not open source raster: {source_path}")

    src_nodata = scalar_or_none(src_ds.GetRasterBand(1).GetNoDataValue())

    warp_kwargs = {
        "outputBounds": [ulx, lry, lrx, uly],
        "dstSRS": dst_srs,
        "width": width,
        "height": height,
        "format": "MEM",
        "resampleAlg": resample_alg,
    }

    if safe_nodata and src_nodata is not None:
        warp_kwargs["srcNodata"] = src_nodata
        warp_kwargs["dstNodata"] = output_nodata

    clip_ds = gdal.Warp("", src_ds, **warp_kwargs)
    if clip_ds is None:
        raise RuntimeError(f"gdal.Warp returned None for {source_path}")

    return clip_ds, src_nodata


def arrays_from_clip(
    clip_ds: gdal.Dataset,
    source_path: Path,
    source_nodata: float | None,
) -> list[tuple[str, np.ndarray, float | None, RasterStats]]:
    data = clip_ds.ReadAsArray()
    if data is None:
        raise RuntimeError(f"ReadAsArray returned None for {source_path}")

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    records = []
    for band_idx in range(data.shape[0]):
        band = clip_ds.GetRasterBand(band_idx + 1)
        clipped_nodata = scalar_or_none(band.GetNoDataValue())
        stats_nodata = clipped_nodata if clipped_nodata is not None else source_nodata
        name = source_path.stem if data.shape[0] == 1 else f"{source_path.stem}-{band_idx}"
        pixels = data[band_idx]
        records.append((name, pixels, stats_nodata, count_stats(pixels, stats_nodata)))

    return records


def write_filtered_static_cube(
    out_file: Path,
    band_records: list[tuple[str, Path, np.ndarray, float | None, RasterStats]],
    tile_def: TmsTileDef,
    ulx: float,
    uly: float,
    output_nodata: float,
    normalize_on_write: bool,
) -> list[BandRecord]:
    if not band_records:
        raise ValueError("No band records to write.")

    first_pixels = band_records[0][2]
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(first_pixels.dtype)
    height, width = first_pixels.shape

    ds = gdal.GetDriverByName("GTiff").Create(
        str(out_file),
        width,
        height,
        len(band_records),
        data_type,
        options=["BIGTIFF=YES", "TILED=YES", "COMPRESS=LZW"],
    )
    ds.SetSpatialRef(tile_def.srs)
    ds.SetGeoTransform([ulx, tile_def.cellSize, 0, uly, 0, -tile_def.cellSize])

    report_records: list[BandRecord] = []

    for band_idx, (name, source_path, pixels, source_nodata, clipped_stats) in enumerate(
        band_records,
        start=1,
    ):
        write_pixels = pixels
        if normalize_on_write and source_nodata is not None:
            write_pixels = np.where(pixels == source_nodata, output_nodata, pixels)

        band = ds.GetRasterBand(band_idx)
        band.WriteArray(write_pixels)
        band.SetMetadataItem("Name", name)
        band.SetMetadataItem("SourcePath", str(source_path))
        band.SetNoDataValue(output_nodata)

        written_stats = count_stats(write_pixels, output_nodata)
        report_records.append(
            BandRecord(
                name=name,
                source_path=str(source_path),
                source_nodata=source_nodata,
                clipped_stats=clipped_stats,
                written_stats=written_stats,
            )
        )

    ds = None
    out_file.chmod(0o664)

    return report_records


def resampling_from_name(name: str) -> int:
    mapping = {
        "average": gdal.GRA_Average,
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported resampling method: {name}") from exc


def process_aoi(
    label: str,
    bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> dict:
    ul_lat, ul_lon, lr_lat, lr_lon = bounds
    aoi_dir = args.out_dir / label
    aoi_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(args.tile_db, aoi_dir, debug=args.debug)
    tmsi = TmsIntersector()
    tile_indexes = tmsi.getTids(ul_lat, ul_lon, lr_lat, lr_lon, args.zoom_level)

    report = {
        "label": label,
        "bounds": {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
        },
        "zoom_level": args.zoom_level,
        "safe_nodata": args.safe_nodata,
        "normalize_on_write": args.normalize_on_write,
        "resampling": args.resampling,
        "tiles": [],
    }

    resample_alg = resampling_from_name(args.resampling)

    for idx in tile_indexes:
        tile_x = idx["tileX"]
        tile_y = idx["tileY"]
        zone = idx["zone"]
        zoom_level = idx["zoomLevel"]

        tile_def = TmsTileDef.initFromParams(zone, zoom_level)
        ulx, uly, lrx, lry = tile_def.getTileBbox(tile_x, tile_y)
        query_ul_lat, query_ul_lon = tile_def.ltmToLatLon(ulx, uly)
        query_lr_lat, query_lr_lon = tile_def.ltmToLatLon(lrx, lry)

        selected_paths = query_filtered_static_paths(
            pipeline,
            query_ul_lat,
            query_ul_lon,
            query_lr_lat,
            query_lr_lon,
        )

        out_name = (
            f"FilteredStaticCube-LTM{zone}_Zoom-{zoom_level}"
            f"_Tile-{tile_x}-{tile_y}.tif"
        )
        out_file = aoi_dir / out_name

        band_records = []
        errors = []

        for source_path in selected_paths:
            try:
                clip_ds, source_nodata = warp_static_source(
                    source_path=source_path,
                    ulx=ulx,
                    uly=uly,
                    lrx=lrx,
                    lry=lry,
                    dst_srs=tile_def.srs,
                    width=tile_def.tileWidth,
                    height=tile_def.tileHeight,
                    resample_alg=resample_alg,
                    safe_nodata=args.safe_nodata,
                    output_nodata=args.output_nodata,
                )

                for name, pixels, stats_nodata, clipped_stats in arrays_from_clip(
                    clip_ds,
                    source_path,
                    source_nodata,
                ):
                    band_records.append(
                        (name, source_path, pixels, stats_nodata, clipped_stats)
                    )

            except Exception as exc:
                errors.append({"source_path": str(source_path), "error": str(exc)})

        if band_records:
            written_records = write_filtered_static_cube(
                out_file=out_file,
                band_records=band_records,
                tile_def=tile_def,
                ulx=ulx,
                uly=uly,
                output_nodata=args.output_nodata,
                normalize_on_write=args.normalize_on_write,
            )
        else:
            written_records = []

        report["tiles"].append(
            {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "zone": zone,
                "zoom_level": zoom_level,
                "output_file": str(out_file) if band_records else None,
                "selected_source_count": len(selected_paths),
                "selected_sources": [str(path) for path in selected_paths],
                "band_count": len(written_records),
                "bands": [asdict(record) for record in written_records],
                "errors": errors,
            }
        )

    return report


def parse_bounds(values: Iterable[str]) -> tuple[float, float, float, float]:
    vals = [float(value) for value in values]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--bounds requires exactly 4 values")
    return vals[0], vals[1], vals[2], vals[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create filtered static cubes and invalid-value diagnostics."
    )
    parser.add_argument(
        "--aoi-tif",
        action="append",
        type=Path,
        default=[],
        help="AOI/sample GeoTIFF. May be supplied multiple times.",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        metavar=("UL_LAT", "UL_LON", "LR_LAT", "LR_LON"),
        help="Explicit AOI bounds in Moon lat/lon.",
    )
    parser.add_argument(
        "--tile-db",
        type=Path,
        required=True,
        help="Dynamic tile DB shapefile path required by Pipeline construction.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for filtered cubes and diagnostics JSON.",
    )
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument(
        "--resampling",
        choices=("bilinear", "average", "nearest", "cubic"),
        default="bilinear",
        help="GDAL warp resampling. Default matches Pipeline._clip.",
    )
    parser.add_argument(
        "--safe-nodata",
        action="store_true",
        help="Pass srcNodata/dstNodata into gdal.Warp.",
    )
    parser.add_argument(
        "--normalize-on-write",
        action="store_true",
        help=(
            "Replace source nodata with output nodata before writing. "
            "By default, the script writes raw clipped pixels, matching the "
            "current effective Pipeline._writeStaticCube behavior."
        ),
    )
    parser.add_argument(
        "--output-nodata",
        type=float,
        default=DEFAULT_OUTPUT_NODATA,
        help="NoData value assigned to written filtered cubes.",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if not args.aoi_tif and args.bounds is None:
        parser.error("Provide at least one --aoi-tif or one --bounds AOI.")

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tile_db = args.tile_db.resolve()

    if not args.tile_db.exists():
        parser.error(f"--tile-db does not exist: {args.tile_db}")

    return args


def main() -> int:
    args = parse_args()

    aoi_specs: list[tuple[str, tuple[float, float, float, float]]] = []

    for aoi_tif in args.aoi_tif:
        aoi_tif = aoi_tif.resolve()
        label = aoi_tif.stem
        bounds = dataset_bounds_latlon(aoi_tif)
        aoi_specs.append((label, bounds))

    if args.bounds is not None:
        aoi_specs.append(("explicit_bounds", parse_bounds(args.bounds)))

    reports = []
    for label, bounds in aoi_specs:
        print(f"Processing AOI {label}: {bounds}")
        reports.append(process_aoi(label, bounds, args))

    report_path = args.out_dir / "filtered_static_cube_diagnostics.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    print(f"Wrote diagnostics: {report_path}")
    print(f"Wrote filtered cubes under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
