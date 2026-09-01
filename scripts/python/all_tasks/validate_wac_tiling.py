#!/usr/bin/env python
"""Validate product-scoped WAC AOI tiling through the modern public API."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import TileConfig, TileCubeRecord, TileSourceConfig
from lfm.model import create_tiles_for_aoi


DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)
DEFAULT_PRODUCT_ID = "M1187363083CE"
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and inspect a product-scoped WAC AOI using TileConfig and "
            "the modern create_tiles_for_aoi API."
        )
    )
    parser.add_argument(
        "--wac-data-dir",
        type=Path,
        default=DEFAULT_WAC_DATA_DIR,
        help="Directory containing WAC GeoTIFFs.",
    )
    parser.add_argument(
        "--wac-index",
        type=Path,
        default=None,
        help="WAC .shp or .gpkg raster index (default: DATA_DIR/output_index.shp).",
    )
    parser.add_argument(
        "--index-layer",
        default=None,
        help="Optional GeoPackage layer name.",
    )
    parser.add_argument(
        "--location-field",
        default="location",
        help="Raster-path field in the WAC vector index.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-id", default=DEFAULT_PRODUCT_ID)
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("UL_LAT", "UL_LON", "LR_LAT", "LR_LON"),
        default=DEFAULT_BOUNDS,
    )
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument("--expected-zone", default="42N")
    parser.add_argument("--expected-record-count", type=int, default=2)
    parser.add_argument("--expected-band-count", type=int, default=7)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON report path (default: OUTPUT_DIR/validation_report.json).",
    )
    return parser.parse_args()


def _same_optional_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def inspect_record(
    record: TileCubeRecord,
    *,
    expected_zone: str,
    expected_product_id: str,
    expected_band_count: int,
) -> dict[str, object]:
    """Validate one written cube against its structured result metadata."""
    from osgeo import gdal, gdalconst, osr

    from lfm.model.TmsTileDef import TmsTileDef

    gdal.UseExceptions()

    if record.source_name != "wac":
        raise AssertionError(f"Unexpected source name: {record.source_name!r}")
    if record.zone != expected_zone:
        raise AssertionError(
            f"Expected zone {expected_zone!r}, got {record.zone!r}."
        )
    if record.product_id != expected_product_id:
        raise AssertionError(
            f"Expected product {expected_product_id!r}, got {record.product_id!r}."
        )
    if not record.path.is_file():
        raise AssertionError(f"Output cube does not exist: {record.path}")

    dataset = gdal.Open(str(record.path), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise AssertionError(f"GDAL could not open output cube: {record.path}")

    tile_def = TmsTileDef.initFromParams(record.zone, record.zoom_level)
    ulx, uly, _, _ = tile_def.getTileBbox(record.tile_x, record.tile_y)
    expected_transform = (
        ulx,
        tile_def.cellSize,
        0.0,
        uly,
        0.0,
        -tile_def.cellSize,
    )
    transform = dataset.GetGeoTransform()
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
        for actual, expected in zip(transform, expected_transform)
    ):
        raise AssertionError(
            f"Unexpected transform for {record.path}: {transform}; "
            f"expected {expected_transform}."
        )

    if dataset.RasterXSize != tile_def.tileWidth:
        raise AssertionError(
            f"Unexpected width for {record.path}: {dataset.RasterXSize}."
        )
    if dataset.RasterYSize != tile_def.tileHeight:
        raise AssertionError(
            f"Unexpected height for {record.path}: {dataset.RasterYSize}."
        )
    if dataset.RasterCount != expected_band_count:
        raise AssertionError(
            f"Expected {expected_band_count} bands in {record.path}, "
            f"got {dataset.RasterCount}."
        )
    if dataset.RasterCount != len(record.band_names):
        raise AssertionError(
            f"Raster and TileCubeRecord band counts disagree for {record.path}."
        )

    raster_srs = dataset.GetSpatialRef()
    record_srs = osr.SpatialReference()
    record_srs.ImportFromWkt(record.crs_wkt)
    if raster_srs is None or not raster_srs.IsSame(record_srs):
        raise AssertionError(
            f"Raster and TileCubeRecord CRS disagree for {record.path}."
        )
    if not raster_srs.IsSame(tile_def.srs):
        raise AssertionError(f"Raster is not on the expected LTM CRS: {record.path}")

    band_names: list[str] = []
    nodata_values: list[float | None] = []
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        name = band.GetMetadataItem("Name")
        nodata = band.GetNoDataValue()
        expected_name = record.band_names[index - 1]
        expected_nodata = record.nodata_values[index - 1]
        if name != expected_name:
            raise AssertionError(
                f"Band {index} name mismatch in {record.path}: "
                f"{name!r} != {expected_name!r}."
            )
        if not _same_optional_number(nodata, expected_nodata):
            raise AssertionError(
                f"Band {index} NoData mismatch in {record.path}: "
                f"{nodata!r} != {expected_nodata!r}."
            )
        band_names.append(name)
        nodata_values.append(nodata)

    details = {
        "source_name": record.source_name,
        "product_id": record.product_id,
        "zone": record.zone,
        "zoom_level": record.zoom_level,
        "tile_x": record.tile_x,
        "tile_y": record.tile_y,
        "path": str(record.path),
        "file_size_bytes": record.path.stat().st_size,
        "shape": [dataset.RasterYSize, dataset.RasterXSize],
        "band_count": dataset.RasterCount,
        "band_names": band_names,
        "nodata_values": nodata_values,
        "crs_name": raster_srs.GetName(),
        "crs_wkt": raster_srs.ExportToWkt(),
        "geotransform": list(transform),
    }
    dataset = None
    return details


def main() -> None:
    args = parse_args()
    wac_index = args.wac_index or args.wac_data_dir / "output_index.shp"
    report_path = args.report_path or args.output_dir / "validation_report.json"

    if not args.wac_data_dir.is_dir():
        raise FileNotFoundError(f"WAC data directory does not exist: {args.wac_data_dir}")
    if not wac_index.is_file():
        raise FileNotFoundError(f"WAC vector index does not exist: {wac_index}")
    if args.expected_record_count < 1:
        raise ValueError("--expected-record-count must be positive.")
    if args.expected_band_count < 1:
        raise ValueError("--expected-band-count must be positive.")

    source = TileSourceConfig(
        name="wac",
        data_dir=args.wac_data_dir,
        index_path=wac_index,
        index_layer=args.index_layer,
        location_field=args.location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    config = TileConfig(
        output_dir=args.output_dir,
        zoom_level=args.zoom_level,
        sources=(source,),
    )
    if config.source("wac").resampling != "bilinear":
        raise AssertionError("WAC tiling must use bilinear resampling.")
    ul_lat, ul_lon, lr_lat, lr_lon = args.bounds
    records = create_tiles_for_aoi(
        config,
        ul_lat=ul_lat,
        ul_lon=ul_lon,
        lr_lat=lr_lat,
        lr_lon=lr_lon,
        selectors={"wac": args.product_id},
    )

    if len(records) != args.expected_record_count:
        raise AssertionError(
            f"Expected {args.expected_record_count} WAC cube records, "
            f"got {len(records)}."
        )

    record_details = [
        inspect_record(
            record,
            expected_zone=args.expected_zone,
            expected_product_id=args.product_id,
            expected_band_count=args.expected_band_count,
        )
        for record in records
    ]
    expected_order = sorted(
        record_details,
        key=lambda item: (item["zone"], item["tile_y"], item["tile_x"]),
    )
    if record_details != expected_order:
        raise AssertionError("TileCubeRecord results are not deterministically ordered.")

    report = {
        "status": "passed",
        "query": {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
            "zoom_level": args.zoom_level,
            "product_id": args.product_id,
        },
        "source": {
            "data_dir": str(args.wac_data_dir),
            "index_path": str(wac_index),
            "index_layer": args.index_layer,
            "location_field": args.location_field,
            "resampling": source.resampling,
        },
        "record_count": len(record_details),
        "records": record_details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWAC tiling validation passed. Report: {report_path}")


if __name__ == "__main__":
    main()
