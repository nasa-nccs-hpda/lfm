#!/usr/bin/env python
"""Validate product-scoped NAC AOI tiling through the modern public API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import TileConfig, TileSourceConfig, create_tiles_for_aoi

from validate_wac_tiling import inspect_record


DEFAULT_NAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_NAC_Pho_Sites"
)
DEFAULT_PRODUCT_ID = "M1117899885LE"
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and inspect a product-scoped NAC AOI using TileConfig and "
            "the modern create_tiles_for_aoi API."
        )
    )
    parser.add_argument("--nac-data-dir", type=Path, default=DEFAULT_NAC_DATA_DIR)
    parser.add_argument(
        "--nac-index",
        type=Path,
        default=None,
        help="NAC .shp or .gpkg index (default: DATA_DIR/output_index.shp).",
    )
    parser.add_argument("--index-layer", default=None)
    parser.add_argument("--location-field", default="location")
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
    parser.add_argument("--expected-band-count", type=int, default=1)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON report path (default: OUTPUT_DIR/validation_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nac_index = args.nac_index or args.nac_data_dir / "output_index.shp"
    report_path = args.report_path or args.output_dir / "validation_report.json"

    if not args.nac_data_dir.is_dir():
        raise FileNotFoundError(f"NAC data directory does not exist: {args.nac_data_dir}")
    if not nac_index.is_file():
        raise FileNotFoundError(f"NAC vector index does not exist: {nac_index}")
    if args.expected_band_count < 1:
        raise ValueError("--expected-band-count must be positive.")

    source = TileSourceConfig(
        name="nac",
        data_dir=args.nac_data_dir,
        index_path=nac_index,
        index_layer=args.index_layer,
        location_field=args.location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
        # NAC is sparse. AOI tiles without the selected observation are an
        # expected skip rather than a fatal missing-source error.
        required=False,
    )
    config = TileConfig(
        output_dir=args.output_dir,
        zoom_level=args.zoom_level,
        sources=(source,),
    )
    if config.source("nac").resampling != "bilinear":
        raise AssertionError("NAC tiling must use bilinear resampling.")

    ul_lat, ul_lon, lr_lat, lr_lon = args.bounds
    records = create_tiles_for_aoi(
        config,
        ul_lat=ul_lat,
        ul_lon=ul_lon,
        lr_lat=lr_lat,
        lr_lon=lr_lon,
        selectors={"nac": args.product_id},
    )
    if not records:
        raise AssertionError(
            f"NAC product {args.product_id!r} produced no cubes for the AOI."
        )

    record_details = [
        inspect_record(
            record,
            expected_source_name="nac",
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
            "data_dir": str(args.nac_data_dir),
            "index_path": str(nac_index),
            "index_layer": args.index_layer,
            "location_field": args.location_field,
            "resampling": source.resampling,
            "required": source.required,
        },
        "record_count": len(record_details),
        "records": record_details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nNAC tiling validation passed. Report: {report_path}")


if __name__ == "__main__":
    main()
