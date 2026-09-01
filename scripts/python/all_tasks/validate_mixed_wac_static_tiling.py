#!/usr/bin/env python
"""Validate ordered WAC-plus-static AOI tiling through the modern API."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import (
    BandNoDataOverride,
    MINIRF_PRESERVE_SOURCE_NODATA_BANDS,
    MINIRF_STATIC_NODATA,
    STATIC_BAND_NAMES,
    STATIC_DEFAULT_NODATA,
    TileConfig,
    TileSourceConfig,
    create_tiles_for_aoi,
)

from validate_wac_tiling import inspect_record


DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)
DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")
DEFAULT_PRODUCT_ID = "M1187363083CE"
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate ordered WAC-plus-static LTM AOI cubes."
    )
    parser.add_argument("--wac-data-dir", type=Path, default=DEFAULT_WAC_DATA_DIR)
    parser.add_argument("--wac-index", type=Path, default=None)
    parser.add_argument("--static-data-dir", type=Path, default=DEFAULT_STATIC_DATA_DIR)
    parser.add_argument("--static-index", type=Path, default=None)
    parser.add_argument("--static-index-layer", default=None)
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
    parser.add_argument("--expected-tile-count", type=int, default=2)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def _assert_nodata(name: str, actual: float | None, expected: float) -> None:
    if actual is None or not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-7,
        abs_tol=0.0,
    ):
        raise AssertionError(
            f"Static band {name!r} has NoData {actual!r}; expected {expected!r}."
        )


def validate_static_policy(record) -> None:
    if record.band_names != STATIC_BAND_NAMES:
        raise AssertionError("Static output does not follow STATIC_BAND_NAMES order.")
    preserve_names = set(MINIRF_PRESERVE_SOURCE_NODATA_BANDS)
    for name, nodata in zip(record.band_names, record.nodata_values, strict=True):
        expected = (
            MINIRF_STATIC_NODATA
            if name in preserve_names
            else STATIC_DEFAULT_NODATA
        )
        _assert_nodata(name, nodata, expected)


def main() -> None:
    args = parse_args()
    wac_index = args.wac_index or args.wac_data_dir / "output_index.shp"
    static_index = args.static_index or args.static_data_dir / "db2.shp"
    report_path = args.report_path or args.output_dir / "validation_report.json"

    for name, directory in (
        ("WAC", args.wac_data_dir),
        ("static", args.static_data_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} data directory does not exist: {directory}")
    for name, index_path in (("WAC", wac_index), ("static", static_index)):
        if not index_path.is_file():
            raise FileNotFoundError(f"{name} vector index does not exist: {index_path}")
    if args.expected_tile_count < 1:
        raise ValueError("--expected-tile-count must be positive.")

    wac_source = TileSourceConfig(
        name="wac",
        data_dir=args.wac_data_dir,
        index_path=wac_index,
        location_field=args.location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    static_source = TileSourceConfig(
        name="static",
        data_dir=args.static_data_dir,
        index_path=static_index,
        index_layer=args.static_index_layer,
        location_field=args.location_field,
        selection_mode="all_intersecting",
        band_names=STATIC_BAND_NAMES,
        resampling="bilinear",
        output_nodata=STATIC_DEFAULT_NODATA,
        band_nodata_overrides=tuple(
            BandNoDataOverride(
                band_name=name,
                source_value=MINIRF_STATIC_NODATA,
                preserve_source=True,
            )
            for name in MINIRF_PRESERVE_SOURCE_NODATA_BANDS
        ),
    )
    config = TileConfig(
        output_dir=args.output_dir,
        zoom_level=args.zoom_level,
        sources=(wac_source, static_source),
    )
    if any(source.resampling != "bilinear" for source in config.sources):
        raise AssertionError("Every tiling source must use bilinear resampling.")

    ul_lat, ul_lon, lr_lat, lr_lon = args.bounds
    records = create_tiles_for_aoi(
        config,
        ul_lat=ul_lat,
        ul_lon=ul_lon,
        lr_lat=lr_lat,
        lr_lon=lr_lon,
        selectors={"wac": args.product_id},
    )
    expected_record_count = args.expected_tile_count * 2
    if len(records) != expected_record_count:
        raise AssertionError(
            f"Expected {expected_record_count} mixed records, got {len(records)}."
        )

    details: list[dict[str, object]] = []
    for offset in range(0, len(records), 2):
        wac_record, static_record = records[offset : offset + 2]
        if (wac_record.zone, wac_record.tile_x, wac_record.tile_y) != (
            static_record.zone,
            static_record.tile_x,
            static_record.tile_y,
        ):
            raise AssertionError("WAC/static record pairs do not refer to the same tile.")
        validate_static_policy(static_record)
        details.append(
            inspect_record(
                wac_record,
                expected_source_name="wac",
                expected_zone=args.expected_zone,
                expected_product_id=args.product_id,
                expected_band_count=7,
            )
        )
        details.append(
            inspect_record(
                static_record,
                expected_source_name="static",
                expected_zone=args.expected_zone,
                expected_product_id=None,
                expected_band_count=len(STATIC_BAND_NAMES),
            )
        )

    expected_order = sorted(
        details,
        key=lambda item: (
            item["zone"],
            item["tile_y"],
            item["tile_x"],
            0 if item["source_name"] == "wac" else 1,
        ),
    )
    if details != expected_order:
        raise AssertionError("Mixed records are not in tile and source declaration order.")

    report = {
        "status": "passed",
        "query": {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
            "zoom_level": args.zoom_level,
            "wac_product_id": args.product_id,
        },
        "sources": [
            {
                "name": "wac",
                "data_dir": str(args.wac_data_dir),
                "index_path": str(wac_index),
                "resampling": wac_source.resampling,
            },
            {
                "name": "static",
                "data_dir": str(args.static_data_dir),
                "index_path": str(static_index),
                "resampling": static_source.resampling,
                "band_count": len(STATIC_BAND_NAMES),
                "default_nodata": STATIC_DEFAULT_NODATA,
                "preserve_source_nodata_bands": list(
                    MINIRF_PRESERVE_SOURCE_NODATA_BANDS
                ),
            },
        ],
        "record_count": len(details),
        "records": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nMixed WAC/static validation passed. Report: {report_path}")


if __name__ == "__main__":
    main()
