#!/usr/bin/env python3
"""Create static LTM cubes and diagnose their per-band NoData behavior.

The diagnostic intentionally uses the current mixed-tiling static contract:
most bands normalize invalid source pixels to ``-32768``, while the two
Mini-RF bands preserve their source sentinel. It then reopens each written
GeoTIFF and compares, per band:

* source NoData metadata and the resolved source NoData policy;
* intended output NoData retained by :class:`TileCubeRecord`;
* NoData metadata that survives the GeoTIFF close/reopen cycle;
* exact source, intended, standardized, and alternate sentinel counts; and
* finite non-sentinel values at or below the proposed standardized sentinel.

This does not modify the tiling policy. In bands already normalized to the
standardized sentinel, a valid source pixel equal to that sentinel is no longer
distinguishable from normalized NoData. Those bands are marked as requiring a
source-level collision check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, gdalconst


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import (  # noqa: E402
    BandNoDataOverride,
    MINIRF_PRESERVE_SOURCE_NODATA_BANDS,
    MINIRF_STATIC_NODATA,
    STATIC_BAND_NAMES,
    STATIC_DEFAULT_NODATA,
    TileConfig,
    TileSourceConfig,
    create_tiles_for_aoi,
)
from lfm.model.raster_cube import _band_name  # noqa: E402
from lfm.model.tiling_policy import band_nodata_values  # noqa: E402
from lfm.model.vector_index import (  # noqa: E402
    open_vector_layer,
    resolve_indexed_raster_path,
)


gdal.UseExceptions()

DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create representative static LTM cubes and diagnose per-band "
            "source, intended, persisted, and pixel-level NoData behavior."
        )
    )
    parser.add_argument("--static-data-dir", type=Path, default=DEFAULT_STATIC_DATA_DIR)
    parser.add_argument("--static-index", type=Path, default=None)
    parser.add_argument("--static-index-layer", default=None)
    parser.add_argument("--location-field", default="location")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("UL_LAT", "UL_LON", "LR_LAT", "LR_LON"),
        default=DEFAULT_BOUNDS,
    )
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument(
        "--standardized-nodata",
        type=float,
        default=STATIC_DEFAULT_NODATA,
        help="Candidate common output NoData value. Default: -32768.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def same_value(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return math.isclose(float(first), float(second), rel_tol=1e-7, abs_tol=0.0)


def exact_mask(array: np.ndarray, value: float | None) -> np.ndarray:
    if value is None:
        return np.zeros(array.shape, dtype=bool)
    if math.isnan(value):
        return np.isnan(array)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(value, dtype=array.dtype).item()
    except (OverflowError, TypeError, ValueError):
        return np.zeros(array.shape, dtype=bool)
    if not np.isfinite(converted):
        return np.zeros(array.shape, dtype=bool)
    return array == converted


def json_value(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def format_value(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return format(float(value), ".17g")


def build_static_source(
    *,
    data_dir: Path,
    index_path: Path,
    index_layer: str | None,
    location_field: str,
) -> TileSourceConfig:
    return TileSourceConfig(
        name="static",
        data_dir=data_dir,
        index_path=index_path,
        index_layer=index_layer,
        location_field=location_field,
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


def indexed_paths(
    source: TileSourceConfig,
) -> list[Path]:
    dataset, layer = open_vector_layer(
        source.index_path,
        layer_name=source.index_layer,
    )
    try:
        definition = layer.GetLayerDefn()
        if definition.GetFieldIndex(source.location_field) < 0:
            fields = [
                definition.GetFieldDefn(index).GetName()
                for index in range(definition.GetFieldCount())
            ]
            raise KeyError(
                f"Index {source.index_path} has no {source.location_field!r} "
                f"field; available fields: {fields}"
            )
        layer.ResetReading()
        paths = [
            resolve_indexed_raster_path(
                source.data_dir,
                feature.GetField(source.location_field),
            )
            for feature in layer
        ]
    finally:
        layer = None
        dataset = None
    return sorted(set(paths), key=lambda path: str(path).lower())


def read_source_band_contract(
    source: TileSourceConfig,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read source metadata only for paths matching canonical band filenames."""
    canonical_names = set(STATIC_BAND_NAMES)
    candidate_paths = [
        path for path in indexed_paths(source) if path.stem in canonical_names
    ]
    contract: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for path in candidate_paths:
        dataset = gdal.Open(str(path), gdalconst.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Could not open canonical static source: {path}")
        try:
            for band_index in range(1, dataset.RasterCount + 1):
                name = _band_name(dataset, path, band_index)
                if name not in canonical_names:
                    continue
                if name in contract:
                    duplicates.append(name)
                    continue
                metadata_nodata = dataset.GetRasterBand(band_index).GetNoDataValue()
                metadata_nodata = (
                    float(metadata_nodata) if metadata_nodata is not None else None
                )
                resolved_source, intended_output = band_nodata_values(
                    source,
                    band_name=name,
                    metadata_source_nodata=metadata_nodata,
                )
                contract[name] = {
                    "source_path": str(path),
                    "source_band_index": band_index,
                    "metadata_source_nodata": metadata_nodata,
                    "resolved_source_nodata": resolved_source,
                    "configured_output_nodata": intended_output,
                }
        finally:
            dataset = None
    return contract, sorted(set(duplicates))


def analyze_written_band(
    *,
    array: np.ndarray,
    band_name: str,
    contract_index: int,
    source_details: dict[str, Any],
    intended_output_nodata: float | None,
    persisted_nodata: float | None,
    standardized_nodata: float,
    known_output_sentinels: tuple[float, ...],
) -> dict[str, Any]:
    finite = np.isfinite(array)
    standardized_mask = exact_mask(array, standardized_nodata)
    intended_mask = exact_mask(array, intended_output_nodata)
    resolved_source_nodata = source_details.get("resolved_source_nodata")
    source_mask = exact_mask(array, resolved_source_nodata)

    all_known_sentinels = np.zeros(array.shape, dtype=bool)
    sentinel_counts: dict[str, int] = {}
    for sentinel in known_output_sentinels:
        mask = exact_mask(array, sentinel)
        all_known_sentinels |= mask
        sentinel_counts[format_value(json_value(sentinel))] = int(
            np.count_nonzero(mask)
        )

    if resolved_source_nodata is not None:
        all_known_sentinels |= source_mask
    scientific_valid = finite & ~all_known_sentinels
    if np.any(scientific_valid):
        scientific_values = array[scientific_valid].astype(np.float64, copy=False)
        scientific_min = float(scientific_values.min())
        scientific_max = float(scientific_values.max())
        below_standardized = int(
            np.count_nonzero(scientific_values < standardized_nodata)
        )
    else:
        scientific_min = None
        scientific_max = None
        below_standardized = 0

    intended_is_standardized = same_value(
        intended_output_nodata,
        standardized_nodata,
    )
    alternate_output_count = sum(
        count
        for sentinel, count in zip(
            known_output_sentinels,
            sentinel_counts.values(),
            strict=True,
        )
        if not same_value(sentinel, intended_output_nodata)
    )
    source_sentinel_leak_count = (
        0
        if same_value(resolved_source_nodata, intended_output_nodata)
        else int(np.count_nonzero(source_mask))
    )

    if below_standardized > 0:
        assessment = "scientific_values_below_standardized_nodata"
    elif not intended_is_standardized and np.any(standardized_mask):
        assessment = "standardized_value_present_as_non_intended_sentinel"
    elif source_sentinel_leak_count > 0:
        assessment = "source_sentinel_leaked_after_normalization"
    elif alternate_output_count > 0:
        assessment = "alternate_output_sentinel_present"
    elif intended_is_standardized:
        assessment = "already_standardized_source_collision_not_testable_in_cube"
    else:
        assessment = "no_cube_level_collision_observed"

    return {
        "contract_index": contract_index,
        "band_name": band_name,
        "dtype": str(array.dtype),
        "pixel_count": int(array.size),
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "nonfinite_pixel_count": int(np.count_nonzero(~finite)),
        "metadata_source_nodata": json_value(
            source_details.get("metadata_source_nodata")
        ),
        "resolved_source_nodata": json_value(resolved_source_nodata),
        "intended_output_nodata": json_value(intended_output_nodata),
        "persisted_geotiff_nodata": json_value(persisted_nodata),
        "persisted_metadata_matches_intended": same_value(
            persisted_nodata,
            intended_output_nodata,
        ),
        "standardized_nodata": standardized_nodata,
        "standardized_sentinel_count": int(np.count_nonzero(standardized_mask)),
        "intended_output_sentinel_count": int(np.count_nonzero(intended_mask)),
        "source_sentinel_count_after_warp": int(np.count_nonzero(source_mask)),
        "source_sentinel_leak_count": source_sentinel_leak_count,
        "known_output_sentinel_counts": sentinel_counts,
        "alternate_output_sentinel_count": alternate_output_count,
        "scientific_valid_pixel_count": int(np.count_nonzero(scientific_valid)),
        "scientific_valid_min": scientific_min,
        "scientific_valid_max": scientific_max,
        "scientific_values_below_standardized_count": below_standardized,
        "assessment": assessment,
    }


def inspect_cube(
    record: Any,
    *,
    source_contract: dict[str, dict[str, Any]],
    standardized_nodata: float,
    known_output_sentinels: tuple[float, ...],
) -> dict[str, Any]:
    dataset = gdal.Open(str(record.path), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not reopen written static cube: {record.path}")
    try:
        if dataset.RasterCount != len(STATIC_BAND_NAMES):
            raise AssertionError(
                f"{record.path} has {dataset.RasterCount} bands; expected "
                f"{len(STATIC_BAND_NAMES)}."
            )
        bands = []
        for band_index, (band_name, intended_nodata) in enumerate(
            zip(record.band_names, record.nodata_values, strict=True),
            start=1,
        ):
            if band_name != STATIC_BAND_NAMES[band_index - 1]:
                raise AssertionError(
                    f"Band {band_index} is {band_name!r}; expected "
                    f"{STATIC_BAND_NAMES[band_index - 1]!r}."
                )
            band = dataset.GetRasterBand(band_index)
            array = band.ReadAsArray()
            if array is None:
                raise RuntimeError(
                    f"Could not read {record.path}, band {band_index}."
                )
            persisted_nodata = band.GetNoDataValue()
            persisted_nodata = (
                float(persisted_nodata) if persisted_nodata is not None else None
            )
            bands.append(
                analyze_written_band(
                    array=np.asarray(array),
                    band_name=band_name,
                    contract_index=band_index - 1,
                    source_details=source_contract.get(band_name, {}),
                    intended_output_nodata=intended_nodata,
                    persisted_nodata=persisted_nodata,
                    standardized_nodata=standardized_nodata,
                    known_output_sentinels=known_output_sentinels,
                )
            )
        return {
            "path": str(record.path),
            "zone": record.zone,
            "zoom_level": record.zoom_level,
            "tile_x": record.tile_x,
            "tile_y": record.tile_y,
            "bands": bands,
        }
    finally:
        dataset = None


def aggregate_bands(cubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for contract_index, band_name in enumerate(STATIC_BAND_NAMES):
        rows = [cube["bands"][contract_index] for cube in cubes]
        min_values = [
            row["scientific_valid_min"]
            for row in rows
            if row["scientific_valid_min"] is not None
        ]
        max_values = [
            row["scientific_valid_max"]
            for row in rows
            if row["scientific_valid_max"] is not None
        ]
        assessment_counts = Counter(row["assessment"] for row in rows)
        persisted_values = sorted(
            {format_value(row["persisted_geotiff_nodata"]) for row in rows}
        )
        summaries.append(
            {
                "contract_index": contract_index,
                "band_name": band_name,
                "tile_count": len(rows),
                "metadata_source_nodata": rows[0]["metadata_source_nodata"],
                "resolved_source_nodata": rows[0]["resolved_source_nodata"],
                "intended_output_nodata": rows[0]["intended_output_nodata"],
                "persisted_geotiff_nodata_values": persisted_values,
                "persisted_metadata_mismatch_tile_count": sum(
                    not row["persisted_metadata_matches_intended"] for row in rows
                ),
                "pixel_count": sum(row["pixel_count"] for row in rows),
                "nonfinite_pixel_count": sum(
                    row["nonfinite_pixel_count"] for row in rows
                ),
                "standardized_sentinel_count": sum(
                    row["standardized_sentinel_count"] for row in rows
                ),
                "intended_output_sentinel_count": sum(
                    row["intended_output_sentinel_count"] for row in rows
                ),
                "source_sentinel_leak_count": sum(
                    row["source_sentinel_leak_count"] for row in rows
                ),
                "alternate_output_sentinel_count": sum(
                    row["alternate_output_sentinel_count"] for row in rows
                ),
                "scientific_valid_min": min(min_values) if min_values else None,
                "scientific_valid_max": max(max_values) if max_values else None,
                "scientific_values_below_standardized_count": sum(
                    row["scientific_values_below_standardized_count"] for row in rows
                ),
                "assessment_counts": dict(sorted(assessment_counts.items())),
            }
        )
    return summaries


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print(
        "idx\tband_name\tsource_nodata\tintended_nodata\tpersisted_nodata"
        "\tstandard_count\tintended_count\tsource_leaks\tvalid_min\tvalid_max"
        "\tvalid_below_standard\tassessment"
    )
    for row in summaries:
        print(
            "\t".join(
                [
                    str(row["contract_index"]),
                    row["band_name"],
                    format_value(row["resolved_source_nodata"]),
                    format_value(row["intended_output_nodata"]),
                    ",".join(row["persisted_geotiff_nodata_values"]),
                    str(row["standardized_sentinel_count"]),
                    str(row["intended_output_sentinel_count"]),
                    str(row["source_sentinel_leak_count"]),
                    format_value(row["scientific_valid_min"]),
                    format_value(row["scientific_valid_max"]),
                    str(row["scientific_values_below_standardized_count"]),
                    ",".join(row["assessment_counts"]),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.standardized_nodata):
        raise ValueError("--standardized-nodata must be finite.")
    static_index = args.static_index or args.static_data_dir / "db2.shp"
    report_path = args.report_path or args.output_dir / "static_cube_nodata_report.json"
    if not args.static_data_dir.is_dir():
        raise FileNotFoundError(
            f"Static data directory does not exist: {args.static_data_dir}"
        )
    if not static_index.is_file():
        raise FileNotFoundError(f"Static vector index does not exist: {static_index}")

    static_source = build_static_source(
        data_dir=args.static_data_dir,
        index_path=static_index,
        index_layer=args.static_index_layer,
        location_field=args.location_field,
    )
    source_contract, duplicate_source_bands = read_source_band_contract(static_source)
    missing_source_bands = [
        name for name in STATIC_BAND_NAMES if name not in source_contract
    ]
    if duplicate_source_bands:
        raise ValueError(
            f"Duplicate canonical source bands in static index: {duplicate_source_bands}"
        )
    if missing_source_bands:
        raise ValueError(
            f"Canonical bands missing from static index: {missing_source_bands}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TileConfig(
        output_dir=args.output_dir,
        zoom_level=args.zoom_level,
        sources=(static_source,),
    )
    ul_lat, ul_lon, lr_lat, lr_lon = args.bounds
    records = create_tiles_for_aoi(
        config,
        ul_lat=ul_lat,
        ul_lon=ul_lon,
        lr_lat=lr_lat,
        lr_lon=lr_lon,
    )
    if not records:
        raise AssertionError("Static tiling produced no cube records.")

    known_output_sentinels = tuple(
        dict.fromkeys(
            float(value)
            for record in records
            for value in record.nodata_values
            if value is not None and math.isfinite(float(value))
        )
    )
    if args.standardized_nodata not in known_output_sentinels:
        known_output_sentinels = (
            args.standardized_nodata,
            *known_output_sentinels,
        )

    cubes = [
        inspect_cube(
            record,
            source_contract=source_contract,
            standardized_nodata=args.standardized_nodata,
            known_output_sentinels=known_output_sentinels,
        )
        for record in records
    ]
    summaries = aggregate_bands(cubes)

    metadata_mismatch_bands = [
        row["band_name"]
        for row in summaries
        if row["persisted_metadata_mismatch_tile_count"] > 0
    ]
    source_sentinel_leak_bands = [
        row["band_name"] for row in summaries if row["source_sentinel_leak_count"] > 0
    ]
    alternate_sentinel_bands = [
        row["band_name"]
        for row in summaries
        if row["alternate_output_sentinel_count"] > 0
    ]
    below_standardized_bands = [
        row["band_name"]
        for row in summaries
        if row["scientific_values_below_standardized_count"] > 0
    ]
    nonstandardized_bands = [
        row["band_name"]
        for row in summaries
        if not same_value(
            float(row["intended_output_nodata"]),
            args.standardized_nodata,
        )
    ]

    payload = {
        "status": "completed",
        "query": {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
            "zoom_level": args.zoom_level,
        },
        "static_data_dir": str(args.static_data_dir),
        "static_index": str(static_index),
        "standardized_nodata": args.standardized_nodata,
        "known_output_sentinels": list(known_output_sentinels),
        "cube_count": len(cubes),
        "band_count": len(STATIC_BAND_NAMES),
        "nonstandardized_intended_nodata_bands": nonstandardized_bands,
        "persisted_metadata_mismatch_bands": metadata_mismatch_bands,
        "source_sentinel_leak_bands": source_sentinel_leak_bands,
        "alternate_output_sentinel_bands": alternate_sentinel_bands,
        "scientific_values_below_standardized_bands": below_standardized_bands,
        "source_contract": {
            name: {
                key: json_value(value) if key.endswith("nodata") else value
                for key, value in details.items()
            }
            for name, details in source_contract.items()
        },
        "band_summary": summaries,
        "cubes": cubes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Static cube NoData diagnostic summary:")
    print_summary(summaries)
    print(f"\nCubes inspected: {len(cubes)}")
    print(f"Nonstandard intended-NoData bands: {nonstandardized_bands}")
    print(f"Persisted metadata mismatch bands: {metadata_mismatch_bands}")
    print(f"Source sentinel leak bands: {source_sentinel_leak_bands}")
    print(f"Alternate output sentinel bands: {alternate_sentinel_bands}")
    print(f"Scientific values below {args.standardized_nodata}: {below_standardized_bands}")
    print(f"JSON report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
