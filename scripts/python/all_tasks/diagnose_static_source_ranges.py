#!/usr/bin/env python3
"""Sample source rasters to diagnose valid ranges and NoData collisions.

This script reads the raster paths declared by the static vector index. It
samples deterministic, non-overlapping windows across every raster band rather
than loading lunar-scale rasters in full. A pixel is considered valid when it
is finite, is not equal to that band's native NoData metadata, and is valid in
the GDAL mask band.

The resulting ranges are sampled ranges, not guaranteed full-dataset extrema.
Candidate values are counted both before and after applying the source band's
own validity rules. A nonzero ``valid_count`` for a candidate means that using
that value as a shared output NoData sentinel would collide with sampled valid
source data for that band.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model.static_band_contract import (  # noqa: E402
    MINIRF_SOURCE_NODATA,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
)
from lfm.model.vector_index import (  # noqa: E402
    open_vector_layer,
    resolve_indexed_raster_path,
)


DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")
DEFAULT_CANDIDATES = (
    STATIC_OUTPUT_NODATA,
    -32767.0,
    MINIRF_SOURCE_NODATA,
)
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample all indexed static source bands and report per-band valid "
            "ranges and candidate NoData collisions."
        )
    )
    parser.add_argument("--static-data-dir", type=Path, default=DEFAULT_STATIC_DATA_DIR)
    parser.add_argument(
        "--static-index",
        type=Path,
        default=None,
        help="Default: STATIC_DATA_DIR/db2.shp.",
    )
    parser.add_argument("--index-layer", default=None)
    parser.add_argument("--location-field", default="location")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/static_source_range_diagnostic"),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Native-resolution width and height of each sampled window.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help=(
            "Maximum windows per raster axis. The default samples at most "
            "8x8 windows (4,194,304 pixels) per band."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=float,
        dest="candidates",
        help=(
            "Candidate NoData value to test; repeat for multiple values. "
            "Defaults to -32768, -32767, and the Mini-RF float32 sentinel. "
            "Use --candidate=-3.4e38 syntax for negative scientific notation."
        ),
    )
    parser.add_argument(
        "--name-regex",
        default=None,
        help="Only inspect source paths or band names matching this regex.",
    )
    parser.add_argument(
        "--max-rasters",
        type=int,
        default=None,
        help="Inspect only the first N deduplicated index paths (for smoke tests).",
    )
    return parser.parse_args()


def _require_positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    return format(float(value), ".17g")


def sampling_offsets(length: int, window_size: int, grid_size: int) -> list[int]:
    """Return deterministic offsets whose windows do not overlap."""
    size = min(length, window_size)
    if length <= size:
        return [0]
    count = min(grid_size, max(1, length // size))
    if count == 1:
        return [(length - size) // 2]
    return [
        int(round(index * (length - size) / (count - 1)))
        for index in range(count)
    ]


def sample_windows(
    width: int,
    height: int,
    window_size: int,
    grid_size: int,
) -> list[tuple[int, int, int, int]]:
    x_size = min(width, window_size)
    y_size = min(height, window_size)
    return [
        (x_offset, y_offset, x_size, y_size)
        for y_offset in sampling_offsets(height, window_size, grid_size)
        for x_offset in sampling_offsets(width, window_size, grid_size)
    ]


def candidate_as_dtype(candidate: float, dtype: np.dtype) -> Any | None:
    """Return an exactly representable comparison scalar, or None."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not candidate.is_integer() or candidate < info.min or candidate > info.max:
            return None
        return dtype.type(int(candidate))
    if np.issubdtype(dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            converted = dtype.type(candidate)
        return converted if np.isfinite(converted) else None
    return None


def nodata_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.zeros(data.shape, dtype=bool)
    if math.isnan(nodata):
        return np.isnan(data)
    converted = candidate_as_dtype(float(nodata), data.dtype)
    if converted is None:
        return np.zeros(data.shape, dtype=bool)
    return data == converted


def _band_name(dataset: Any, path: Path, band_index: int) -> str:
    band = dataset.GetRasterBand(band_index)
    return (
        band.GetMetadataItem("Name")
        or band.GetDescription()
        or (path.stem if dataset.RasterCount == 1 else f"{path.stem}-{band_index - 1}")
    )


def indexed_raster_paths(
    *,
    data_dir: Path,
    index_path: Path,
    index_layer: str | None,
    location_field: str,
) -> list[Path]:
    dataset, layer = open_vector_layer(index_path, layer_name=index_layer)
    try:
        definition = layer.GetLayerDefn()
        if definition.GetFieldIndex(location_field) < 0:
            fields = [
                definition.GetFieldDefn(index).GetName()
                for index in range(definition.GetFieldCount())
            ]
            raise KeyError(
                f"Index {index_path} has no {location_field!r} field; "
                f"available fields: {fields}"
            )
        layer.ResetReading()
        paths = [
            resolve_indexed_raster_path(data_dir, feature.GetField(location_field))
            for feature in layer
        ]
    finally:
        layer = None
        dataset = None
    return sorted(set(paths), key=lambda path: str(path).lower())


def inspect_band(
    *,
    dataset: Any,
    path: Path,
    band_index: int,
    band_name: str,
    windows: list[tuple[int, int, int, int]],
    candidates: tuple[float, ...],
) -> dict[str, Any]:
    from osgeo import gdal

    band = dataset.GetRasterBand(band_index)
    mask_band = band.GetMaskBand()
    mask_flags = int(band.GetMaskFlags())
    native_nodata = band.GetNoDataValue()
    native_nodata = float(native_nodata) if native_nodata is not None else None

    valid_chunks: list[np.ndarray] = []
    total_pixels = 0
    nonfinite_pixels = 0
    native_nodata_pixels = 0
    masked_pixels = 0
    candidate_counts = {
        _format_float(candidate): {
            "requested_value": candidate,
            "value_as_band_dtype": None,
            "representable": None,
            "raw_count": 0,
            "valid_count": 0,
        }
        for candidate in candidates
    }

    dtype_name: str | None = None
    for x_offset, y_offset, x_size, y_size in windows:
        data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)
        if data is None:
            raise RuntimeError(
                f"Could not read band {band_index} window from {path}: "
                f"({x_offset}, {y_offset}, {x_size}, {y_size})"
            )
        data = np.asarray(data)
        dtype_name = str(data.dtype)
        finite = np.isfinite(data)
        source_nodata = nodata_mask(data, native_nodata)
        if mask_flags == gdal.GMF_ALL_VALID:
            source_masked = np.zeros(data.shape, dtype=bool)
        else:
            mask = mask_band.ReadAsArray(x_offset, y_offset, x_size, y_size)
            if mask is None:
                raise RuntimeError(f"Could not read GDAL mask for band {band_index}: {path}")
            source_masked = np.asarray(mask) == 0
        valid = finite & ~source_nodata & ~source_masked

        total_pixels += int(data.size)
        nonfinite_pixels += int(np.count_nonzero(~finite))
        native_nodata_pixels += int(np.count_nonzero(source_nodata))
        masked_pixels += int(np.count_nonzero(source_masked))
        if np.any(valid):
            valid_chunks.append(data[valid].astype(np.float64, copy=False))

        for candidate in candidates:
            record = candidate_counts[_format_float(candidate)]
            converted = candidate_as_dtype(candidate, data.dtype)
            record["representable"] = converted is not None
            if converted is None:
                continue
            record["value_as_band_dtype"] = float(converted)
            matches = data == converted
            record["raw_count"] += int(np.count_nonzero(matches))
            record["valid_count"] += int(np.count_nonzero(matches & valid))

    if dtype_name is None:
        raise RuntimeError(f"No sample windows were generated for {path}, band {band_index}.")

    if valid_chunks:
        values = np.concatenate(valid_chunks)
        quantiles = np.quantile(values, QUANTILES)
        sampled_min = float(values.min())
        sampled_max = float(values.max())
        valid_pixels = int(values.size)
        quantile_values = {
            "p00_1": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "p99_9": float(quantiles[4]),
        }
    else:
        sampled_min = None
        sampled_max = None
        valid_pixels = 0
        quantile_values = {
            "p00_1": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "p99_9": None,
        }

    contract_index = (
        STATIC_BAND_NAMES.index(band_name) if band_name in STATIC_BAND_NAMES else None
    )
    return {
        "contract_index": contract_index,
        "band_name": band_name,
        "source_path": str(path),
        "source_band_index": band_index,
        "dtype": dtype_name,
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "native_nodata": native_nodata,
        "gdal_mask_flags": mask_flags,
        "cached_statistics_minimum": _float_or_none(
            band.GetMetadataItem("STATISTICS_MINIMUM")
        ),
        "cached_statistics_maximum": _float_or_none(
            band.GetMetadataItem("STATISTICS_MAXIMUM")
        ),
        "sample_window_count": len(windows),
        "sampled_pixels": total_pixels,
        "sampled_valid_pixels": valid_pixels,
        "sampled_valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        "sampled_nonfinite_pixels": nonfinite_pixels,
        "sampled_native_nodata_pixels": native_nodata_pixels,
        "sampled_masked_pixels": masked_pixels,
        "sampled_valid_min": sampled_min,
        **quantile_values,
        "sampled_valid_max": sampled_max,
        "candidate_counts": candidate_counts,
        "candidate_collision": any(
            record["valid_count"] > 0 for record in candidate_counts.values()
        ),
    }


def inspect_raster(
    path: Path,
    *,
    window_size: int,
    grid_size: int,
    candidates: tuple[float, ...],
    name_pattern: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    from osgeo import gdal, gdalconst

    dataset = gdal.Open(str(path), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open indexed static raster: {path}")
    try:
        windows = sample_windows(
            dataset.RasterXSize,
            dataset.RasterYSize,
            window_size,
            grid_size,
        )
        records = []
        for band_index in range(1, dataset.RasterCount + 1):
            band_name = _band_name(dataset, path, band_index)
            if name_pattern is not None and not (
                name_pattern.search(str(path)) or name_pattern.search(band_name)
            ):
                continue
            records.append(
                inspect_band(
                    dataset=dataset,
                    path=path,
                    band_index=band_index,
                    band_name=band_name,
                    windows=windows,
                    candidates=candidates,
                )
            )
        return records
    finally:
        dataset = None


def _record_sort_key(record: dict[str, Any]) -> tuple[int, str, int]:
    contract_index = record["contract_index"]
    return (
        contract_index if contract_index is not None else len(STATIC_BAND_NAMES),
        str(record["band_name"]).lower(),
        int(record["source_band_index"]),
    )


def write_tsv(
    path: Path,
    records: list[dict[str, Any]],
    candidates: tuple[float, ...],
) -> None:
    candidate_keys = [_format_float(candidate) for candidate in candidates]
    fields = [
        "contract_index",
        "band_name",
        "source_path",
        "source_band_index",
        "dtype",
        "width",
        "height",
        "native_nodata",
        "sample_window_count",
        "sampled_pixels",
        "sampled_valid_pixels",
        "sampled_valid_fraction",
        "sampled_valid_min",
        "p00_1",
        "p01",
        "p50",
        "p99",
        "p99_9",
        "sampled_valid_max",
        "candidate_collision",
    ]
    for key in candidate_keys:
        fields.extend(
            [
                f"candidate_{key}_raw_count",
                f"candidate_{key}_valid_count",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            for key in candidate_keys:
                counts = record["candidate_counts"][key]
                row[f"candidate_{key}_raw_count"] = counts["raw_count"]
                row[f"candidate_{key}_valid_count"] = counts["valid_count"]
            writer.writerow(row)


def print_summary(records: list[dict[str, Any]], candidates: tuple[float, ...]) -> None:
    candidate_labels = [_format_float(value) for value in candidates]
    headings = [
        "idx",
        "band_name",
        "dtype",
        "native_nodata",
        "valid_min",
        "valid_max",
        *[f"valid@{label}" for label in candidate_labels],
    ]
    print("\t".join(headings))
    for record in records:
        values = [
            "" if record["contract_index"] is None else str(record["contract_index"]),
            str(record["band_name"]),
            str(record["dtype"]),
            _format_float(record["native_nodata"]),
            _format_float(record["sampled_valid_min"]),
            _format_float(record["sampled_valid_max"]),
            *[
                str(record["candidate_counts"][label]["valid_count"])
                for label in candidate_labels
            ],
        ]
        print("\t".join(values))


def main() -> int:
    args = parse_args()
    window_size = _require_positive(args.window_size, "--window-size")
    grid_size = _require_positive(args.grid_size, "--grid-size")
    if args.max_rasters is not None:
        _require_positive(args.max_rasters, "--max-rasters")
    candidates = tuple(dict.fromkeys(args.candidates or DEFAULT_CANDIDATES))
    name_pattern = re.compile(args.name_regex, re.IGNORECASE) if args.name_regex else None
    static_index = args.static_index or args.static_data_dir / "db2.shp"

    paths = indexed_raster_paths(
        data_dir=args.static_data_dir,
        index_path=static_index,
        index_layer=args.index_layer,
        location_field=args.location_field,
    )
    if args.max_rasters is not None:
        paths = paths[: args.max_rasters]
    if not paths:
        raise FileNotFoundError(f"No raster paths found in static index: {static_index}")

    print(f"Static index: {static_index}")
    print(f"Unique indexed rasters: {len(paths)}")
    print(
        f"Sampling: up to {grid_size}x{grid_size} non-overlapping "
        f"{window_size}x{window_size} native-resolution windows per band"
    )
    print(f"Candidate values: {[float(value) for value in candidates]}")
    print()

    records: list[dict[str, Any]] = []
    for position, path in enumerate(paths, start=1):
        print(f"Inspecting raster {position}/{len(paths)}: {path}", flush=True)
        records.extend(
            inspect_raster(
                path,
                window_size=window_size,
                grid_size=grid_size,
                candidates=candidates,
                name_pattern=name_pattern,
            )
        )

    if not records:
        raise ValueError("No raster bands matched the requested diagnostic filters.")
    records.sort(key=_record_sort_key)

    observed_contract_names = {
        record["band_name"]
        for record in records
        if record["band_name"] in STATIC_BAND_NAMES
    }
    missing_contract_names = [
        name for name in STATIC_BAND_NAMES if name not in observed_contract_names
    ]
    extra_names = sorted(
        record["band_name"]
        for record in records
        if record["band_name"] not in STATIC_BAND_NAMES
    )
    collisions = [record for record in records if record["candidate_collision"]]

    payload = {
        "diagnostic_scope": "sampled_native_source_pixels",
        "static_data_dir": str(args.static_data_dir),
        "static_index": str(static_index),
        "index_layer": args.index_layer,
        "location_field": args.location_field,
        "raster_count": len(paths),
        "band_count": len(records),
        "window_size": window_size,
        "grid_size": grid_size,
        "candidates": list(candidates),
        "candidate_collision_band_count": len(collisions),
        "missing_contract_band_names": missing_contract_names,
        "extra_band_names": extra_names,
        "bands": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "static_source_range_report.json"
    tsv_path = args.output_dir / "static_source_ranges.tsv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_tsv(tsv_path, records, candidates)

    print("\nSampled valid ranges and candidate collision counts:")
    print_summary(records, candidates)
    print(f"\nCandidate collision bands: {len(collisions)}/{len(records)}")
    if collisions:
        for record in collisions:
            collided = [
                key
                for key, counts in record["candidate_counts"].items()
                if counts["valid_count"] > 0
            ]
            print(f"  {record['band_name']}: {', '.join(collided)}")
    if name_pattern is None:
        print(f"Missing canonical static bands: {len(missing_contract_names)}")
        print(f"Unexpected static bands: {len(extra_names)}")
    print(f"JSON report: {json_path}")
    print(f"TSV report: {tsv_path}")
    print(
        "NOTE: These are sampled ranges. Zero sampled collisions do not prove "
        "that a candidate is absent from the full source raster."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
