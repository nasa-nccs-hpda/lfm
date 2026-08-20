#!/usr/bin/env python3
"""Count exact values in selected static chip bands.

By default this counts only sentinel-scale affected values, because full exact
value counts for continuous resampled rasters can be very large. Use
``--include-all-values`` when the full distribution is needed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/"
    "fm_all_static_all_wac_iseg/train/chips"
)
DEFAULT_OUT_DIR = Path("scripts/outputs/static_band_value_counts")
DEFAULT_STATIC_BANDS = (8, 9)
DEFAULT_EXTREME_THRESHOLD = 1.0e30


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def iter_tifs(data_dir: Path, image_glob: str, max_files: int | None) -> list[Path]:
    paths = sorted(data_dir.glob(image_glob))
    paths = [path for path in paths if path.suffix.lower() in {".tif", ".tiff"}]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No GeoTIFF files matched {data_dir / image_glob}")
    return paths


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(float(value), ".17g")


def float32_bits_to_value(bits: int) -> float:
    arr = np.asarray([bits], dtype=np.uint32)
    return float(arr.view(np.float32)[0])


def classify_value(value: float, threshold: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "positive_inf" if value > 0 else "negative_inf"
    if value > threshold:
        return "positive_extreme"
    if value < -threshold:
        return "negative_extreme"
    return "normal"


def count_band_values(
    paths: list[Path],
    *,
    static_band: int,
    include_all_values: bool,
    extreme_threshold: float,
) -> tuple[Counter[int], int]:
    import rasterio

    counts: Counter[int] = Counter()
    rasterio_band_index = static_band + 1
    total_pixels_scanned = 0

    for index, path in enumerate(paths, start=1):
        with rasterio.open(path) as src:
            if src.count < rasterio_band_index:
                raise ValueError(
                    f"{path} has {src.count} bands, but static_{static_band:02d} "
                    f"requires raster band {rasterio_band_index}."
                )
            band = src.read(rasterio_band_index)

        band = np.asarray(band, dtype=np.float32)
        total_pixels_scanned += int(band.size)
        if not include_all_values:
            band = band[np.abs(band.astype(np.float64, copy=False)) > extreme_threshold]
            if band.size == 0:
                if index == 1 or index % 25 == 0 or index == len(paths):
                    print(
                        f"Processed {index}/{len(paths)} chips for "
                        f"static_{static_band:02d}: {path}"
                    )
                continue

        bit_values, value_counts = np.unique(
            band.view(np.uint32).reshape(-1),
            return_counts=True,
        )
        counts.update(
            {int(bits): int(count) for bits, count in zip(bit_values, value_counts)}
        )

        if index == 1 or index % 25 == 0 or index == len(paths):
            print(
                f"Processed {index}/{len(paths)} chips for "
                f"static_{static_band:02d}: {path}"
            )

    return counts, total_pixels_scanned


def summarize_counts(
    counts: Counter[int],
    *,
    total_pixels_scanned: int,
    extreme_threshold: float,
    top_n: int,
) -> dict[str, Any]:
    rows = [
        (float32_bits_to_value(bits), bits, count)
        for bits, count in counts.items()
    ]
    negative_extreme = [
        (value, bits, count) for value, bits, count in rows if value < -extreme_threshold
    ]
    positive_extreme = [
        (value, bits, count) for value, bits, count in rows if value > extreme_threshold
    ]

    rows_by_count = sorted(rows, key=lambda item: item[2], reverse=True)
    rows_by_value = sorted(rows, key=lambda item: item[0])

    return {
        "total_pixels_scanned": total_pixels_scanned,
        "counted_pixels": int(sum(counts.values())),
        "unique_counted_values": len(counts),
        "unique_values_lt_negative_threshold": len(negative_extreme),
        "unique_values_gt_positive_threshold": len(positive_extreme),
        "top_values_by_count": [
            value_count_record(value, bits, count, extreme_threshold)
            for value, bits, count in rows_by_count[:top_n]
        ],
        "smallest_values": [
            value_count_record(value, bits, count, extreme_threshold)
            for value, bits, count in rows_by_value[:top_n]
        ],
        "largest_values": [
            value_count_record(value, bits, count, extreme_threshold)
            for value, bits, count in rows_by_value[-top_n:]
        ],
    }


def value_count_record(
    value: float,
    bits: int,
    count: int,
    extreme_threshold: float,
) -> dict[str, Any]:
    return {
        "value": value,
        "value_text": format_float(value),
        "float32_bits_hex": f"0x{bits:08x}",
        "count": int(count),
        "category": classify_value(value, extreme_threshold),
    }


def write_counts_tsv(
    path: Path,
    counts: Counter[int],
    *,
    extreme_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (
            (float32_bits_to_value(bits), bits, count)
            for bits, count in counts.items()
        ),
        key=lambda item: item[0],
    )
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("value\tfloat32_bits_hex\tcount\tcategory\n")
        for value, bits, count in rows:
            f.write(
                "\t".join(
                    [
                        format_float(value),
                        f"0x{bits:08x}",
                        str(int(count)),
                        classify_value(value, extreme_threshold),
                    ]
                )
                + "\n"
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count exact float32 values for selected static chip bands."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument(
        "--static-bands",
        type=parse_int_csv,
        default=DEFAULT_STATIC_BANDS,
        help="Comma-separated 0-based chip/static band numbers. Default: 8,9.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=DEFAULT_EXTREME_THRESHOLD,
        help="Affected-value threshold. Default: 1e30.",
    )
    parser.add_argument(
        "--include-all-values",
        action="store_true",
        help="Count every exact value, not just abs(value) > extreme threshold.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = iter_tifs(args.data_dir, args.image_glob, args.max_files)
    summary: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "image_glob": args.image_glob,
        "files_scanned": len(paths),
        "include_all_values": args.include_all_values,
        "extreme_threshold": args.extreme_threshold,
        "bands": {},
    }

    for static_band in args.static_bands:
        counts, total_pixels_scanned = count_band_values(
            paths,
            static_band=static_band,
            include_all_values=args.include_all_values,
            extreme_threshold=args.extreme_threshold,
        )
        band_name = f"static_{static_band:02d}"
        tsv_path = args.out_dir / f"{band_name}_value_counts.tsv"
        write_counts_tsv(tsv_path, counts, extreme_threshold=args.extreme_threshold)

        band_summary = summarize_counts(
            counts,
            total_pixels_scanned=total_pixels_scanned,
            extreme_threshold=args.extreme_threshold,
            top_n=args.top_n,
        )
        band_summary["value_counts_tsv"] = str(tsv_path)
        summary["bands"][band_name] = band_summary

        print(
            f"{band_name}: counted {band_summary['counted_pixels']} pixels "
            f"across {band_summary['unique_counted_values']} unique values. "
            f"TSV: {tsv_path}"
        )

    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Wrote summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
