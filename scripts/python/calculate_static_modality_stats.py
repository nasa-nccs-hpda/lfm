#!/usr/bin/env python3
"""Compute copy/pasteable modality_info.yaml fields for WAC+STATIC chips.

The expected chip layout is 70 bands:
  - bands 0..4: WAC VIS
  - bands 5..6: WAC UV
  - bands 7..69: STATIC

This script computes only the 63 static-band mean/std values. It excludes only
nonfinite values, TIFF nodata, and exact configured nodata sentinels.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


COMMON_CUBE_NODATA = -3.40282265508890445e38
CPR_S1_SOURCE_NODATA = -3.4028230607370965e38
FLOAT32_MIN_NODATA = float(np.finfo(np.float32).min)

DEFAULT_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/"
    "fm_all_static_all_wac_iseg"
)
DEFAULT_OUTPUT_YAML = Path("scripts/outputs/static_modality_info_fields.yaml")


@dataclass
class StaticStats:
    count: np.ndarray
    extreme_count: np.ndarray
    extreme_value_counts: list[Counter[float]]
    total: np.ndarray
    total_sq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def init(cls, num_bands: int) -> "StaticStats":
        return cls(
            count=np.zeros(num_bands, dtype=np.int64),
            extreme_count=np.zeros(num_bands, dtype=np.int64),
            extreme_value_counts=[Counter() for _ in range(num_bands)],
            total=np.zeros(num_bands, dtype=np.float64),
            total_sq=np.zeros(num_bands, dtype=np.float64),
            minimum=np.full(num_bands, np.inf, dtype=np.float64),
            maximum=np.full(num_bands, -np.inf, dtype=np.float64),
        )

    def update(
        self,
        data: np.ndarray,
        *,
        raster_nodata: float | int | None,
        excluded_nodata_values: tuple[float, ...],
        extreme_nodata_threshold: float | None,
    ) -> None:
        invalid = ~np.isfinite(data)
        if raster_nodata is not None:
            invalid |= data == np.asarray(raster_nodata, dtype=data.dtype)
        for value in excluded_nodata_values:
            invalid |= data == np.asarray(value, dtype=data.dtype)
        if extreme_nodata_threshold is not None:
            extreme = np.abs(data.astype(np.float64, copy=False)) > extreme_nodata_threshold
            self.extreme_count += extreme.reshape(extreme.shape[0], -1).sum(axis=1)
            for band_index in range(data.shape[0]):
                band_extreme_values = data[band_index][extreme[band_index]]
                if band_extreme_values.size == 0:
                    continue
                values, counts = np.unique(
                    band_extreme_values.astype(np.float64, copy=False),
                    return_counts=True,
                )
                self.extreme_value_counts[band_index].update(
                    {
                        float(value): int(count)
                        for value, count in zip(values, counts, strict=True)
                    }
                )
            invalid |= extreme

        valid = ~invalid
        valid_count = valid.reshape(valid.shape[0], -1).sum(axis=1)
        data64 = np.where(valid, data, 0).astype(np.float64, copy=False)

        self.count += valid_count
        self.total += data64.reshape(data64.shape[0], -1).sum(axis=1)
        self.total_sq += np.square(data64).reshape(data64.shape[0], -1).sum(axis=1)

        for band_index in range(data.shape[0]):
            if valid_count[band_index] == 0:
                continue
            values = data[band_index][valid[band_index]].astype(np.float64, copy=False)
            self.minimum[band_index] = min(self.minimum[band_index], float(values.min()))
            self.maximum[band_index] = max(self.maximum[band_index], float(values.max()))

    def merge(self, other: "StaticStats") -> None:
        self.count += other.count
        self.extreme_count += other.extreme_count
        self.total += other.total
        self.total_sq += other.total_sq
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)
        for self_counts, other_counts in zip(
            self.extreme_value_counts,
            other.extreme_value_counts,
            strict=True,
        ):
            self_counts.update(other_counts)

    def mean(self) -> np.ndarray:
        return np.divide(
            self.total,
            self.count,
            out=np.full_like(self.total, np.nan, dtype=np.float64),
            where=self.count > 0,
        )

    def std(self) -> np.ndarray:
        mean = self.mean()
        variance = np.divide(
            self.total_sq,
            self.count,
            out=np.full_like(self.total_sq, np.nan, dtype=np.float64),
            where=self.count > 0,
        ) - np.square(mean)
        variance = np.maximum(variance, 0.0)
        return np.sqrt(variance)

    def bad_extreme_indexes(self, threshold: float) -> dict[str, list[int]]:
        arrays = {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean(),
            "std": self.std(),
        }
        return {
            name: np.where(np.abs(values) > threshold)[0].tolist()
            for name, values in arrays.items()
        }

    def format_extreme_value_report(
        self,
        *,
        channel_names: list[str],
        threshold: float,
        tail_count: int,
    ) -> list[str]:
        lines: list[str] = []
        for band_index, counts in enumerate(self.extreme_value_counts):
            if not counts:
                continue

            negative_values = {
                value: count for value, count in counts.items() if value < -threshold
            }
            positive_values = {
                value: count for value, count in counts.items() if value > threshold
            }
            smallest = sorted(counts.items(), key=lambda item: item[0])[:tail_count]
            largest = sorted(counts.items(), key=lambda item: item[0])[-tail_count:]

            lines.extend(
                [
                    f"{channel_names[band_index]}:",
                    f"  extreme_pixels: {sum(counts.values())}",
                    f"  unique_values_lt_negative_threshold: {len(negative_values)}",
                    f"  unique_values_gt_positive_threshold: {len(positive_values)}",
                    "  smallest_extreme_values:",
                ]
            )
            lines.extend(
                f"    {format_float(value)}: {count}" for value, count in smallest
            )
            lines.append("  largest_extreme_values:")
            lines.extend(
                f"    {format_float(value)}: {count}" for value, count in largest
            )
        return lines


def parse_csv_floats(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_band_range(value: str) -> range:
    if ":" not in value:
        raise argparse.ArgumentTypeError("band range must use START:STOP syntax")
    start_text, stop_text = value.split(":", 1)
    start = int(start_text)
    stop = int(stop_text)
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("band range must satisfy 0 <= START < STOP")
    return range(start, stop)


def iter_tifs(data_dir: Path, image_glob: str, max_files: int | None) -> list[Path]:
    paths = sorted(data_dir.glob(image_glob))
    paths = [path for path in paths if path.suffix.lower() in {".tif", ".tiff"}]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No GeoTIFF files matched {data_dir / image_glob}")
    return paths


def process_one_chip(task: tuple) -> tuple[int, Path, StaticStats]:
    (
        index,
        path,
        rasterio_indexes,
        num_bands,
        excluded_nodata_values,
        extreme_nodata_threshold,
    ) = task

    import rasterio

    with rasterio.open(path) as src:
        if src.count < max(rasterio_indexes):
            raise ValueError(
                f"{path} has {src.count} bands, but static band range "
                f"requires band {max(rasterio_indexes)}."
            )
        data = src.read(rasterio_indexes)
        raster_nodata = src.nodata

    stats = StaticStats.init(num_bands)
    stats.update(
        data,
        raster_nodata=raster_nodata,
        excluded_nodata_values=excluded_nodata_values,
        extreme_nodata_threshold=extreme_nodata_threshold,
    )
    return index, path, stats


def format_float(value: float) -> str:
    if math.isnan(value):
        return ".nan"
    if math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    return format(float(value), ".17g")


def yaml_list(values: np.ndarray, indent: str) -> list[str]:
    return [f"{indent}- {format_float(float(value))}" for value in values]


def write_static_yaml(
    output_yaml: Path,
    *,
    stats: StaticStats,
    channel_names: list[str],
    patch_size: int,
    input_size: int,
    image_size: int | None,
    gsd_m: float | None,
) -> None:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Copy this top-level block into modality_info.yaml.",
        "# The existing vis and uv blocks should remain unchanged.",
        "static:",
        "  encoder_embedding: terramind.models.encoder_embeddings.ImageEncoderEmbedding",
        "  encoder_kwargs: {}",
        "  decoder_embedding: null",
        "  decoder_kwargs: {}",
        "  type: img",
        "  pretokenized: false",
        "  data_range: null",
        "  one_hot_encoding: null",
        "  path: static_numpy",
        "  min_tokens: 0",
        "  max_tokens: 256",
        f"  input_size: {input_size}",
        "  pre_resize: 512",
        "  crop_settings:",
        "  - - 0",
        "    - 0",
        f"    - {input_size}",
        f"    - {input_size}",
        "  keep: null",
        "  stats:",
        "    static:",
        "      min:",
        *yaml_list(stats.minimum, "      "),
        "      max:",
        *yaml_list(stats.maximum, "      "),
        "      mean:",
        *yaml_list(stats.mean(), "      "),
        "      std:",
        *yaml_list(stats.std(), "      "),
        "      channels:",
        *[f"      - {name}" for name in channel_names],
        f"      image_size: {image_size if image_size is not None else input_size}",
        f"      gsd_m: {format_float(gsd_m) if gsd_m is not None else 'null'}",
        f"  num_channels: {len(channel_names)}",
        f"  patch_size: {patch_size}",
    ]
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 63-band static mean/std values from 70-band WAC+STATIC chips "
            "and write a copy/pasteable modality_info.yaml block."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument(
        "--static-band-range",
        type=parse_band_range,
        default=parse_band_range("7:70"),
        help="0-based static band range as START:STOP. Default: 7:70.",
    )
    parser.add_argument(
        "--excluded-nodata-values",
        type=parse_csv_floats,
        default=(COMMON_CUBE_NODATA, CPR_S1_SOURCE_NODATA, FLOAT32_MIN_NODATA),
        help=(
            "Comma-separated exact NoData sentinels to exclude. Use "
            "--excluded-nodata-values=VALUE1,VALUE2 for negative scientific notation."
        ),
    )
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--extreme-nodata-threshold",
        type=float,
        default=1.0e30,
        help=(
            "Exclude values whose absolute magnitude exceeds this threshold before "
            "computing stats. Default 1e30 catches float32 NoData/resampling artifacts "
            "without filtering physically plausible negative static values. Pass a "
            "negative value to disable."
        ),
    )
    parser.add_argument(
        "--max-abs-stat",
        type=float,
        default=1.0e30,
        help=(
            "Fail if any generated min/max/mean/std exceeds this absolute value. "
            "This catches leaked e38 NoData sentinels."
        ),
    )
    parser.add_argument(
        "--extreme-report-tail-count",
        type=int,
        default=5,
        help="Number of smallest/largest unique extreme values to print per band.",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gsd-m", type=float, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for per-chip statistics. Use 1 for serial execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    band_range = list(args.static_band_range)
    paths = iter_tifs(args.data_dir, args.image_glob, args.max_files)
    stats = StaticStats.init(len(band_range))
    extreme_nodata_threshold = (
        None if args.extreme_nodata_threshold < 0 else args.extreme_nodata_threshold
    )

    rasterio_indexes = [band + 1 for band in band_range]
    tasks = [
        (
            index,
            path,
            rasterio_indexes,
            len(band_range),
            args.excluded_nodata_values,
            extreme_nodata_threshold,
        )
        for index, path in enumerate(paths, start=1)
    ]

    if args.max_workers <= 1:
        iterator = tqdm(
            tasks,
            total=len(tasks),
            desc="Static modality stats",
            unit="chip",
            file=sys.stdout,
        )
        for task in iterator:
            index, path, partial_stats = process_one_chip(task)
            stats.merge(partial_stats)
            iterator.set_postfix_str(path.name)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_one_chip, task) for task in tasks]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Static modality stats",
                unit="chip",
                file=sys.stdout,
            )
            for future in iterator:
                index, path, partial_stats = future.result()
                stats.merge(partial_stats)
                iterator.set_postfix_str(path.name)

    bad_extremes = stats.bad_extreme_indexes(args.max_abs_stat)
    bad_extremes = {name: indexes for name, indexes in bad_extremes.items() if indexes}
    if bad_extremes:
        raise ValueError(
            "Generated static stats still contain extreme values above "
            f"{args.max_abs_stat:g}: {bad_extremes}. Check excluded NoData values."
        )

    channel_names = [f"static_{band:02d}" for band in band_range]
    write_static_yaml(
        args.output_yaml,
        stats=stats,
        channel_names=channel_names,
        patch_size=args.patch_size,
        input_size=args.input_size,
        image_size=args.image_size,
        gsd_m=args.gsd_m,
    )

    print(f"Wrote static modality YAML fields: {args.output_yaml}")
    print("Static valid counts per band:", stats.count.tolist())
    print("Static extreme-value exclusions per band:", stats.extreme_count.tolist())
    if extreme_nodata_threshold is not None and int(stats.extreme_count.sum()) > 0:
        print("Static extreme value-count report:")
        for line in stats.format_extreme_value_report(
            channel_names=channel_names,
            threshold=extreme_nodata_threshold,
            tail_count=args.extreme_report_tail_count,
        ):
            print(line)
    print("Static means:", stats.mean().tolist())
    print("Static stds:", stats.std().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
