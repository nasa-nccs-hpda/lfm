"""Compare common instance dataset chips for NoData and extreme values.

The comparison is intentionally membership-tolerant: samples that exist in only
one dataset root are reported, but only common sample IDs are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


DEFAULT_BASELINE_ROOT = Path(
    "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/fm_all_static_all_wac_iseg"
)
DEFAULT_CANDIDATE_ROOT = Path(
    "/explore/nobackup/people/ajkerr1/Lunar_FM/instance_wac_static_chips_preserve_bilinear"
)
DEFAULT_KNOWN_NODATA_VALUES = (
    -32768.0,
    -3.40282265508890445e38,
    -3.4028230607370965e38,
    -3.4028234663852886e38,
)
SAMPLE_ID_SUFFIXES = (
    "_wac_static_chip",
    "_input_nac_chip",
    "_input_chip",
    "_static_chip",
    "_label",
    "_mask",
    "_chip",
    "_img",
)


def sample_id(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in sorted(SAMPLE_ID_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_files(
    directory: Path,
    patterns: tuple[str, ...],
    *,
    required: bool = False,
) -> dict[str, Path]:
    if not directory.exists():
        if required:
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        return {}
    indexed: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            key = sample_id(path)
            if key in indexed:
                duplicates.setdefault(key, [indexed[key]]).append(path)
                continue
            indexed[key] = path
    if duplicates:
        examples = "\n".join(
            f"{key}: {items[0]} and {items[1]}"
            for key, items in list(duplicates.items())[:5]
        )
        raise ValueError(f"Duplicate normalized sample IDs in {directory}:\n{examples}")
    return indexed


def parse_known_nodata_values(raw: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return tuple(values)


def numeric_summary(arr: np.ndarray, extreme_threshold: float) -> dict[str, Any]:
    flat = np.asarray(arr).reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    return {
        "nan_count": int(np.isnan(flat).sum()),
        "pos_inf_count": int(np.isposinf(flat).sum()),
        "neg_inf_count": int(np.isneginf(flat).sum()),
        "finite_count": int(finite.sum()),
        "negative_extreme_count": int((finite_values < -extreme_threshold).sum()),
        "positive_extreme_count": int((finite_values > extreme_threshold).sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else math.nan,
        "finite_max": float(finite_values.max()) if finite_values.size else math.nan,
    }


def exact_nodata_counts(
    arr: np.ndarray,
    known_nodata_values: tuple[float, ...],
) -> dict[str, int]:
    counts = {}
    arr_float32 = np.asarray(arr, dtype=np.float32)
    for value in known_nodata_values:
        value32 = np.float32(value)
        counts[f"{float(value32):.9e}"] = int((arr_float32 == value32).sum())
    return counts


def band_numeric_summary(
    arr: np.ndarray,
    known_nodata_values: tuple[float, ...],
    extreme_threshold: float,
) -> list[dict[str, Any]]:
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    rows = []
    for band_idx in range(arr.shape[0]):
        band = arr[band_idx]
        summary = numeric_summary(band, extreme_threshold)
        summary["band"] = band_idx + 1
        summary["known_nodata_counts"] = exact_nodata_counts(
            band,
            known_nodata_values,
        )
        rows.append(summary)
    return rows


def chip_summary(
    path: Path,
    known_nodata_values: tuple[float, ...],
    extreme_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with rasterio.open(path) as src:
        arr = src.read()
        sample = numeric_summary(arr, extreme_threshold)
        sample.update(
            {
                "path": str(path),
                "shape": "x".join(str(x) for x in arr.shape),
                "dtype": str(arr.dtype),
                "band_count": int(src.count),
                "width": int(src.width),
                "height": int(src.height),
                "crs": str(src.crs) if src.crs else "",
                "nodata_metadata": (
                    "" if src.nodata is None else f"{float(src.nodata):.9e}"
                ),
                "known_nodata_counts": exact_nodata_counts(
                    arr,
                    known_nodata_values,
                ),
            }
        )
        bands = band_numeric_summary(arr, known_nodata_values, extreme_threshold)
    return sample, bands


def load_label(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if "mask" in data:
                return data["mask"]
            if data.files:
                return data[data.files[0]]
            raise KeyError(f"{path} has no arrays")
    raise ValueError(f"Unsupported label file: {path}")


def label_summary(path: Path) -> dict[str, Any]:
    arr = load_label(path)
    foreground = arr > 0
    return {
        "path": str(path),
        "shape": "x".join(str(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "foreground_pixels": int(foreground.sum()),
        "unique_values": int(np.unique(arr).size),
        "max_value": int(arr.max()) if arr.size else 0,
    }


def add_chip_aggregate(
    aggregate: dict[str, Any],
    sample: dict[str, Any],
    band_rows: list[dict[str, Any]],
) -> None:
    aggregate["samples"] += 1
    for key in (
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
        "negative_extreme_count",
        "positive_extreme_count",
        "finite_count",
    ):
        aggregate[key] += int(sample[key])
    for key, value in sample["known_nodata_counts"].items():
        aggregate["known_nodata_counts"][key] += int(value)
    finite_min = sample["finite_min"]
    finite_max = sample["finite_max"]
    if not math.isnan(finite_min):
        aggregate["finite_min"] = min(aggregate["finite_min"], finite_min)
    if not math.isnan(finite_max):
        aggregate["finite_max"] = max(aggregate["finite_max"], finite_max)

    for row in band_rows:
        band = int(row["band"])
        band_agg = aggregate["bands"][band]
        band_agg["samples"] += 1
        for key in (
            "nan_count",
            "pos_inf_count",
            "neg_inf_count",
            "negative_extreme_count",
            "positive_extreme_count",
            "finite_count",
        ):
            band_agg[key] += int(row[key])
        for key, value in row["known_nodata_counts"].items():
            band_agg["known_nodata_counts"][key] += int(value)


def make_chip_aggregate() -> dict[str, Any]:
    return {
        "samples": 0,
        "nan_count": 0,
        "pos_inf_count": 0,
        "neg_inf_count": 0,
        "negative_extreme_count": 0,
        "positive_extreme_count": 0,
        "finite_count": 0,
        "finite_min": math.inf,
        "finite_max": -math.inf,
        "known_nodata_counts": Counter(),
        "bands": defaultdict(make_band_aggregate),
    }


def make_band_aggregate() -> dict[str, Any]:
    return {
        "samples": 0,
        "nan_count": 0,
        "pos_inf_count": 0,
        "neg_inf_count": 0,
        "negative_extreme_count": 0,
        "positive_extreme_count": 0,
        "finite_count": 0,
        "known_nodata_counts": Counter(),
    }


def serializable_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    result = dict(aggregate)
    result["finite_min"] = None if result["finite_min"] == math.inf else result["finite_min"]
    result["finite_max"] = None if result["finite_max"] == -math.inf else result["finite_max"]
    result["known_nodata_counts"] = dict(result["known_nodata_counts"])
    result["bands"] = {
        str(band): {
            **dict(values),
            "known_nodata_counts": dict(values["known_nodata_counts"]),
        }
        for band, values in sorted(result["bands"].items())
    }
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def compare_split(
    *,
    split: str,
    baseline_root: Path,
    candidate_root: Path,
    known_nodata_values: tuple[float, ...],
    extreme_threshold: float,
    max_samples: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_chips_dir = baseline_root / split / "chips"
    candidate_chips_dir = candidate_root / split / "chips"
    baseline_labels_dir = baseline_root / split / "labels"
    candidate_labels_dir = candidate_root / split / "labels"
    baseline_chips = index_files(baseline_chips_dir, ("*.tif", "*.tiff"))
    candidate_chips = index_files(candidate_chips_dir, ("*.tif", "*.tiff"))
    baseline_labels = index_files(baseline_labels_dir, ("*.npz", "*.npy"))
    candidate_labels = index_files(candidate_labels_dir, ("*.npz", "*.npy"))

    baseline_ids = set(baseline_chips)
    candidate_ids = set(candidate_chips)
    common_ids = sorted(baseline_ids & candidate_ids)
    if max_samples is not None:
        common_ids = common_ids[:max_samples]

    split_summary = {
        "split": split,
        "baseline_chips_dir": str(baseline_chips_dir),
        "candidate_chips_dir": str(candidate_chips_dir),
        "baseline_labels_dir": str(baseline_labels_dir),
        "candidate_labels_dir": str(candidate_labels_dir),
        "baseline_chips_dir_exists": baseline_chips_dir.exists(),
        "candidate_chips_dir_exists": candidate_chips_dir.exists(),
        "baseline_labels_dir_exists": baseline_labels_dir.exists(),
        "candidate_labels_dir_exists": candidate_labels_dir.exists(),
        "baseline_chip_count": len(baseline_chips),
        "candidate_chip_count": len(candidate_chips),
        "common_chip_count": len(baseline_ids & candidate_ids),
        "evaluated_common_chip_count": len(common_ids),
        "baseline_only_count": len(baseline_ids - candidate_ids),
        "candidate_only_count": len(candidate_ids - baseline_ids),
        "baseline_only_examples": sorted(baseline_ids - candidate_ids)[:10],
        "candidate_only_examples": sorted(candidate_ids - baseline_ids)[:10],
        "baseline_label_count": len(baseline_labels),
        "candidate_label_count": len(candidate_labels),
    }

    chip_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    aggregate = {
        "baseline": make_chip_aggregate(),
        "candidate": make_chip_aggregate(),
    }

    for key in common_ids:
        for dataset_name, chip_index in (
            ("baseline", baseline_chips),
            ("candidate", candidate_chips),
        ):
            sample, bands = chip_summary(
                chip_index[key],
                known_nodata_values,
                extreme_threshold,
            )
            sample["split"] = split
            sample["dataset"] = dataset_name
            sample["sample_id"] = key
            chip_rows.append(sample)
            add_chip_aggregate(aggregate[dataset_name], sample, bands)

        baseline_label = baseline_labels.get(key)
        candidate_label = candidate_labels.get(key)
        if baseline_label is None or candidate_label is None:
            missing_rows.append(
                {
                    "split": split,
                    "sample_id": key,
                    "issue": "missing_common_label",
                    "baseline_label": str(baseline_label) if baseline_label else "",
                    "candidate_label": str(candidate_label) if candidate_label else "",
                }
            )
            continue

        baseline_label_summary = label_summary(baseline_label)
        candidate_label_summary = label_summary(candidate_label)
        baseline_arr = load_label(baseline_label)
        candidate_arr = load_label(candidate_label)
        same_shape = baseline_arr.shape == candidate_arr.shape
        exact_equal = bool(same_shape and np.array_equal(baseline_arr, candidate_arr))
        label_rows.append(
            {
                "split": split,
                "sample_id": key,
                "same_shape": same_shape,
                "exact_equal": exact_equal,
                "baseline_path": baseline_label_summary["path"],
                "candidate_path": candidate_label_summary["path"],
                "baseline_shape": baseline_label_summary["shape"],
                "candidate_shape": candidate_label_summary["shape"],
                "baseline_foreground_pixels": baseline_label_summary["foreground_pixels"],
                "candidate_foreground_pixels": candidate_label_summary["foreground_pixels"],
                "baseline_unique_values": baseline_label_summary["unique_values"],
                "candidate_unique_values": candidate_label_summary["unique_values"],
                "baseline_max_value": baseline_label_summary["max_value"],
                "candidate_max_value": candidate_label_summary["max_value"],
            }
        )

    split_summary["chip_aggregate"] = {
        name: serializable_aggregate(value)
        for name, value in aggregate.items()
    }
    split_summary["common_missing_label_count"] = len(missing_rows)
    return split_summary, chip_rows, label_rows, missing_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/instance_chip_regression"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--known-nodata-values",
        default=",".join(str(value) for value in DEFAULT_KNOWN_NODATA_VALUES),
    )
    parser.add_argument("--extreme-threshold", type=float, default=1e30)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split cap for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    known_nodata_values = parse_known_nodata_values(args.known_nodata_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "baseline_root": str(args.baseline_root),
        "candidate_root": str(args.candidate_root),
        "known_nodata_values": known_nodata_values,
        "extreme_threshold": args.extreme_threshold,
        "max_samples": args.max_samples,
        "splits": [],
    }
    all_chip_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_missing_rows: list[dict[str, Any]] = []

    for split in args.splits:
        print(f"Comparing split: {split}", flush=True)
        split_summary, chip_rows, label_rows, missing_rows = compare_split(
            split=split,
            baseline_root=args.baseline_root,
            candidate_root=args.candidate_root,
            known_nodata_values=known_nodata_values,
            extreme_threshold=args.extreme_threshold,
            max_samples=args.max_samples,
        )
        summary["splits"].append(split_summary)
        all_chip_rows.extend(chip_rows)
        all_label_rows.extend(label_rows)
        all_missing_rows.extend(missing_rows)
        print(
            f"  common={split_summary['common_chip_count']} "
            f"evaluated={split_summary['evaluated_common_chip_count']} "
            f"baseline_only={split_summary['baseline_only_count']} "
            f"candidate_only={split_summary['candidate_only_count']}",
            flush=True,
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    chip_fields = [
        "split",
        "dataset",
        "sample_id",
        "path",
        "shape",
        "dtype",
        "band_count",
        "width",
        "height",
        "crs",
        "nodata_metadata",
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
        "finite_count",
        "negative_extreme_count",
        "positive_extreme_count",
        "finite_min",
        "finite_max",
        "known_nodata_counts",
    ]
    label_fields = [
        "split",
        "sample_id",
        "same_shape",
        "exact_equal",
        "baseline_path",
        "candidate_path",
        "baseline_shape",
        "candidate_shape",
        "baseline_foreground_pixels",
        "candidate_foreground_pixels",
        "baseline_unique_values",
        "candidate_unique_values",
        "baseline_max_value",
        "candidate_max_value",
    ]
    missing_fields = ["split", "sample_id", "issue", "baseline_label", "candidate_label"]

    write_tsv(args.output_dir / "chip_samples.tsv", all_chip_rows, chip_fields)
    write_tsv(args.output_dir / "label_samples.tsv", all_label_rows, label_fields)
    write_tsv(args.output_dir / "missing_common_labels.tsv", all_missing_rows, missing_fields)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
