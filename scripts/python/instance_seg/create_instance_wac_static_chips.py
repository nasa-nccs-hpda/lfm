"""Create WAC + static chips for the split instance-segmentation dataset.

This is a thin task-specific wrapper around ``model.chip_making.chip_utils``.
It keeps Pipeline/tiling code untouched, creates refreshed chips per split, and
copies matching existing instance labels into the output dataset layout:

```
output_root/{train,val,test}/chips
output_root/{train,val,test}/labels
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[3]
REPO_PARENT = REPO_DIR.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from lfm.model.chip_making.chip_utils import (  # noqa: E402
    NODATA_POLICIES,
    RESAMPLING_METHODS,
    create_chips,
)


DEFAULT_PROJECT_DIR = Path("/explore/nobackup/projects/lfm")
DEFAULT_REFERENCE_ROOT = (
    DEFAULT_PROJECT_DIR / "model_inputs/300_300_inputs/fm_all_static_all_wac_iseg"
)
DEFAULT_CONTAINER_OUTPUT_ROOT = (
    Path("/explore/nobackup/people")
    / Path.home().name
    / "Lunar_FM/instance_wac_static_chips"
)

SAMPLE_ID_SUFFIXES = (
    "_input_wac_static_chip",
    "_input_wac_chip",
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


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def split_root(root: Path, split: str) -> Path:
    candidate = root / split
    return candidate if candidate.exists() else root


def label_index(labels_dir: Path) -> dict[str, Path]:
    paths = sorted(labels_dir.glob("*.npz")) + sorted(labels_dir.glob("*.npy"))
    indexed: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
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
        raise ValueError(f"Duplicate label sample IDs in {labels_dir}:\n{examples}")
    return indexed


def prepare_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_summary(output_root: Path, summary: dict[str, object]) -> None:
    summary_path = output_root / "create_instance_wac_static_chips_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)


def prepare_requested_output_splits(args: argparse.Namespace) -> None:
    for split in args.splits:
        split_output_root = args.output_root / split
        if args.overwrite_output and split_output_root.exists():
            shutil.rmtree(split_output_root)
        (split_output_root / "chips").mkdir(parents=True, exist_ok=True)
        (split_output_root / "labels").mkdir(parents=True, exist_ok=True)


def assemble_split(
    *,
    split: str,
    generated_chips_dir: Path,
    label_source_dir: Path,
    output_split_root: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    output_chips_dir = output_split_root / "chips"
    output_labels_dir = output_split_root / "labels"
    prepare_dir(output_chips_dir, overwrite=overwrite)
    prepare_dir(output_labels_dir, overwrite=overwrite)

    labels_by_id = label_index(require_path(label_source_dir, "Label source directory"))
    copied_chips = 0
    copied_labels = 0
    missing_labels: list[str] = []

    for chip_path in sorted(generated_chips_dir.glob("*.tif")):
        key = sample_id(chip_path)
        shutil.copy2(chip_path, output_chips_dir / chip_path.name)
        copied_chips += 1

        label_path = labels_by_id.get(key)
        if label_path is None:
            missing_labels.append(chip_path.name)
            continue
        shutil.copy2(label_path, output_labels_dir / label_path.name)
        copied_labels += 1

    if missing_labels:
        examples = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"{len(missing_labels)} generated {split} chip(s) had no matching label. "
            f"First examples:\n{examples}"
        )

    return {
        "split": split,
        "generated_chips_dir": str(generated_chips_dir),
        "label_source_dir": str(label_source_dir),
        "output_chips_dir": str(output_chips_dir),
        "output_labels_dir": str(output_labels_dir),
        "copied_chips": copied_chips,
        "copied_labels": copied_labels,
    }


def process_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    reference_split_root = split_root(args.reference_data_root, split)
    label_split_root = split_root(args.label_source_root, split)

    reference_chip_dir = require_path(
        reference_split_root / "chips",
        f"{split} reference chips directory",
    )
    reference_gpkg_path = require_path(
        reference_chip_dir / "WAC_TILES.gpkg",
        f"{split} reference WAC_TILES.gpkg",
    )
    label_source_dir = require_path(
        label_split_root / "labels",
        f"{split} label source directory",
    )

    split_work_dir = args.working_root / split
    create_chips(
        band_regex=args.band_regex,
        expected_static=args.expected_static,
        zoom_level=args.zoom_level,
        working_dir=split_work_dir,
        nodata_policy=args.nodata_policy,
        resampling_method=args.resampling_method,
        project_dir=args.project_dir,
        train_dir=reference_split_root,
        chip_dir=reference_chip_dir,
        gpkg_path=reference_gpkg_path,
        tile_db_path=args.tile_db_path,
        max_workers=args.max_workers,
        max_entries=args.max_entries,
        verbose=args.verbose,
        extreme_nodata_threshold=args.extreme_nodata_threshold,
    )

    generated_chips_dir = require_path(
        split_work_dir / "chips",
        f"{split} generated chips directory",
    )
    assembly = assemble_split(
        split=split,
        generated_chips_dir=generated_chips_dir,
        label_source_dir=label_source_dir,
        output_split_root=args.output_root / split,
        overwrite=args.overwrite_output,
    )

    return {
        "split": split,
        "reference_split_root": str(reference_split_root),
        "reference_chip_dir": str(reference_chip_dir),
        "reference_gpkg_path": str(reference_gpkg_path),
        "working_dir": str(split_work_dir),
        "assembly": assembly,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-data-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Existing split instance dataset used for chip grids and WAC_TILES.gpkg files.",
    )
    parser.add_argument(
        "--label-source-root",
        type=Path,
        default=None,
        help="Existing split instance dataset to copy labels from. Defaults to --reference-data-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CONTAINER_OUTPUT_ROOT,
        help="Output split dataset root to create.",
    )
    parser.add_argument(
        "--working-root",
        type=Path,
        default=None,
        help="Intermediate chip/datacube root. Defaults to output_root/_chip_creation_work.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Project data root containing processed_data/Lunar.",
    )
    parser.add_argument(
        "--tile-db-path",
        type=Path,
        default=None,
        help="Optional override for the WAC tile database shapefile.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--band-regex",
        default=".*",
        help="Regex selecting static bands by filename/band description.",
    )
    parser.add_argument(
        "--expected-static",
        type=int,
        default=63,
        help="Expected number of selected static bands.",
    )
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument(
        "--nodata-policy",
        choices=NODATA_POLICIES,
        default="preserve",
        help="preserve keeps multiple static NoData sentinels; normalize maps all NoData to the common value.",
    )
    parser.add_argument(
        "--resampling-method",
        choices=tuple(RESAMPLING_METHODS),
        default="bilinear",
    )
    parser.add_argument(
        "--extreme-nodata-threshold",
        type=float,
        default=1.0e30,
        help=(
            "Mask values below -threshold as NoData before/after chip "
            "reprojection. Pass a negative value to disable."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Remove existing output split chips/labels before copying refreshed files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.reference_data_root = args.reference_data_root.resolve()
    args.label_source_root = (
        args.label_source_root.resolve()
        if args.label_source_root is not None
        else args.reference_data_root
    )
    args.output_root = args.output_root.resolve()
    args.working_root = (
        args.working_root.resolve()
        if args.working_root is not None
        else args.output_root / "_chip_creation_work"
    )
    args.project_dir = args.project_dir.resolve()
    if args.tile_db_path is not None:
        args.tile_db_path = args.tile_db_path.resolve()
    if args.extreme_nodata_threshold is not None and args.extreme_nodata_threshold < 0:
        args.extreme_nodata_threshold = None

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.working_root.mkdir(parents=True, exist_ok=True)
    prepare_requested_output_splits(args)

    summary = {
        "reference_data_root": str(args.reference_data_root),
        "label_source_root": str(args.label_source_root),
        "output_root": str(args.output_root),
        "working_root": str(args.working_root),
        "project_dir": str(args.project_dir),
        "tile_db_path": str(args.tile_db_path) if args.tile_db_path else None,
        "band_regex": args.band_regex,
        "expected_static": args.expected_static,
        "zoom_level": args.zoom_level,
        "nodata_policy": args.nodata_policy,
        "resampling_method": args.resampling_method,
        "extreme_nodata_threshold": args.extreme_nodata_threshold,
        "max_workers": args.max_workers,
        "max_entries": args.max_entries,
        "splits": [],
    }

    print(f"Requested splits: {', '.join(args.splits)}", flush=True)
    write_summary(args.output_root, summary)

    for split in args.splits:
        print(f"\n=== Creating {split} WAC + static chips ===", flush=True)
        split_summary = process_split(args, split)
        summary["splits"].append(split_summary)
        write_summary(args.output_root, summary)

    summary_path = args.output_root / "create_instance_wac_static_chips_summary.json"
    print(f"\nWrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
