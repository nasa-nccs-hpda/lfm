#!/usr/bin/env python3
"""Build and reuse a small raster index from deterministic real-data links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm.data_processing.tiling.vector_index_builder import (
    DEFAULT_RASTER_GLOBS,
    VectorIndexBuildConfig,
    ensure_vector_index,
)


DEFAULT_SOURCE_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--image-glob",
        action="append",
        dest="image_globs",
        help=(
            "Raster glob to include; repeat for multiple formats. Defaults to "
            f"{DEFAULT_RASTER_GLOBS}."
        ),
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def discover_sample(
    source_dir: Path,
    *,
    image_globs: tuple[str, ...],
    limit: int,
) -> tuple[Path, ...]:
    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"Real-data source directory does not exist: {source_dir}"
        )
    if limit < 1:
        raise ValueError("limit must be positive.")
    paths = tuple(
        sorted(
            {
                path.resolve()
                for pattern in image_globs
                for path in source_dir.glob(pattern)
                if path.is_file()
            },
            key=str,
        )[:limit]
    )
    if not paths:
        raise FileNotFoundError(
            f"No real rasters matched {image_globs!r} in {source_dir}"
        )
    return paths


def main() -> None:
    args = parse_args()
    if args.work_dir.exists():
        raise FileExistsError(
            f"Real-data progress work directory already exists: {args.work_dir}"
        )
    linked_data_dir = args.work_dir / "source_links"
    linked_data_dir.mkdir(parents=True)
    image_globs = tuple(args.image_globs or DEFAULT_RASTER_GLOBS)
    selected_paths = discover_sample(
        args.source_dir,
        image_globs=image_globs,
        limit=args.limit,
    )
    for source_path in selected_paths:
        (linked_data_dir / source_path.name).symlink_to(source_path)

    index_path = linked_data_dir / "output_index.gpkg"
    config = VectorIndexBuildConfig(
        data_dir=linked_data_dir,
        index_path=index_path,
        image_globs=image_globs,
        layer_name="real_data_progress",
    )
    print(
        f"Selected {len(selected_paths)} real raster(s) from {args.source_dir}.",
        flush=True,
    )
    print("The next operation should display per-raster tqdm progress:", flush=True)
    creation_started = perf_counter()
    created = ensure_vector_index(config, stdout=sys.stdout)
    creation_seconds = perf_counter() - creation_started

    print("\nRe-running ensure to validate read-only index reuse...", flush=True)
    reuse_started = perf_counter()
    reused = ensure_vector_index(config, stdout=sys.stdout)
    reuse_seconds = perf_counter() - reuse_started
    metadata_matches = created == reused
    if not metadata_matches:
        raise AssertionError("Created and reused index validation results differ.")

    report = {
        "source_dir": str(args.source_dir),
        "source_image_globs": image_globs,
        "requested_limit": args.limit,
        "selected_count": len(selected_paths),
        "selected_real_paths": [str(path) for path in selected_paths],
        "linked_data_dir": str(linked_data_dir),
        "index_path": str(index_path),
        "index_size_bytes": index_path.stat().st_size,
        "driver_name": created.driver_name,
        "layer_name": created.layer_name,
        "location_field": created.location_field,
        "feature_count": created.feature_count,
        "indexed_paths": [str(path) for path in created.raster_paths],
        "creation_seconds": creation_seconds,
        "reuse_seconds": reuse_seconds,
        "creation_and_reuse_metadata_match": metadata_matches,
        "progress_stream": "stdout",
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
