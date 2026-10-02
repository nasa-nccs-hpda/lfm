#!/usr/bin/env python3
"""Create validated shared GeoPackage indexes for the default tiling data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from lfm.model.vector_index_builder import (
    DEFAULT_RASTER_GLOBS,
    VectorIndexBuildConfig,
    ensure_vector_index,
    resolve_index_worker_count,
)


DEFAULT_PROJECT_DATA_DIR = Path("/explore/nobackup/projects/lfm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-data-dir",
        type=Path,
        default=DEFAULT_PROJECT_DATA_DIR,
        help="Root containing the notebook's default WAC, NAC, and static data.",
    )
    parser.add_argument(
        "--index-name",
        default="output_index.gpkg",
        help="GeoPackage filename created inside each source directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help=(
            "Raster-footprint workers. Defaults to SLURM_CPUS_PER_TASK; set "
            "to 1 to disable parallel processing."
        ),
    )
    return parser.parse_args()


def collection_directories(project_data_dir: Path) -> tuple[tuple[str, Path], ...]:
    return (
        (
            "wac",
            project_data_dir / "processed_data/Lunar/LRO_WAC_Pho_Sites",
        ),
        (
            "nac",
            project_data_dir / "processed_data/Lunar/LRO_NAC_Pho_Sites",
        ),
        ("static", project_data_dir / "staticLinks"),
    )


def main() -> None:
    args = parse_args()
    if Path(args.index_name).name != args.index_name:
        raise ValueError("--index-name must be a filename, not a path.")
    if Path(args.index_name).suffix.lower() != ".gpkg":
        raise ValueError("--index-name must end with .gpkg.")

    collections = collection_directories(args.project_data_dir)
    resolved_worker_count = resolve_index_worker_count(args.worker_count)
    for name, data_dir in collections:
        if not data_dir.is_dir():
            raise NotADirectoryError(
                f"Default {name.upper()} data directory does not exist: {data_dir}"
            )

    print(f"Project data root: {args.project_data_dir}", flush=True)
    print(f"Shared index filename: {args.index_name}", flush=True)
    print(f"Raster-footprint workers: {resolved_worker_count}", flush=True)
    print(
        "Existing indexes will be validated and reused. Invalid or stale shared "
        "indexes will not be replaced automatically.",
        flush=True,
    )

    results = []
    for position, (name, data_dir) in enumerate(collections, start=1):
        index_path = data_dir / args.index_name
        existed_before = index_path.is_file()
        print(
            f"\n[{position}/{len(collections)}] Preparing {name.upper()} index: "
            f"{index_path}",
            flush=True,
        )
        started = perf_counter()
        result = ensure_vector_index(
            VectorIndexBuildConfig(
                data_dir=data_dir,
                index_path=index_path,
                image_globs=DEFAULT_RASTER_GLOBS,
                rebuild_invalid_index=False,
                worker_count=args.worker_count,
            ),
            stdout=sys.stdout,
        )
        elapsed_seconds = perf_counter() - started
        if not existed_before:
            index_path.chmod(0o664)
        results.append(
            {
                "name": name,
                "data_dir": str(data_dir),
                "index_path": str(result.index_path),
                "action": "reused" if existed_before else "created",
                "driver_name": result.driver_name,
                "layer_name": result.layer_name,
                "location_field": result.location_field,
                "feature_count": result.feature_count,
                "elapsed_seconds": elapsed_seconds,
                "mode": oct(index_path.stat().st_mode & 0o777),
            }
        )
        print(
            f"{name.upper()} complete: {result.feature_count} feature(s) in "
            f"{elapsed_seconds:.2f} seconds.",
            flush=True,
        )

    report = {
        "project_data_dir": str(args.project_data_dir),
        "index_name": args.index_name,
        "raster_globs": list(DEFAULT_RASTER_GLOBS),
        "resolved_worker_count": resolved_worker_count,
        "shared_indexes_are_automatically_replaceable": False,
        "collections": results,
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
