#!/usr/bin/env python3
"""Inventory lunar raster directories for polar-tiling test candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm.data_processing.tiling.lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH  # noqa: E402
from lfm.data_processing.tiling.vector_index_builder import (  # noqa: E402
    FOOTPRINT_EDGE_SAMPLES,
    _raster_footprint,
)


DEFAULT_PROJECT_ROOT = Path("/explore/nobackup/projects/lfm")
DEFAULT_ROOTS = (
    DEFAULT_PROJECT_ROOT / "data",
    DEFAULT_PROJECT_ROOT / "processed_data",
    DEFAULT_PROJECT_ROOT / "rawdata",
)
DEFAULT_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".vrt",
    ".nc",
    ".img",
    ".jp2",
    ".cub",
)
TILING_DEFAULT_EXTENSIONS = frozenset({".tif", ".tiff", ".vrt", ".nc"})
POLAR_THRESHOLD = 82.0


_WORKER_MODULES: dict[str, Any] = {}
_WORKER_OUTPUT_SRS = None
_WORKER_EDGE_SAMPLES = FOOTPRINT_EDGE_SAMPLES


@dataclass(frozen=True)
class Inventory:
    paths: tuple[Path, ...]
    discovered_by_directory: dict[Path, int]
    selected_by_directory: dict[Path, int]
    walk_errors: tuple[dict[str, str], ...]


def _default_workers() -> int:
    raw_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw_value:
        try:
            workers = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"SLURM_CPUS_PER_TASK must be an integer, got {raw_value!r}."
            ) from exc
        if workers > 0:
            return workers
    return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        dest="roots",
        help=(
            "Directory tree to scan; repeat for multiple roots. Defaults to "
            "the project data, processed_data, and rawdata directories."
        ),
    )
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help=(
            "Case-insensitive raster suffix to include; repeat as needed. "
            f"Defaults to {DEFAULT_EXTENSIONS}."
        ),
    )
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--edge-samples",
        type=int,
        default=FOOTPRINT_EDGE_SAMPLES,
        help="Densified footprint samples per raster edge (default: 21).",
    )
    parser.add_argument(
        "--max-files-per-directory",
        type=int,
        default=0,
        help=(
            "Deterministically sample at most this many rasters from each "
            "leaf directory; zero scans every discovered raster."
        ),
    )
    parser.add_argument(
        "--max-total-files",
        type=int,
        default=0,
        help=(
            "Deterministically cap the total selected rasters after per-directory "
            "sampling; zero has no global cap."
        ),
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory basename to skip recursively; repeat as needed.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include directory names beginning with a period.",
    )
    parser.add_argument(
        "--include-non-candidates",
        action="store_true",
        help="Include every successful raster result in the JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory matching paths without opening them through GDAL.",
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _normalized_extensions(values: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        suffix = str(value).strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        normalized.append(suffix)
    result = tuple(sorted(set(normalized)))
    if not result:
        raise ValueError("At least one raster extension is required.")
    return result


def _stratified_sample(paths: list[Path], limit: int) -> tuple[Path, ...]:
    ordered = sorted(paths, key=str)
    if limit <= 0 or len(ordered) <= limit:
        return tuple(ordered)
    if limit == 1:
        return (ordered[len(ordered) // 2],)
    indexes = {
        round(index * (len(ordered) - 1) / (limit - 1))
        for index in range(limit)
    }
    return tuple(ordered[index] for index in sorted(indexes))


def discover_inventory(
    roots: tuple[Path, ...],
    *,
    extensions: tuple[str, ...],
    max_files_per_directory: int,
    max_total_files: int,
    excluded_directories: frozenset[str],
    include_hidden: bool,
) -> Inventory:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    walk_errors: list[dict[str, str]] = []

    def on_error(error: OSError) -> None:
        walk_errors.append(
            {
                "path": str(error.filename or "unknown"),
                "error": f"{type(error).__name__}: {error}",
            }
        )

    for root in roots:
        if not root.is_dir():
            walk_errors.append(
                {
                    "path": str(root),
                    "error": "Root directory does not exist or is inaccessible.",
                }
            )
            continue
        for directory, names, filenames in os.walk(
            root,
            topdown=True,
            onerror=on_error,
            followlinks=False,
        ):
            names[:] = sorted(
                name
                for name in names
                if name not in excluded_directories
                and (include_hidden or not name.startswith("."))
            )
            parent = Path(directory)
            for filename in sorted(filenames):
                path = parent / filename
                if path.suffix.lower() in extensions and path.is_file():
                    grouped[parent].append(path.absolute())

    discovered = {
        directory: len(paths)
        for directory, paths in sorted(grouped.items(), key=lambda item: str(item[0]))
    }
    selected_by_directory = {
        directory: _stratified_sample(paths, max_files_per_directory)
        for directory, paths in sorted(grouped.items(), key=lambda item: str(item[0]))
    }
    selected = tuple(
        path
        for paths in selected_by_directory.values()
        for path in paths
    )
    if max_total_files > 0 and len(selected) > max_total_files:
        selected = _stratified_sample(list(selected), max_total_files)
    selected_set = set(selected)
    selected_counts = {
        directory: sum(path in selected_set for path in paths)
        for directory, paths in selected_by_directory.items()
    }
    return Inventory(
        paths=tuple(sorted(selected, key=str)),
        discovered_by_directory=discovered,
        selected_by_directory=selected_counts,
        walk_errors=tuple(walk_errors),
    )


def _initialize_worker(wkt_path: str, edge_samples: int) -> None:
    global _WORKER_MODULES, _WORKER_OUTPUT_SRS, _WORKER_EDGE_SAMPLES

    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
    output_srs = osr.SpatialReference()
    if output_srs.ImportFromWkt(Path(wkt_path).read_text()) != 0:
        raise ValueError(f"Could not import repository lunar WKT: {wkt_path}")
    output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    _WORKER_MODULES = {"gdal": gdal, "ogr": ogr, "osr": osr}
    _WORKER_OUTPUT_SRS = output_srs
    _WORKER_EDGE_SAMPLES = edge_samples


def _suggested_points(
    *,
    minimum_latitude: float,
    maximum_latitude: float,
    representative_longitude: float,
) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    longitude = (representative_longitude + 180.0) % 360.0 - 180.0
    if maximum_latitude > POLAR_THRESHOLD:
        lower = max(POLAR_THRESHOLD, minimum_latitude)
        latitude = min(89.5, (lower + maximum_latitude) / 2.0)
        points.append(
            {
                "grid_id": "LPS_N",
                "lat": latitude,
                "lon": longitude,
            }
        )
    if minimum_latitude < -POLAR_THRESHOLD:
        upper = min(-POLAR_THRESHOLD, maximum_latitude)
        latitude = max(-89.5, (minimum_latitude + upper) / 2.0)
        points.append(
            {
                "grid_id": "LPS_S",
                "lat": latitude,
                "lon": longitude,
            }
        )
    return points


def _inspect_raster(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    gdal = _WORKER_MODULES["gdal"]
    ogr = _WORKER_MODULES["ogr"]
    osr = _WORKER_MODULES["osr"]
    started = perf_counter()
    try:
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError("GDAL could not open the raster.")
        try:
            driver = dataset.GetDriver()
            source_srs = dataset.GetSpatialRef()
            metadata = {
                "driver": driver.ShortName if driver is not None else None,
                "width": int(dataset.RasterXSize),
                "height": int(dataset.RasterYSize),
                "band_count": int(dataset.RasterCount),
                "crs_name": source_srs.GetName() if source_srs is not None else None,
                "subdataset_count": len(dataset.GetSubDatasets()),
            }
        finally:
            dataset = None

        footprint = _raster_footprint(
            path,
            output_srs=_WORKER_OUTPUT_SRS,
            gdal=gdal,
            ogr=ogr,
            osr=osr,
            samples_per_edge=_WORKER_EDGE_SAMPLES,
        )
        minimum_lon, maximum_lon, minimum_lat, maximum_lat = (
            float(value) for value in footprint.GetEnvelope()
        )
        centroid = footprint.Centroid()
        representative_longitude = (
            float(centroid.GetX())
            if centroid is not None and not centroid.IsEmpty()
            else 0.0
        )
        directions = []
        if maximum_lat > POLAR_THRESHOLD:
            directions.append("north")
        if minimum_lat < -POLAR_THRESHOLD:
            directions.append("south")
        contains_north_pole = maximum_lat >= 90.0 - 1e-9
        contains_south_pole = minimum_lat <= -90.0 + 1e-9
        result = {
            "status": "candidate" if directions else "non_candidate",
            "path": str(path),
            "directory": str(path.parent),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "tiling_default_extension": (
                path.suffix.lower() in TILING_DEFAULT_EXTENSIONS
            ),
            "size_bytes": path.stat().st_size,
            **metadata,
            "geographic_extent": {
                "south": minimum_lat,
                "west": minimum_lon,
                "north": maximum_lat,
                "east": maximum_lon,
            },
            "polar_directions": directions,
            "contains_north_pole": contains_north_pole,
            "contains_south_pole": contains_south_pole,
            "suggested_test_points": _suggested_points(
                minimum_latitude=minimum_lat,
                maximum_latitude=maximum_lat,
                representative_longitude=representative_longitude,
            ),
            "elapsed_seconds": perf_counter() - started,
        }
        return result
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "directory": str(path.parent),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": perf_counter() - started,
        }


def _progress(iterator, *, total: int):
    try:
        from tqdm import tqdm

        return tqdm(
            iterator,
            total=total,
            desc="Inspecting lunar rasters",
            unit="raster",
            file=sys.stdout,
            dynamic_ncols=True,
        )
    except ImportError:
        return iterator


def inspect_paths(
    paths: tuple[Path, ...],
    *,
    workers: int,
    edge_samples: int,
) -> list[dict[str, Any]]:
    if not paths:
        return []
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(LUNAR_GEOGRAPHIC_WKT_PATH), edge_samples),
    ) as executor:
        futures = {
            executor.submit(_inspect_raster, str(path)): path
            for path in paths
        }
        for future in _progress(as_completed(futures), total=len(futures)):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "error",
                    "path": str(path),
                    "directory": str(path.parent),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": None,
                }
            results.append(result)
    return sorted(results, key=lambda result: result["path"])


def directory_summaries(
    inventory: Inventory,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_directory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        results_by_directory[result["directory"]].append(result)

    summaries = []
    for directory, discovered_count in inventory.discovered_by_directory.items():
        directory_results = results_by_directory[str(directory)]
        candidates = [
            result for result in directory_results if result["status"] == "candidate"
        ]
        successful = [
            result
            for result in directory_results
            if result["status"] in {"candidate", "non_candidate"}
        ]
        north_candidates = [
            result
            for result in candidates
            if "north" in result["polar_directions"]
        ]
        south_candidates = [
            result
            for result in candidates
            if "south" in result["polar_directions"]
        ]
        pole_candidates = [
            result
            for result in candidates
            if result["contains_north_pole"] or result["contains_south_pole"]
        ]
        summaries.append(
            {
                "directory": str(directory),
                "discovered_count": discovered_count,
                "selected_count": inventory.selected_by_directory[directory],
                "successful_count": len(successful),
                "error_count": sum(
                    result["status"] == "error" for result in directory_results
                ),
                "candidate_count": len(candidates),
                "north_candidate_count": len(north_candidates),
                "south_candidate_count": len(south_candidates),
                "pole_candidate_count": len(pole_candidates),
                "candidate_paths": [result["path"] for result in candidates],
                "complete_directory_scan": (
                    inventory.selected_by_directory[directory] == discovered_count
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (
            -summary["pole_candidate_count"],
            -summary["candidate_count"],
            -summary["successful_count"],
            summary["directory"],
        ),
    )


def print_summary(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print("\nPolar candidate inventory summary", flush=True)
    print(f"  Raster directories: {totals['directory_count']}", flush=True)
    print(f"  Rasters discovered: {totals['discovered_count']}", flush=True)
    print(f"  Rasters selected: {totals['selected_count']}", flush=True)
    print(f"  Successful inspections: {totals['successful_count']}", flush=True)
    print(f"  Inspection errors: {totals['error_count']}", flush=True)
    print(f"  Polar candidates: {totals['candidate_count']}", flush=True)
    print(f"  North candidates: {totals['north_candidate_count']}", flush=True)
    print(f"  South candidates: {totals['south_candidate_count']}", flush=True)
    print(f"  Pole-containing candidates: {totals['pole_candidate_count']}", flush=True)

    candidate_directories = [
        summary
        for summary in report["directory_summaries"]
        if summary["candidate_count"]
    ]
    print("\nCandidate directories (ranked):", flush=True)
    for summary in candidate_directories[:30]:
        completeness = "complete" if summary["complete_directory_scan"] else "sampled"
        print(
            "  "
            f"{summary['directory']} | candidates={summary['candidate_count']} "
            f"north={summary['north_candidate_count']} "
            f"south={summary['south_candidate_count']} "
            f"poles={summary['pole_candidate_count']} | {completeness}",
            flush=True,
        )

    print("\nCandidate rasters:", flush=True)
    for candidate in report["candidates"][:50]:
        directions = ",".join(candidate["polar_directions"])
        extent = candidate["geographic_extent"]
        print(
            f"  [{directions}] {candidate['path']} "
            f"lat=[{extent['south']:.6f}, {extent['north']:.6f}]",
            flush=True,
        )
    if len(report["candidates"]) > 50:
        print(
            f"  ... {len(report['candidates']) - 50} additional candidates in JSON",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.edge_samples < 2:
        raise ValueError("--edge-samples must be at least 2.")
    if args.max_files_per_directory < 0 or args.max_total_files < 0:
        raise ValueError("Sampling limits must be nonnegative.")

    roots = tuple(Path(root).absolute() for root in (args.roots or DEFAULT_ROOTS))
    extensions = _normalized_extensions(args.extensions or DEFAULT_EXTENSIONS)
    report_path = args.report.absolute()
    started = perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()

    print(f"Roots: {', '.join(str(root) for root in roots)}", flush=True)
    print(f"Extensions: {', '.join(extensions)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Edge samples: {args.edge_samples}", flush=True)
    print(
        "Per-directory limit: "
        f"{args.max_files_per_directory or 'all rasters'}",
        flush=True,
    )
    print(f"Global limit: {args.max_total_files or 'all selected rasters'}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print("\nDiscovering raster paths...", flush=True)

    inventory = discover_inventory(
        roots,
        extensions=extensions,
        max_files_per_directory=args.max_files_per_directory,
        max_total_files=args.max_total_files,
        excluded_directories=frozenset(args.exclude_dir),
        include_hidden=args.include_hidden,
    )
    discovered_count = sum(inventory.discovered_by_directory.values())
    print(
        f"Discovered {discovered_count} raster(s) in "
        f"{len(inventory.discovered_by_directory)} director(ies); "
        f"selected {len(inventory.paths)} for inspection.",
        flush=True,
    )

    results = [] if args.dry_run else inspect_paths(
        inventory.paths,
        workers=args.workers,
        edge_samples=args.edge_samples,
    )
    summaries = directory_summaries(inventory, results)
    candidates = [result for result in results if result["status"] == "candidate"]
    candidates.sort(
        key=lambda result: (
            not (result["contains_north_pole"] or result["contains_south_pole"]),
            not result["tiling_default_extension"],
            result["path"],
        )
    )
    errors = [result for result in results if result["status"] == "error"]
    successful = [result for result in results if result["status"] != "error"]
    north_candidates = [
        result for result in candidates if "north" in result["polar_directions"]
    ]
    south_candidates = [
        result for result in candidates if "south" in result["polar_directions"]
    ]
    pole_candidates = [
        result
        for result in candidates
        if result["contains_north_pole"] or result["contains_south_pole"]
    ]
    exhaustive = (
        args.max_files_per_directory == 0
        and args.max_total_files == 0
        and not inventory.walk_errors
    )
    report = {
        "status": "inventory_only" if args.dry_run else "completed",
        "generated_at_utc": generated_at,
        "elapsed_seconds": perf_counter() - started,
        "roots": [str(root) for root in roots],
        "extensions": list(extensions),
        "workers": args.workers,
        "edge_samples": args.edge_samples,
        "polar_threshold_degrees": POLAR_THRESHOLD,
        "sampling": {
            "max_files_per_directory": args.max_files_per_directory,
            "max_total_files": args.max_total_files,
            "exhaustive": exhaustive,
            "strategy": "deterministic evenly spaced filenames",
        },
        "totals": {
            "directory_count": len(inventory.discovered_by_directory),
            "discovered_count": discovered_count,
            "selected_count": len(inventory.paths),
            "successful_count": len(successful),
            "error_count": len(errors),
            "candidate_count": len(candidates),
            "north_candidate_count": len(north_candidates),
            "south_candidate_count": len(south_candidates),
            "pole_candidate_count": len(pole_candidates),
        },
        "walk_errors": list(inventory.walk_errors),
        "directory_summaries": summaries,
        "candidates": candidates,
        "inspection_errors": errors,
        "non_candidates": (
            [
                result
                for result in results
                if result["status"] == "non_candidate"
            ]
            if args.include_non_candidates
            else []
        ),
        "notes": [
            "Candidate status is footprint-based and does not read raster pixels.",
            "Positive-area coverage north of +82 or south of -82 qualifies.",
            "Suggested points are envelope diagnostics and should be verified "
            "visually.",
            "Default tiling discovery accepts tif, tiff, nc, and vrt; other GDAL "
            "formats require explicit source preparation patterns.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print("\nDry run completed; no rasters were opened.", flush=True)
    else:
        print_summary(report)
    print(f"\nJSON report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
