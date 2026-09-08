#!/usr/bin/env python
"""Profile serial and process-parallel modern chip creation on HPC."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import (  # noqa: E402
    AcquisitionGroupConfig,
    BandNoDataOverride,
    ChipConfig,
    MINIRF_SOURCE_NODATA,
    MINIRF_SOURCE_NODATA_BANDS,
    NoSplitConfig,
    OutputModalityConfig,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
    TileConfig,
    TileSourceConfig,
    chip_requests_from_reference_directory,
    create_chips,
)


DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)
DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timing_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum_seconds": None,
            "maximum_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "sum_seconds": None,
        }
    return {
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "sum_seconds": sum(values),
    }


def _build_config(args: argparse.Namespace) -> ChipConfig:
    wac_data_dir = _require_directory(args.wac_data_dir, "WAC data directory")
    wac_index = _require_file(
        args.wac_index or wac_data_dir / "output_index.shp",
        "WAC vector index",
    )
    sources = [
        TileSourceConfig(
            name="wac",
            data_dir=wac_data_dir,
            index_path=wac_index,
            location_field=args.location_field,
            selection_mode="product_id",
            resampling="bilinear",
            preserve_source_nodata=True,
        )
    ]
    modalities = [OutputModalityConfig("wac_grid", "wac", "wac")]

    if args.include_static:
        static_data_dir = _require_directory(
            args.static_data_dir,
            "static data directory",
        )
        static_index = _require_file(
            args.static_index or static_data_dir / "db2.shp",
            "static vector index",
        )
        sources.append(
            TileSourceConfig(
                name="static",
                data_dir=static_data_dir,
                index_path=static_index,
                location_field=args.location_field,
                selection_mode="all_intersecting",
                band_names=STATIC_BAND_NAMES,
                resampling="bilinear",
                output_nodata=STATIC_OUTPUT_NODATA,
                band_nodata_overrides=tuple(
                    BandNoDataOverride(
                        band_name=name,
                        source_value=MINIRF_SOURCE_NODATA,
                    )
                    for name in MINIRF_SOURCE_NODATA_BANDS
                ),
            )
        )
        modalities.append(
            OutputModalityConfig("wac_grid", "static", "static")
        )

    group = AcquisitionGroupConfig(
        "wac_grid",
        TileConfig(
            output_dir=args.output_root / ".tiling-template",
            zoom_level=args.zoom_level,
            sources=tuple(sources),
        ),
    )
    return ChipConfig(
        output_root=args.output_root,
        intermediate_root=args.output_root / ".intermediate",
        label_source=args.label_source,
        acquisition_groups=(group,),
        output_modalities=tuple(modalities),
        split_config=NoSplitConfig(),
        sample_limit=args.sample_limit,
        intermediate_retention=args.intermediate_retention,
    )


def _run_profile(args: argparse.Namespace) -> int:
    args.reference_dir = _require_directory(
        args.reference_dir,
        "reference-chip directory",
    )
    args.label_source = _require_directory(args.label_source, "label directory")
    args.output_root = args.output_root.expanduser().resolve()
    args.report_path = args.report_path.expanduser().resolve()
    if args.report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing profile report: {args.report_path}"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            "Profile output is not empty; choose a clean "
            f"--output-root: {args.output_root}"
        )
    config = _build_config(args)
    requests = chip_requests_from_reference_directory(
        args.reference_dir,
        split_group_key=lambda reference: reference.sample_id.split("_", 1)[0],
        recursive=args.recursive,
        sample_limit=args.sample_limit,
    )
    if not requests:
        raise ValueError("No reference TIFFs were discovered for profiling.")

    started_at = _utc_now()
    batch = create_chips(
        requests,
        config,
        max_workers=args.max_workers,
    )
    finished_at = _utc_now()
    statuses = Counter(result.status for result in batch.results)
    sample_times = [
        result.elapsed_seconds
        for result in batch.results
        if result.elapsed_seconds is not None
    ]
    manifest_samples = json.loads(
        batch.manifest_path.read_text(encoding="utf-8")
    )["samples"]
    report = {
        "profile_version": 1,
        "case_name": args.case_name,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "inputs": {
            "reference_dir": str(args.reference_dir),
            "label_source": str(args.label_source),
            "wac_data_dir": str(args.wac_data_dir),
            "wac_index": str(args.wac_index or args.wac_data_dir / "output_index.shp"),
            "include_static": args.include_static,
            "static_data_dir": (
                str(args.static_data_dir) if args.include_static else None
            ),
            "static_index": (
                str(args.static_index or args.static_data_dir / "db2.shp")
                if args.include_static
                else None
            ),
            "zoom_level": args.zoom_level,
            "sample_limit": args.sample_limit,
        },
        "execution": {
            "requested_max_workers": args.max_workers,
            "effective_worker_count": batch.worker_count,
            "batch_elapsed_seconds": batch.elapsed_seconds,
            "samples_per_second": len(batch.results) / batch.elapsed_seconds,
            "sample_timing": _timing_summary(sample_times),
            "coordinator_max_rss_kb": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "children_max_rss_kb": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
        },
        "outcome": {
            "sample_count": len(batch.results),
            "status_counts": dict(sorted(statuses.items())),
            "manifest_path": str(batch.manifest_path),
            "manifest_sha256": _sha256(batch.manifest_path),
            "samples": [
                {
                    "sample_id": item["sample_id"],
                    "assigned_split": item["assigned_split"],
                    "processing_status": item["processing_status"],
                }
                for item in manifest_samples
            ],
        },
    }
    _write_json(args.report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    if statuses != {"success": len(batch.results)}:
        raise RuntimeError(
            "Profiling requires every sample to succeed; see report status_counts "
            f"at {args.report_path}."
        )
    return 0


def _elapsed_text_seconds(value: str) -> float:
    pieces = value.strip().split(":")
    if len(pieces) == 2:
        minutes, seconds = pieces
        return float(minutes) * 60.0 + float(seconds)
    if len(pieces) == 3:
        hours, minutes, seconds = pieces
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Unrecognized GNU time elapsed value: {value!r}")


def _read_gnu_time(path: Path) -> dict[str, float | int]:
    elapsed = None
    max_rss = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            elapsed = _elapsed_text_seconds(line.rsplit(": ", 1)[1])
        elif line.startswith("Maximum resident set size (kbytes)"):
            max_rss = int(line.rsplit(": ", 1)[1])
    if elapsed is None or max_rss is None:
        raise ValueError(f"Incomplete GNU time report: {path}")
    return {"elapsed_seconds": elapsed, "maximum_rss_kb": max_rss}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_profiles(args: argparse.Namespace) -> int:
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing comparison: {args.output_path}"
        )
    serial = _load_json(args.serial_report)
    parallel = _load_json(args.parallel_report)
    serial_samples = serial["outcome"]["samples"]
    parallel_samples = parallel["outcome"]["samples"]
    if serial_samples != parallel_samples:
        raise AssertionError(
            "Serial and parallel sample identity, assignments, or statuses differ."
        )
    serial_time = _read_gnu_time(args.serial_time_report)
    parallel_time = _read_gnu_time(args.parallel_time_report)
    serial_seconds = float(serial["execution"]["batch_elapsed_seconds"])
    parallel_seconds = float(parallel["execution"]["batch_elapsed_seconds"])
    worker_count = int(parallel["execution"]["effective_worker_count"])
    speedup = serial_seconds / parallel_seconds
    report = {
        "comparison_version": 1,
        "created_at_utc": _utc_now(),
        "sample_count": len(serial_samples),
        "parallel_worker_count": worker_count,
        "chip_batch": {
            "serial_seconds": serial_seconds,
            "parallel_seconds": parallel_seconds,
            "speedup": speedup,
            "parallel_efficiency": speedup / worker_count,
        },
        "whole_process_gnu_time": {
            "serial": serial_time,
            "parallel": parallel_time,
            "speedup": (
                float(serial_time["elapsed_seconds"])
                / float(parallel_time["elapsed_seconds"])
            ),
            "maximum_rss_ratio": (
                int(parallel_time["maximum_rss_kb"])
                / int(serial_time["maximum_rss_kb"])
            ),
        },
        "reports": {
            "serial": str(args.serial_report),
            "parallel": str(args.parallel_report),
            "serial_gnu_time": str(args.serial_time_report),
            "parallel_gnu_time": str(args.parallel_time_report),
        },
    }
    for name, value in (
        ("chip batch speedup", speedup),
        ("parallel efficiency", speedup / worker_count),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid {name}: {value!r}")
    _write_json(args.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one chip-creation profile case.")
    run.add_argument("--case-name", required=True)
    run.add_argument("--reference-dir", type=Path, required=True)
    run.add_argument("--label-source", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--report-path", type=Path, required=True)
    run.add_argument("--max-workers", type=int, required=True)
    run.add_argument("--sample-limit", type=int, default=8)
    run.add_argument("--zoom-level", type=int, default=5)
    run.add_argument("--recursive", action="store_true")
    run.add_argument(
        "--intermediate-retention",
        choices=("never", "on_failure", "always"),
        default="never",
    )
    run.add_argument("--location-field", default="location")
    run.add_argument("--wac-data-dir", type=Path, default=DEFAULT_WAC_DATA_DIR)
    run.add_argument("--wac-index", type=Path, default=None)
    run.add_argument("--include-static", action="store_true")
    run.add_argument(
        "--static-data-dir",
        type=Path,
        default=DEFAULT_STATIC_DATA_DIR,
    )
    run.add_argument("--static-index", type=Path, default=None)
    run.set_defaults(handler=_run_profile)

    compare = subparsers.add_parser(
        "compare",
        help="Combine two profile and GNU time reports.",
    )
    compare.add_argument("--serial-report", type=Path, required=True)
    compare.add_argument("--parallel-report", type=Path, required=True)
    compare.add_argument("--serial-time-report", type=Path, required=True)
    compare.add_argument("--parallel-time-report", type=Path, required=True)
    compare.add_argument("--output-path", type=Path, required=True)
    compare.set_defaults(handler=_compare_profiles)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
