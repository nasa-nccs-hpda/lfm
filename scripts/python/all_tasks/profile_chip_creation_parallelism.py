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
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

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


def _diagnostic_document(diagnostic: Any) -> dict[str, Any]:
    return {
        "stage": diagnostic.stage,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "acquisition_group": diagnostic.acquisition_group,
        "source_name": diagnostic.source_name,
        "zone": diagnostic.zone,
        "zoom_level": diagnostic.zoom_level,
        "tile_x": diagnostic.tile_x,
        "tile_y": diagnostic.tile_y,
    }


def _result_document(result: Any, manifest_sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": result.request.sample_id,
        "assigned_split": manifest_sample["assigned_split"],
        "processing_status": result.status,
        "elapsed_seconds": result.elapsed_seconds,
        "message": result.message,
        "chip_path": None if result.chip_path is None else str(result.chip_path),
        "label_path": None if result.label_path is None else str(result.label_path),
        "diagnostic_path": (
            None if result.diagnostic_path is None else str(result.diagnostic_path)
        ),
        "diagnostics": [
            _diagnostic_document(diagnostic) for diagnostic in result.diagnostics
        ],
    }


def _build_config(args: argparse.Namespace) -> Any:
    from lfm.model import (
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
    )

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
    from lfm.model import chip_requests_from_reference_directory, create_chips

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
        progress=args.progress,
        progress_mode=args.progress_mode,
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
        "profile_version": 2,
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
            "all_samples_succeeded": statuses == {"success": len(batch.results)},
            "samples": [
                _result_document(result, manifest_sample)
                for result, manifest_sample in zip(
                    batch.results,
                    manifest_samples,
                    strict=True,
                )
            ],
        },
    }
    _write_json(args.report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    failed = [result for result in batch.results if result.status != "success"]
    if failed:
        print(
            f"Profile completed with {len(failed)} non-successful chip(s); "
            f"full diagnostics: {args.report_path}",
            file=sys.stderr,
        )
        for result in failed:
            errors = [
                f"{item.stage}/{item.code}: {item.message}"
                for item in result.diagnostics
                if item.severity == "error"
            ]
            detail = "; ".join(errors) or result.message or "no error diagnostic"
            print(
                f"  {result.request.sample_id} [{result.status}]: {detail}",
                file=sys.stderr,
            )
    return 0


def _process_children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ()
    return tuple(int(value) for value in text.split()) if text else ()


def _process_tree(root_pid: int) -> set[int]:
    discovered: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in discovered:
            continue
        discovered.add(pid)
        pending.extend(_process_children(pid))
    return discovered


def _resident_kb(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _measure_command(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("measure requires a command after --.")
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite a measurement: {args.output_path}"
        )
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0.0:
        raise ValueError("--interval-seconds must be finite and positive.")
    started = time.monotonic()
    process = subprocess.Popen(command)
    peak_rss_kb = 0
    peak_process_count = 0
    sample_count = 0
    while True:
        pids = _process_tree(process.pid)
        current_rss_kb = sum(_resident_kb(pid) for pid in pids)
        peak_rss_kb = max(peak_rss_kb, current_rss_kb)
        peak_process_count = max(peak_process_count, len(pids))
        sample_count += 1
        return_code = process.poll()
        if return_code is not None:
            break
        time.sleep(args.interval_seconds)
    elapsed_seconds = time.monotonic() - started
    report = {
        "measurement_version": 1,
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "peak_process_tree_rss_kb": peak_rss_kb,
        "peak_process_count": peak_process_count,
        "sampling_interval_seconds": args.interval_seconds,
        "sample_count": sample_count,
        "return_code": return_code,
    }
    _write_json(args.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return return_code


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

    def comparable(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        diagnostic_keys = (
            "stage",
            "code",
            "severity",
            "acquisition_group",
            "source_name",
            "zone",
            "zoom_level",
            "tile_x",
            "tile_y",
        )
        return [
            {
                "sample_id": sample["sample_id"],
                "assigned_split": sample["assigned_split"],
                "processing_status": sample["processing_status"],
                "diagnostics": [
                    {key: diagnostic.get(key) for key in diagnostic_keys}
                    for diagnostic in sample.get("diagnostics", ())
                ],
            }
            for sample in samples
        ]

    if comparable(serial_samples) != comparable(parallel_samples):
        raise AssertionError(
            "Serial and parallel sample identity, assignments, statuses, or "
            "diagnostic codes differ."
        )
    serial_measurement = _load_json(args.serial_measurement)
    parallel_measurement = _load_json(args.parallel_measurement)
    for name, measurement in (
        ("serial", serial_measurement),
        ("parallel", parallel_measurement),
    ):
        if measurement.get("return_code") != 0:
            raise ValueError(f"The {name} measured command did not succeed.")
    serial_seconds = float(serial["execution"]["batch_elapsed_seconds"])
    parallel_seconds = float(parallel["execution"]["batch_elapsed_seconds"])
    worker_count = int(parallel["execution"]["effective_worker_count"])
    serial_monitor_seconds = float(serial_measurement["elapsed_seconds"])
    parallel_monitor_seconds = float(parallel_measurement["elapsed_seconds"])
    serial_peak_rss = int(serial_measurement["peak_process_tree_rss_kb"])
    parallel_peak_rss = int(parallel_measurement["peak_process_tree_rss_kb"])
    if min(
        serial_seconds,
        parallel_seconds,
        serial_monitor_seconds,
        parallel_monitor_seconds,
        serial_peak_rss,
        parallel_peak_rss,
        worker_count,
    ) <= 0:
        raise ValueError("Profile timings, RSS values, and worker count must be positive.")
    speedup = serial_seconds / parallel_seconds
    report = {
        "comparison_version": 2,
        "created_at_utc": _utc_now(),
        "sample_count": len(serial_samples),
        "status_counts": serial["outcome"]["status_counts"],
        "all_samples_succeeded": serial["outcome"].get(
            "all_samples_succeeded",
            False,
        ),
        "parallel_worker_count": worker_count,
        "chip_batch": {
            "serial_seconds": serial_seconds,
            "parallel_seconds": parallel_seconds,
            "speedup": speedup,
            "parallel_efficiency": speedup / worker_count,
        },
        "whole_process_monitor": {
            "serial": serial_measurement,
            "parallel": parallel_measurement,
            "speedup": serial_monitor_seconds / parallel_monitor_seconds,
            "maximum_rss_ratio": parallel_peak_rss / serial_peak_rss,
        },
        "reports": {
            "serial": str(args.serial_report),
            "parallel": str(args.parallel_report),
            "serial_measurement": str(args.serial_measurement),
            "parallel_measurement": str(args.parallel_measurement),
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
    run.add_argument("--progress", action="store_true")
    run.add_argument(
        "--progress-mode",
        choices=("auto", "live", "log"),
        default="auto",
    )
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

    measure = subparsers.add_parser(
        "measure",
        help="Run a command while sampling aggregate process-tree RSS.",
    )
    measure.add_argument("--output-path", type=Path, required=True)
    measure.add_argument("--interval-seconds", type=float, default=0.1)
    measure.add_argument("command", nargs=argparse.REMAINDER)
    measure.set_defaults(handler=_measure_command)

    compare = subparsers.add_parser(
        "compare",
        help="Combine two chip profiles and process-tree measurements.",
    )
    compare.add_argument("--serial-report", type=Path, required=True)
    compare.add_argument("--parallel-report", type=Path, required=True)
    compare.add_argument("--serial-measurement", type=Path, required=True)
    compare.add_argument("--parallel-measurement", type=Path, required=True)
    compare.add_argument("--output-path", type=Path, required=True)
    compare.set_defaults(handler=_compare_profiles)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
