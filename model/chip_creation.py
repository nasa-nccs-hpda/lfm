"""Deterministic orchestration for target-grid-driven chip creation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import shutil
import sys
import time
from typing import Any, Literal
from uuid import uuid4

from .chip_acquisition import (
    AcquisitionDiagnostic,
    ChipAcquisitionResult,
    acquire_prepared_request,
)
from .chip_assembly import ChipAssemblyError, assemble_and_write_chip
from .chip_config import ChipConfig
from .chip_preflight import (
    BatchPreflightResult,
    PreparedChipRequest,
    preflight_chip_requests,
)
from .chip_publication import (
    ChipPublicationError,
    publish_chip_pair,
    write_dataset_manifest,
)
from .chip_reprojection import ChipReprojectionError, reproject_acquisition
from .chip_requests import (
    DEFAULT_EDGE_SAMPLES,
    chip_requests_from_reference_directory,
)
from .chip_splits import SplitPlan
from .chip_types import (
    ChipDiagnostic,
    ChipRequest,
    ChipResult,
    LabelMismatchError,
    ReferenceSample,
    SourceSelector,
)


CHIP_DIAGNOSTIC_VERSION = 1

ChipProgressStage = Literal[
    "preflight",
    "tiling",
    "mosaic/reproject/clip",
    "assemble/write",
    "publish",
    "cleanup",
]
ChipProgressState = Literal["started", "completed", "failed", "skipped"]
ProgressMode = Literal["auto", "live", "log"]
_PROGRESS_STAGES = frozenset(
    {
        "preflight",
        "tiling",
        "mosaic/reproject/clip",
        "assemble/write",
        "publish",
        "cleanup",
    }
)
_PROGRESS_STATES = frozenset({"started", "completed", "failed", "skipped"})


@dataclass(frozen=True)
class ChipProgressEvent:
    """Picklable worker-to-coordinator update for one reference chip."""

    sample_id: str
    stage: ChipProgressStage
    state: ChipProgressState
    worker_pid: int
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a nonempty string.")
        if self.stage not in _PROGRESS_STAGES:
            raise ValueError(f"Unsupported chip progress stage: {self.stage!r}.")
        if self.state not in _PROGRESS_STATES:
            raise ValueError(f"Unsupported chip progress state: {self.state!r}.")
        if isinstance(self.worker_pid, bool) or not isinstance(self.worker_pid, int):
            raise TypeError("worker_pid must be an integer.")
        if self.worker_pid < 1:
            raise ValueError("worker_pid must be positive.")
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("detail must be a string or None.")


def _load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Progress display requires tqdm; install tqdm or use progress=False."
        ) from exc
    return tqdm


def _supports_live_progress() -> bool:
    if sys.stdout.isatty():
        return True
    ipython = sys.modules.get("IPython")
    get_ipython = (
        None if ipython is None else getattr(ipython, "get_ipython", None)
    )
    if get_ipython is None:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


class _ChipProgressReporter:
    """Render all progress from the coordinator process onto stdout."""

    def __init__(
        self,
        total: int,
        worker_count: int,
        *,
        enabled: bool,
        mode: ProgressMode,
    ) -> None:
        self.enabled = enabled
        self.mode: Literal["live", "log"] = (
            "live"
            if mode == "auto" and _supports_live_progress()
            else "log"
            if mode == "auto"
            else mode
        )
        self.worker_count = worker_count
        self._counts: Counter[str] = Counter()
        self._completed_samples: set[str] = set()
        self._sample_workers: dict[str, int] = {}
        self._worker_bars: dict[int, Any] = {}
        self._tqdm: Any | None = None
        self._overall: Any | None = None
        if enabled:
            self._tqdm = _load_tqdm()
            self._overall = self._tqdm(
                total=total,
                desc="Reference chips",
                unit="chip",
                file=sys.stdout,
                dynamic_ncols=self.mode == "live",
                mininterval=0.5,
                position=0,
                leave=True,
            )

    def _worker_bar(self, worker_pid: int) -> Any:
        bar = self._worker_bars.get(worker_pid)
        if bar is None:
            position = len(self._worker_bars) + 1
            bar = self._tqdm(
                total=0,
                bar_format="{desc}",
                file=sys.stdout,
                position=position,
                leave=False,
                dynamic_ncols=True,
            )
            self._worker_bars[worker_pid] = bar
        return bar

    def stage(self, event: ChipProgressEvent) -> None:
        if not self.enabled or event.sample_id in self._completed_samples:
            return
        self._sample_workers[event.sample_id] = event.worker_pid
        message = (
            f"worker {event.worker_pid} | {event.sample_id} | "
            f"{event.stage}: {event.state}"
        )
        if event.detail:
            message = f"{message} ({event.detail})"
        if self.mode == "log":
            self._tqdm.write(message, file=sys.stdout)
            return
        bar = self._worker_bar(event.worker_pid)
        bar.set_description_str(message, refresh=True)

    def complete(self, result: ChipResult) -> None:
        if not self.enabled:
            return
        sample_id = result.request.sample_id
        if sample_id in self._completed_samples:
            return
        self._completed_samples.add(sample_id)
        self._counts[result.status] += 1
        worker_pid = self._sample_workers.get(sample_id, os.getpid())
        terminal = f"worker {worker_pid} | {sample_id} | terminal: {result.status}"
        if result.status in {"failed", "partial"}:
            errors = [
                f"{item.stage}/{item.code}: {item.message}"
                for item in result.diagnostics
                if item.severity == "error"
            ]
            if errors:
                terminal = f"{terminal} ({'; '.join(errors)})"
            elif result.message:
                terminal = f"{terminal} ({result.message})"
        if self.mode == "log":
            self._tqdm.write(terminal, file=sys.stdout)
        else:
            bar = self._worker_bars.get(worker_pid)
            if bar is not None:
                bar.set_description_str(terminal, refresh=True)
        self._overall.set_postfix(
            dict(sorted(self._counts.items())),
            refresh=False,
        )
        self._overall.update(1)

    def close(self) -> None:
        if not self.enabled:
            return
        for bar in self._worker_bars.values():
            bar.close()
        self._overall.close()


_WORKER_PROGRESS_QUEUE: Any | None = None


def _emit_progress(
    callback: Callable[[ChipProgressEvent], None] | None,
    prepared: PreparedChipRequest,
    stage: ChipProgressStage,
    state: ChipProgressState,
    detail: str | None = None,
) -> None:
    if callback is not None:
        callback(
            ChipProgressEvent(
                sample_id=prepared.request.sample_id,
                stage=stage,
                state=state,
                worker_pid=os.getpid(),
                detail=detail,
            )
        )


def _queue_worker_progress(event: ChipProgressEvent) -> None:
    if _WORKER_PROGRESS_QUEUE is not None:
        _WORKER_PROGRESS_QUEUE.put(event)


@dataclass(frozen=True)
class ChipBatchResult:
    """Complete deterministic outcome of one dataset run."""

    prepared_requests: tuple[PreparedChipRequest, ...]
    results: tuple[ChipResult, ...]
    split_plan: SplitPlan
    manifest_path: Path
    elapsed_seconds: float
    worker_count: int = 1

    def __post_init__(self) -> None:
        prepared = tuple(self.prepared_requests)
        results = tuple(self.results)
        if any(not isinstance(item, PreparedChipRequest) for item in prepared):
            raise TypeError(
                "prepared_requests must contain PreparedChipRequest objects."
            )
        if any(not isinstance(item, ChipResult) for item in results):
            raise TypeError("results must contain ChipResult objects.")
        if len(prepared) != len(results):
            raise ValueError("Every prepared request must have exactly one result.")
        if not isinstance(self.split_plan, SplitPlan):
            raise TypeError("split_plan must be a SplitPlan.")
        object.__setattr__(self, "prepared_requests", prepared)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        if isinstance(self.worker_count, bool) or not isinstance(
            self.worker_count,
            int,
        ):
            raise TypeError("worker_count must be an integer.")
        if self.worker_count < 1:
            raise ValueError("worker_count must be positive.")


def _preflight_diagnostics(prepared: PreparedChipRequest) -> tuple[ChipDiagnostic, ...]:
    return tuple(
        ChipDiagnostic(
            stage="preflight",
            code=item.code,
            message=item.message,
            severity=item.severity,
        )
        for item in prepared.preflight.label_diagnostics
    )


def _acquisition_diagnostic(item: AcquisitionDiagnostic) -> ChipDiagnostic:
    return ChipDiagnostic(
        stage="acquisition",
        code=item.code,
        message=item.message,
        severity=item.severity,
        acquisition_group=item.acquisition_group,
        source_name=item.source_name,
        zone=item.zone,
        zoom_level=item.zoom_level,
        tile_x=item.tile_x,
        tile_y=item.tile_y,
    )


def _effective_selectors(
    acquisition: ChipAcquisitionResult | None,
) -> tuple[SourceSelector, ...]:
    if acquisition is None:
        return ()
    return tuple(
        selector
        for group in acquisition.group_results
        for selector in group.selectors
    )


def _acquisition_diagnostics(
    acquisition: ChipAcquisitionResult,
    config: ChipConfig,
) -> tuple[ChipDiagnostic, ...]:
    diagnostics = [
        _acquisition_diagnostic(item) for item in acquisition.diagnostics
    ]
    attempted_groups = {
        item.acquisition_group.casefold() for item in acquisition.group_results
    }
    if acquisition.status == "failed":
        for group in config.acquisition_groups:
            if group.name.casefold() not in attempted_groups:
                diagnostics.append(
                    ChipDiagnostic(
                        stage="acquisition",
                        code="unattempted_acquisition_group",
                        message=(
                            f"Acquisition group {group.name!r} was not attempted "
                            "after an earlier group failed."
                        ),
                        severity="warning",
                        acquisition_group=group.name,
                        zoom_level=group.tile_config.zoom_level,
                    )
                )
        for group_result in acquisition.group_results:
            attempted_count = len(group_result.attempted_query_parts)
            for index in range(attempted_count, len(group_result.query_parts)):
                diagnostics.append(
                    ChipDiagnostic(
                        stage="acquisition",
                        code="unattempted_query_part",
                        message=(
                            f"Query part {index + 1} of "
                            f"{len(group_result.query_parts)} was not attempted "
                            "after an earlier query part failed."
                        ),
                        severity="warning",
                        acquisition_group=group_result.acquisition_group,
                        zoom_level=group_result.zoom_level,
                    )
                )
            if group_result.failed_query_part is not None:
                failed = next(
                    (
                        item
                        for item in group_result.diagnostics
                        if item.severity == "error"
                    ),
                    None,
                )
                diagnostics.append(
                    ChipDiagnostic(
                        stage="acquisition",
                        code="later_tiles_not_attempted",
                        message=(
                            "The tiler stopped at the recorded failing tile; "
                            "later tiles in that query part were not attempted."
                        ),
                        severity="warning",
                        acquisition_group=group_result.acquisition_group,
                        source_name=None if failed is None else failed.source_name,
                        zone=None if failed is None else failed.zone,
                        zoom_level=group_result.zoom_level,
                        tile_x=None if failed is None else failed.tile_x,
                        tile_y=None if failed is None else failed.tile_y,
                    )
                )
    return tuple(diagnostics)


def _stage_diagnostic(stage: str, exc: Exception) -> ChipDiagnostic:
    return ChipDiagnostic(
        stage=stage,  # type: ignore[arg-type]
        code=str(getattr(exc, "code", f"{stage}_error")),
        message=str(exc),
        severity="error",
        acquisition_group=getattr(exc, "acquisition_group", None),
        source_name=getattr(exc, "source_name", None),
        zone=getattr(exc, "zone", None),
        tile_x=getattr(exc, "tile_x", None),
        tile_y=getattr(exc, "tile_y", None),
    )


def _is_partial(acquisition: ChipAcquisitionResult) -> bool:
    return any(
        group.records or group.inventory_paths
        for group in acquisition.group_results
    )


def _sample_intermediate_dir(config: ChipConfig, sample_id: str) -> Path:
    root = config.intermediate_root.resolve(strict=False)
    sample_dir = (config.intermediate_root / sample_id).resolve(strict=False)
    if sample_dir.parent != root:
        raise ValueError("Resolved sample intermediate directory escaped its root.")
    return sample_dir


def _should_retain_intermediates(config: ChipConfig, status: str) -> bool:
    if config.intermediate_retention == "always":
        return True
    if config.intermediate_retention == "on_failure":
        return status in {"partial", "failed"}
    return False


def _cleanup_intermediates(
    prepared: PreparedChipRequest,
    config: ChipConfig,
    status: str,
) -> ChipDiagnostic | None:
    if _should_retain_intermediates(config, status):
        return None
    return _clear_sample_intermediates(prepared, config)


def _clear_sample_intermediates(
    prepared: PreparedChipRequest,
    config: ChipConfig,
) -> ChipDiagnostic | None:
    """Remove only this sample's validated intermediate directory."""
    sample_dir = _sample_intermediate_dir(config, prepared.request.sample_id)
    try:
        if sample_dir.is_symlink():
            raise OSError(f"Refusing to recursively clean symlink {sample_dir}.")
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
    except OSError as exc:
        return ChipDiagnostic(
            stage="cleanup",
            code="intermediate_cleanup_failed",
            message=str(exc),
            severity="warning",
        )
    return None


def _aoi_document(aoi: Any) -> dict[str, float]:
    return {
        "upper_left_latitude": aoi.upper_left_latitude,
        "upper_left_longitude": aoi.upper_left_longitude,
        "lower_right_latitude": aoi.lower_right_latitude,
        "lower_right_longitude": aoi.lower_right_longitude,
    }


def _record_document(record: Any) -> dict[str, Any]:
    return {
        "source_name": record.source_name,
        "zone": record.zone,
        "zoom_level": record.zoom_level,
        "tile_x": record.tile_x,
        "tile_y": record.tile_y,
        "product_id": record.product_id,
        "path": str(record.path),
        "band_names": list(record.band_names),
        "nodata_values": [
            (
                None
                if value is None
                else float(value)
                if math.isfinite(float(value))
                else str(float(value))
            )
            for value in record.nodata_values
        ],
    }


def _diagnostic_document(
    prepared: PreparedChipRequest,
    result: ChipResult,
    acquisition: ChipAcquisitionResult | None,
    config: ChipConfig,
) -> dict[str, Any]:
    group_results = () if acquisition is None else acquisition.group_results
    attempted_names = {
        item.acquisition_group.casefold() for item in group_results
    }
    return {
        "diagnostic_version": CHIP_DIAGNOSTIC_VERSION,
        "sample_id": prepared.request.sample_id,
        "status": result.status,
        "assigned_split": prepared.assignment.assigned_split,
        "preflight_status": prepared.preflight.status,
        "elapsed_seconds": result.elapsed_seconds,
        "message": result.message,
        "diagnostics": [
            {
                "stage": item.stage,
                "code": item.code,
                "message": item.message,
                "severity": item.severity,
                "acquisition_group": item.acquisition_group,
                "source_name": item.source_name,
                "zone": item.zone,
                "zoom_level": item.zoom_level,
                "tile_x": item.tile_x,
                "tile_y": item.tile_y,
            }
            for item in result.diagnostics
        ],
        "acquisition_groups": [
            {
                "name": item.acquisition_group,
                "zoom_level": item.zoom_level,
                "status": item.status,
                "output_dir": str(item.output_dir),
                "logical_aoi": _aoi_document(item.logical_aoi),
                "query_parts": [
                    {
                        "aoi": _aoi_document(part),
                        "state": (
                            "failed"
                            if item.failed_query_part == part
                            else "completed"
                            if part in item.attempted_query_parts
                            else "unattempted"
                        ),
                    }
                    for part in item.query_parts
                ],
                "selectors": [
                    {
                        "source_name": selector.source_name,
                        "product_id": selector.product_id,
                    }
                    for selector in item.selectors
                ],
                "completed_records": [
                    _record_document(record) for record in item.records
                ],
                "inventory_paths": [str(path) for path in item.inventory_paths],
            }
            for item in group_results
        ],
        "unattempted_acquisition_groups": [
            group.name
            for group in config.acquisition_groups
            if group.name.casefold() not in attempted_names
        ],
    }


def _write_json_atomic(
    path: Path,
    document: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise FileExistsError(f"Refusing to replace unsafe diagnostic path {path}.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite chip diagnostic {path}.")
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.publishing")
    backup = path.with_name(f".{path.name}.{uuid4().hex}.previous")
    replaced = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.replace(path, backup)
            replaced = True
        os.link(temporary, path)
        temporary.unlink()
        if replaced:
            backup.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        if replaced:
            path.unlink(missing_ok=True)
            os.replace(backup, path)
        raise


def _finish_result(
    prepared: PreparedChipRequest,
    result: ChipResult,
    acquisition: ChipAcquisitionResult | None,
    config: ChipConfig,
    *,
    overwrite: bool,
    progress_callback: Callable[[ChipProgressEvent], None] | None = None,
) -> ChipResult:
    retained = _should_retain_intermediates(config, result.status)
    _emit_progress(
        progress_callback,
        prepared,
        "cleanup",
        "started",
        "retaining intermediates" if retained else None,
    )
    diagnostics = list(result.diagnostics)
    cleanup = _cleanup_intermediates(prepared, config, result.status)
    if cleanup is not None:
        diagnostics.append(cleanup)
        _emit_progress(
            progress_callback,
            prepared,
            "cleanup",
            "failed",
            cleanup.message,
        )
    else:
        _emit_progress(
            progress_callback,
            prepared,
            "cleanup",
            "completed",
            "retained by policy" if retained else None,
        )
    diagnostic_path = (
        config.output_root / "diagnostics" / f"{prepared.request.sample_id}.json"
    )
    result = replace(
        result,
        diagnostics=tuple(diagnostics),
        diagnostic_path=diagnostic_path,
    )
    _write_json_atomic(
        diagnostic_path,
        _diagnostic_document(prepared, result, acquisition, config),
        overwrite=overwrite,
    )
    return result


def _failed_preflight_result(
    prepared: PreparedChipRequest,
    *,
    elapsed_seconds: float,
) -> ChipResult:
    diagnostics = _preflight_diagnostics(prepared)
    message = diagnostics[0].message if diagnostics else "Chip preflight failed."
    return ChipResult(
        request=prepared.request,
        status="failed",
        preflight=prepared.preflight,
        diagnostics=diagnostics,
        message=message,
        elapsed_seconds=elapsed_seconds,
    )


def create_chip(
    prepared: PreparedChipRequest,
    config: ChipConfig,
    *,
    overwrite: bool = False,
    _progress_callback: Callable[[ChipProgressEvent], None] | None = None,
) -> ChipResult:
    """Run one prepared request serially through acquisition and publication."""
    if not isinstance(prepared, PreparedChipRequest):
        raise TypeError("prepared must be a PreparedChipRequest.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    started = time.perf_counter()
    preflight_diagnostics = _preflight_diagnostics(prepared)
    if prepared.preflight.status == "failed":
        _emit_progress(
            _progress_callback,
            prepared,
            "preflight",
            "failed",
            preflight_diagnostics[0].message if preflight_diagnostics else None,
        )
        if prepared.assignment.assigned_split is not None:
            message = (
                preflight_diagnostics[0].message
                if preflight_diagnostics
                else "Label preflight failed."
            )
            raise LabelMismatchError(
                message,
                sample_id=prepared.request.sample_id,
                diagnostics=prepared.preflight.label_diagnostics,
                label_path=prepared.preflight.resolved_label_path,
            )
        result = _failed_preflight_result(
            prepared,
            elapsed_seconds=time.perf_counter() - started,
        )
        return _finish_result(
            prepared,
            result,
            None,
            config,
            overwrite=overwrite,
            progress_callback=_progress_callback,
        )
    if not prepared.eligible_for_acquisition:
        _emit_progress(
            _progress_callback,
            prepared,
            "preflight",
            "skipped",
            "not assigned to a dataset split",
        )
        result = ChipResult(
            request=prepared.request,
            status="skipped",
            preflight=prepared.preflight,
            diagnostics=preflight_diagnostics,
            message="Request was not assigned to a dataset split.",
            elapsed_seconds=time.perf_counter() - started,
        )
        return _finish_result(
            prepared,
            result,
            None,
            config,
            overwrite=overwrite,
            progress_callback=_progress_callback,
        )

    _emit_progress(
        _progress_callback,
        prepared,
        "preflight",
        "completed",
    )

    if overwrite:
        cleanup = _clear_sample_intermediates(prepared, config)
        if cleanup is not None:
            result = ChipResult(
                request=prepared.request,
                status="failed",
                preflight=prepared.preflight,
                diagnostics=(*preflight_diagnostics, cleanup),
                message=cleanup.message,
                elapsed_seconds=time.perf_counter() - started,
            )
            return _finish_result(
                prepared,
                result,
                None,
                config,
                overwrite=overwrite,
                progress_callback=_progress_callback,
            )

    acquisition: ChipAcquisitionResult | None = None
    diagnostics = list(preflight_diagnostics)
    try:
        _emit_progress(
            _progress_callback,
            prepared,
            "tiling",
            "started",
        )
        acquisition = acquire_prepared_request(prepared, config)
        diagnostics.extend(_acquisition_diagnostics(acquisition, config))
        selectors = _effective_selectors(acquisition)
        if acquisition.status == "failed":
            status = "partial" if _is_partial(acquisition) else "failed"
            message = next(
                (
                    item.message
                    for item in diagnostics
                    if item.stage == "acquisition" and item.severity == "error"
                ),
                "Chip acquisition failed.",
            )
            _emit_progress(
                _progress_callback,
                prepared,
                "tiling",
                "failed",
                message,
            )
            result = ChipResult(
                request=prepared.request,
                status=status,
                preflight=prepared.preflight,
                cube_records=acquisition.records,
                effective_selectors=selectors,
                diagnostics=tuple(diagnostics),
                message=message,
                elapsed_seconds=time.perf_counter() - started,
            )
        else:
            _emit_progress(
                _progress_callback,
                prepared,
                "tiling",
                "completed",
            )
            try:
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "mosaic/reproject/clip",
                    "started",
                )
                reprojection = reproject_acquisition(acquisition, config)
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "mosaic/reproject/clip",
                    "completed",
                )
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "assemble/write",
                    "started",
                )
                written = assemble_and_write_chip(reprojection, config)
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "assemble/write",
                    "completed",
                )
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "publish",
                    "started",
                )
                result = publish_chip_pair(written, config, overwrite=overwrite)
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "publish",
                    "completed",
                )
                result = replace(
                    result,
                    diagnostics=tuple(diagnostics),
                    elapsed_seconds=time.perf_counter() - started,
                )
            except LabelMismatchError:
                raise
            except ChipReprojectionError as exc:
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "mosaic/reproject/clip",
                    "failed",
                    str(exc),
                )
                diagnostics.append(_stage_diagnostic("reprojection", exc))
                result = ChipResult(
                    prepared.request,
                    "failed",
                    prepared.preflight,
                    cube_records=acquisition.records,
                    effective_selectors=selectors,
                    diagnostics=tuple(diagnostics),
                    message=str(exc),
                    elapsed_seconds=time.perf_counter() - started,
                )
            except ChipAssemblyError as exc:
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "assemble/write",
                    "failed",
                    str(exc),
                )
                diagnostics.append(_stage_diagnostic("assembly", exc))
                result = ChipResult(
                    prepared.request,
                    "failed",
                    prepared.preflight,
                    cube_records=acquisition.records,
                    effective_selectors=selectors,
                    diagnostics=tuple(diagnostics),
                    message=str(exc),
                    elapsed_seconds=time.perf_counter() - started,
                )
            except ChipPublicationError as exc:
                _emit_progress(
                    _progress_callback,
                    prepared,
                    "publish",
                    "failed",
                    str(exc),
                )
                diagnostics.append(_stage_diagnostic("publication", exc))
                result = ChipResult(
                    prepared.request,
                    "failed",
                    prepared.preflight,
                    cube_records=acquisition.records,
                    effective_selectors=selectors,
                    diagnostics=tuple(diagnostics),
                    message=str(exc),
                    elapsed_seconds=time.perf_counter() - started,
                )
        return _finish_result(
            prepared,
            result,
            acquisition,
            config,
            overwrite=overwrite,
            progress_callback=_progress_callback,
        )
    except LabelMismatchError:
        _cleanup_intermediates(prepared, config, "failed")
        raise
    finally:
        # Raster datasets are closed inside their owning stage. Dropping these
        # potentially large arrays promptly keeps each worker bounded.
        acquisition = None


def _label_failure_result(
    prepared: PreparedChipRequest,
    exc: LabelMismatchError,
    config: ChipConfig,
    *,
    overwrite: bool,
    elapsed_seconds: float,
    progress_callback: Callable[[ChipProgressEvent], None] | None = None,
) -> ChipResult:
    diagnostics = tuple(
        ChipDiagnostic(
            stage="preflight",
            code=item.code,
            message=item.message,
            severity=item.severity,
        )
        for item in exc.diagnostics
    )
    result = ChipResult(
        request=prepared.request,
        status="failed",
        preflight=prepared.preflight,
        diagnostics=diagnostics,
        message=str(exc),
        elapsed_seconds=elapsed_seconds,
    )
    return _finish_result(
        prepared,
        result,
        None,
        config,
        overwrite=overwrite,
        progress_callback=progress_callback,
    )


def _run_prepared_request(
    prepared: PreparedChipRequest,
    config: ChipConfig,
    overwrite: bool,
    progress_callback: Callable[[ChipProgressEvent], None] | None = None,
) -> ChipResult:
    """Run one prepared request with the batch-level label-error contract."""
    started = time.perf_counter()
    try:
        if progress_callback is None:
            return create_chip(prepared, config, overwrite=overwrite)
        return create_chip(
            prepared,
            config,
            overwrite=overwrite,
            _progress_callback=progress_callback,
        )
    except LabelMismatchError as exc:
        return _label_failure_result(
            prepared,
            exc,
            config,
            overwrite=overwrite,
            elapsed_seconds=time.perf_counter() - started,
            progress_callback=progress_callback,
        )


def _initialize_chip_worker(progress_queue: Any | None = None) -> None:
    """Prevent nested GDAL threading inside each spawned worker process."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue
    os.environ.setdefault("GDAL_NUM_THREADS", "1")


def _run_prepared_task(
    task: tuple[PreparedChipRequest, ChipConfig, bool],
) -> ChipResult:
    """Picklable process-pool entry point for one isolated sample."""
    callback = (
        _queue_worker_progress if _WORKER_PROGRESS_QUEUE is not None else None
    )
    return _run_prepared_request(*task, progress_callback=callback)


def _validate_max_workers(max_workers: int) -> None:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer.")
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")


def _effective_worker_count(max_workers: int, request_count: int) -> int:
    _validate_max_workers(max_workers)
    return min(max_workers, max(1, request_count))


def _validate_progress_options(progress: bool, progress_mode: ProgressMode) -> None:
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean.")
    if not isinstance(progress_mode, str):
        raise TypeError("progress_mode must be a string.")
    if progress_mode not in {"auto", "live", "log"}:
        raise ValueError("progress_mode must be 'auto', 'live', or 'log'.")


def _drain_progress_events(
    progress_queue: Any,
    reporter: _ChipProgressReporter,
) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except Empty:
            return
        reporter.stage(event)


def _run_parallel_requests(
    requests: tuple[PreparedChipRequest, ...],
    config: ChipConfig,
    overwrite: bool,
    worker_count: int,
    reporter: _ChipProgressReporter,
) -> tuple[ChipResult, ...]:
    context = multiprocessing.get_context("spawn")
    tasks = tuple((prepared, config, overwrite) for prepared in requests)
    if not reporter.enabled:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_chip_worker,
        ) as executor:
            return tuple(executor.map(_run_prepared_task, tasks, chunksize=1))

    ordered_results: list[ChipResult | None] = [None] * len(tasks)
    progress_queue = context.Queue()
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_chip_worker,
            initargs=(progress_queue,),
        ) as executor:
            futures = {
                executor.submit(_run_prepared_task, task): index
                for index, task in enumerate(tasks)
            }
            pending = set(futures)
            while pending:
                _drain_progress_events(progress_queue, reporter)
                completed, pending = wait(
                    pending,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    result = future.result()
                    _drain_progress_events(progress_queue, reporter)
                    ordered_results[futures[future]] = result
                    reporter.complete(result)
            _drain_progress_events(progress_queue, reporter)
    finally:
        progress_queue.close()
        progress_queue.join_thread()
    if any(result is None for result in ordered_results):
        raise RuntimeError("A chip worker completed without returning a result.")
    return tuple(result for result in ordered_results if result is not None)


def create_chips(
    requests: Iterable[ChipRequest],
    config: ChipConfig,
    *,
    overwrite: bool = False,
    max_workers: int = 1,
    progress: bool = False,
    progress_mode: ProgressMode = "auto",
) -> ChipBatchResult:
    """Create a deterministic dataset with opt-in process parallelism."""
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    _validate_max_workers(max_workers)
    _validate_progress_options(progress, progress_mode)
    started = time.perf_counter()
    preflight: BatchPreflightResult = preflight_chip_requests(tuple(requests), config)
    worker_count = _effective_worker_count(max_workers, len(preflight.requests))
    reporter = _ChipProgressReporter(
        len(preflight.requests),
        worker_count,
        enabled=progress,
        mode=progress_mode,
    )
    try:
        if worker_count == 1:
            callback = reporter.stage if progress else None
            collected: list[ChipResult] = []
            for prepared in preflight.requests:
                result = _run_prepared_request(
                    prepared,
                    config,
                    overwrite,
                    progress_callback=callback,
                )
                collected.append(result)
                reporter.complete(result)
            results = tuple(collected)
        else:
            results = _run_parallel_requests(
                preflight.requests,
                config,
                overwrite,
                worker_count,
                reporter,
            )
        manifest_path = write_dataset_manifest(
            preflight.requests,
            results,
            preflight.split_plan,
            config,
            overwrite=overwrite,
        )
    finally:
        reporter.close()
    return ChipBatchResult(
        prepared_requests=preflight.requests,
        results=results,
        split_plan=preflight.split_plan,
        manifest_path=manifest_path,
        elapsed_seconds=time.perf_counter() - started,
        worker_count=worker_count,
    )


def create_chips_from_reference_directory(
    directory: str | Path,
    config: ChipConfig,
    *,
    split_group_key: Callable[[ReferenceSample], str],
    recursive: bool = False,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    overwrite: bool = False,
    max_workers: int = 1,
    progress: bool = False,
    progress_mode: ProgressMode = "auto",
) -> ChipBatchResult:
    """Discover sorted reference TIFFs and create their chips deterministically."""
    requests = chip_requests_from_reference_directory(
        directory,
        split_group_key=split_group_key,
        recursive=recursive,
        edge_samples=edge_samples,
        sample_limit=config.sample_limit,
    )
    return create_chips(
        requests,
        config,
        overwrite=overwrite,
        max_workers=max_workers,
        progress=progress,
        progress_mode=progress_mode,
    )


__all__ = [
    "CHIP_DIAGNOSTIC_VERSION",
    "ChipBatchResult",
    "ChipProgressEvent",
    "ChipProgressStage",
    "ChipProgressState",
    "ProgressMode",
    "create_chip",
    "create_chips",
    "create_chips_from_reference_directory",
]
