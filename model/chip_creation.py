"""Sequential orchestration for target-grid-driven chip creation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any
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


@dataclass(frozen=True)
class ChipBatchResult:
    """Complete deterministic outcome of one sequential dataset run."""

    prepared_requests: tuple[PreparedChipRequest, ...]
    results: tuple[ChipResult, ...]
    split_plan: SplitPlan
    manifest_path: Path
    elapsed_seconds: float

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
) -> ChipResult:
    diagnostics = list(result.diagnostics)
    cleanup = _cleanup_intermediates(prepared, config, result.status)
    if cleanup is not None:
        diagnostics.append(cleanup)
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
        return _finish_result(prepared, result, None, config, overwrite=overwrite)
    if not prepared.eligible_for_acquisition:
        result = ChipResult(
            request=prepared.request,
            status="skipped",
            preflight=prepared.preflight,
            diagnostics=preflight_diagnostics,
            message="Request was not assigned to a dataset split.",
            elapsed_seconds=time.perf_counter() - started,
        )
        return _finish_result(prepared, result, None, config, overwrite=overwrite)

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
            )

    acquisition: ChipAcquisitionResult | None = None
    diagnostics = list(preflight_diagnostics)
    try:
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
            try:
                reprojection = reproject_acquisition(acquisition, config)
                written = assemble_and_write_chip(reprojection, config)
                result = publish_chip_pair(written, config, overwrite=overwrite)
                result = replace(
                    result,
                    diagnostics=tuple(diagnostics),
                    elapsed_seconds=time.perf_counter() - started,
                )
            except LabelMismatchError:
                raise
            except ChipReprojectionError as exc:
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
        )
    except LabelMismatchError:
        _cleanup_intermediates(prepared, config, "failed")
        raise
    finally:
        # Raster datasets are closed inside their owning stage. Dropping these
        # potentially large arrays promptly keeps sequential batches bounded.
        acquisition = None


def _label_failure_result(
    prepared: PreparedChipRequest,
    exc: LabelMismatchError,
    config: ChipConfig,
    *,
    overwrite: bool,
    elapsed_seconds: float,
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
    return _finish_result(prepared, result, None, config, overwrite=overwrite)


def create_chips(
    requests: Iterable[ChipRequest],
    config: ChipConfig,
    *,
    overwrite: bool = False,
) -> ChipBatchResult:
    """Create a deterministic dataset serially while isolating sample failures."""
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    started = time.perf_counter()
    preflight: BatchPreflightResult = preflight_chip_requests(tuple(requests), config)
    results: list[ChipResult] = []
    for prepared in preflight.requests:
        sample_started = time.perf_counter()
        try:
            result = create_chip(prepared, config, overwrite=overwrite)
        except LabelMismatchError as exc:
            result = _label_failure_result(
                prepared,
                exc,
                config,
                overwrite=overwrite,
                elapsed_seconds=time.perf_counter() - sample_started,
            )
        results.append(result)
    manifest_path = write_dataset_manifest(
        preflight.requests,
        tuple(results),
        preflight.split_plan,
        config,
        overwrite=overwrite,
    )
    return ChipBatchResult(
        prepared_requests=preflight.requests,
        results=tuple(results),
        split_plan=preflight.split_plan,
        manifest_path=manifest_path,
        elapsed_seconds=time.perf_counter() - started,
    )


def create_chips_from_reference_directory(
    directory: str | Path,
    config: ChipConfig,
    *,
    split_group_key: Callable[[ReferenceSample], str],
    recursive: bool = False,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    overwrite: bool = False,
) -> ChipBatchResult:
    """Discover sorted reference TIFFs and create their chips serially."""
    requests = chip_requests_from_reference_directory(
        directory,
        split_group_key=split_group_key,
        recursive=recursive,
        edge_samples=edge_samples,
        sample_limit=config.sample_limit,
    )
    return create_chips(requests, config, overwrite=overwrite)


__all__ = [
    "CHIP_DIAGNOSTIC_VERSION",
    "ChipBatchResult",
    "create_chip",
    "create_chips",
    "create_chips_from_reference_directory",
]
