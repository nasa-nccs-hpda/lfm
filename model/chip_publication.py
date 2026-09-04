"""Publish validated chip-label pairs and write deterministic split manifests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from .chip_assembly import WrittenChip, validate_written_chip
from .chip_config import (
    ChipConfig,
    MixedPercentageNumberSplitConfig,
    NumberSplitConfig,
    SimpleSplitConfig,
    SplitConfigType,
    SplitPercentages,
)
from .chip_labels import validate_label
from .chip_preflight import PreparedChipRequest
from .chip_splits import SplitPlan
from .chip_types import ChipResult, GeographicAOI, SourceSelector, TargetGrid


DATASET_MANIFEST_VERSION = 1
CONFIGURATION_ID_ALGORITHM = "sha256"
SPLIT_HASH_ALGORITHM = "blake2b"


class ChipPublicationError(RuntimeError):
    """A validated chip-label pair could not be published safely."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        code: str = "publication_error",
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.code = code


@dataclass(frozen=True)
class DatasetPublicationValidation:
    """Summary of exact split membership validated on disk."""

    sample_count: int
    successful_count: int
    status_counts: tuple[tuple[str, int], ...]
    split_counts: tuple[tuple[str, int], ...]
    manifest_path: Path | None = None


def _publication_error(
    sample_id: str,
    message: str,
    *,
    code: str,
) -> ChipPublicationError:
    return ChipPublicationError(message, sample_id=sample_id, code=code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_output_name(sample_id: str, source_path: Path) -> str:
    suffix = source_path.suffix.casefold()
    if suffix not in {".npy", ".npz"}:
        raise _publication_error(
            sample_id,
            f"Unsupported validated label extension {source_path.suffix!r}.",
            code="unsupported_label_type",
        )
    return f"{sample_id}_label{suffix}"


def _effective_selectors(written: WrittenChip) -> tuple[SourceSelector, ...]:
    selectors = tuple(
        selector
        for group in written.assembled.reprojection.acquisition.group_results
        for selector in group.selectors
    )
    keys = [
        (item.acquisition_group.casefold(), item.source_name.casefold())
        for item in selectors
    ]
    if len(set(keys)) != len(keys):
        raise _publication_error(
            written.assembled.sample_id,
            "Acquisition retained duplicate effective source selectors.",
            code="duplicate_effective_selectors",
        )
    return selectors


def _remove_paths(paths: Sequence[Path]) -> tuple[OSError, ...]:
    failures: list[OSError] = []
    for path in reversed(tuple(paths)):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failures.append(exc)
    return tuple(failures)


def publish_chip_pair(written: WrittenChip, config: ChipConfig) -> ChipResult:
    """Publish one chip and byte-preserved label as a rollback-safe pair."""
    if not isinstance(written, WrittenChip):
        raise TypeError("written must be a WrittenChip.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    assembled = written.assembled
    if assembled.config != config:
        raise _publication_error(
            assembled.sample_id,
            "Written chip and publication configurations differ.",
            code="publication_config_mismatch",
        )
    prepared = assembled.reprojection.acquisition.prepared_request
    request = prepared.request
    split = prepared.assignment.assigned_split
    if (
        not prepared.eligible_for_acquisition
        or split is None
        or prepared.preflight.assigned_split != split
    ):
        raise _publication_error(
            request.sample_id,
            "Only a preflight-valid request with a consistent split may publish.",
            code="ineligible_publication",
        )
    source_label = prepared.preflight.resolved_label_path
    if source_label is None or not source_label.is_file():
        raise _publication_error(
            request.sample_id,
            "The preflight-valid source label is no longer available.",
            code="missing_source_label",
        )
    if not written.path.is_file() or written.validation.path != written.path:
        raise _publication_error(
            request.sample_id,
            "The validated staged chip is no longer available.",
            code="missing_staged_chip",
        )
    expected_chip_name = f"{request.sample_id}{config.final_output_suffix}"
    if written.path.name != expected_chip_name:
        raise _publication_error(
            request.sample_id,
            f"Staged chip name {written.path.name!r} does not match "
            f"{expected_chip_name!r}.",
            code="staged_chip_name_mismatch",
        )

    # Recheck the source label and chip immediately before any final directory
    # is touched. This catches a changed label or staged raster after C2/C5.
    validate_label(request, source_label)
    validate_written_chip(
        written.path,
        assembled,
        dtype_name=config.output_dtype,
    )
    validated_chip_hash = _sha256(written.path)
    validated_label_hash = _sha256(source_label)
    effective_selectors = _effective_selectors(written)

    split_root = config.output_root / split
    chip_path = split_root / "chips" / expected_chip_name
    label_path = split_root / "labels" / _label_output_name(
        request.sample_id,
        source_label,
    )
    result = ChipResult(
        request=request,
        status="success",
        preflight=prepared.preflight,
        chip_path=chip_path,
        label_path=label_path,
        cube_records=assembled.reprojection.acquisition.records,
        effective_selectors=effective_selectors,
    )
    for destination in (chip_path, label_path):
        if destination.exists() or destination.is_symlink():
            raise _publication_error(
                request.sample_id,
                f"Refusing to overwrite existing dataset artifact {destination}.",
                code="publication_output_exists",
            )

    chip_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    staged_chip = chip_path.with_name(
        f".{chip_path.name}.{uuid4().hex}.publishing"
    )
    staged_label = label_path.with_name(
        f".{label_path.name}.{uuid4().hex}.publishing"
    )
    staged = (staged_chip, staged_label)
    published: list[Path] = []
    try:
        shutil.copyfile(written.path, staged_chip)
        shutil.copyfile(source_label, staged_label)
        if _sha256(staged_chip) != validated_chip_hash:
            raise _publication_error(
                request.sample_id,
                "Staged chip bytes differ from the validated chip.",
                code="staged_chip_checksum_mismatch",
            )
        if _sha256(staged_label) != validated_label_hash:
            raise _publication_error(
                request.sample_id,
                "Staged label bytes differ from the preflight-valid label.",
                code="staged_label_checksum_mismatch",
            )
        if (
            _sha256(written.path) != validated_chip_hash
            or _sha256(source_label) != validated_label_hash
        ):
            raise _publication_error(
                request.sample_id,
                "A source artifact changed during pair publication.",
                code="publication_source_changed",
            )
        # A hard link within each destination directory gives no-overwrite
        # publication. If the second link fails, the first is rolled back.
        os.link(staged_chip, chip_path)
        published.append(chip_path)
        os.link(staged_label, label_path)
        published.append(label_path)
    except Exception as exc:
        rollback_failures = _remove_paths((*published, *staged))
        if rollback_failures:
            raise _publication_error(
                request.sample_id,
                "Pair publication failed and rollback could not remove every "
                "sample artifact.",
                code="publication_rollback_failed",
            ) from exc
        if isinstance(exc, ChipPublicationError):
            raise
        raise _publication_error(
            request.sample_id,
            f"Could not publish chip-label pair: {exc}",
            code="pair_publication_failed",
        ) from exc
    cleanup_failures = _remove_paths(staged)
    if cleanup_failures:
        rollback_failures = _remove_paths((*published, *staged))
        if rollback_failures:
            raise _publication_error(
                request.sample_id,
                "Published pair staging cleanup failed and rollback could not "
                "remove every sample artifact.",
                code="publication_rollback_failed",
            )
        raise _publication_error(
            request.sample_id,
            "Pair publication was rolled back because temporary staging cleanup "
            "failed.",
            code="publication_staging_cleanup_failed",
        )

    return result


def _target_grid_document(grid: TargetGrid) -> dict[str, Any]:
    return {
        "crs_wkt": grid.crs_wkt,
        "transform": list(grid.transform),
        "bounds": list(grid.bounds),
        "width": grid.width,
        "height": grid.height,
    }


def _geographic_aoi_document(aoi: GeographicAOI) -> dict[str, float]:
    return {
        "upper_left_latitude": aoi.upper_left_latitude,
        "upper_left_longitude": aoi.upper_left_longitude,
        "lower_right_latitude": aoi.lower_right_latitude,
        "lower_right_longitude": aoi.lower_right_longitude,
    }


def _percentages_document(
    percentages: SplitPercentages | None,
) -> dict[str, float] | None:
    if percentages is None:
        return None
    return {
        "train": percentages.train,
        "val": percentages.val,
        "test": percentages.test,
    }


def _json_number(value: float | None) -> float | str | None:
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _split_type(config: SplitConfigType) -> str:
    if isinstance(config, SimpleSplitConfig):
        return "simple"
    if isinstance(config, MixedPercentageNumberSplitConfig):
        return "mixed_percentage_number"
    if isinstance(config, NumberSplitConfig):
        return "number"
    raise TypeError("config must be a supported split configuration.")


def _split_policy_document(config: SplitConfigType, plan: SplitPlan) -> dict[str, Any]:
    percentages = None
    fixed_counts = None
    fixed_priority = None
    remainder_split = None
    if isinstance(config, SimpleSplitConfig):
        percentages = config.percentages
    elif isinstance(config, MixedPercentageNumberSplitConfig):
        percentages = config.remaining_percentages
        fixed_counts = config.fixed_counts
        fixed_priority = config.fixed_priority
    elif isinstance(config, NumberSplitConfig):
        fixed_counts = config.fixed_counts
        fixed_priority = config.fixed_priority
        remainder_split = config.remainder_split
    else:
        raise TypeError("config must be a supported split configuration.")
    return {
        "type": _split_type(config),
        "percentages": _percentages_document(percentages),
        "fixed_targets": (
            None
            if fixed_counts is None
            else {
                name: getattr(fixed_counts, name)
                for name in ("train", "val", "test")
            }
        ),
        "fixed_priority": (
            None if fixed_priority is None else list(fixed_priority)
        ),
        "remainder_split": remainder_split,
        "requested_counts": (
            None
            if fixed_counts is None
            else {
                name: getattr(fixed_counts, name)
                for name in ("train", "val", "test")
            }
        ),
        "realized_counts": plan.realized_counts,
        "warnings": [
            {
                "code": warning.code,
                "message": warning.message,
                "split": warning.split,
                "requested_count": warning.requested_count,
                "realized_count": warning.realized_count,
                "reason": warning.reason,
            }
            for warning in plan.warnings
        ],
        "fixed_target_policy": "best_effort_whole_groups",
        "dataset_creation_seed": config.seed,
        "hash_algorithm": SPLIT_HASH_ALGORITHM,
        "hash_version": config.hash_version,
        "group_key_policy": config.group_key_policy,
        "prior_manifest_path": (
            None
            if config.prior_manifest_path is None
            else str(config.prior_manifest_path)
        ),
    }


def _configuration_document(config: ChipConfig) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in config.acquisition_groups:
        sources: list[dict[str, Any]] = []
        for source in group.tile_config.sources:
            sources.append(
                {
                    "name": source.name,
                    "data_dir": str(source.data_dir),
                    "index_path": str(source.index_path),
                    "index_layer": source.index_layer,
                    "location_field": source.location_field,
                    "selection_mode": source.selection_mode,
                    "band_names": (
                        None if source.band_names is None else list(source.band_names)
                    ),
                    "band_indices": (
                        None
                        if source.band_indices is None
                        else list(source.band_indices)
                    ),
                    "resampling": source.resampling,
                    "source_nodata": _json_number(source.source_nodata),
                    "output_nodata": _json_number(source.output_nodata),
                    "preserve_source_nodata": source.preserve_source_nodata,
                    "band_nodata_overrides": [
                        {
                            "band_name": override.band_name,
                            "source_value": _json_number(override.source_value),
                            "output_value": _json_number(override.output_value),
                            "preserve_source": override.preserve_source,
                        }
                        for override in source.band_nodata_overrides
                    ],
                    "required": source.required,
                }
            )
        groups.append(
            {
                "name": group.name,
                "zoom_level": group.tile_config.zoom_level,
                "debug": group.tile_config.debug,
                "sources": sources,
            }
        )
    return {
        "output_root": str(config.output_root),
        "intermediate_root": str(config.intermediate_root),
        "label_source": str(config.label_source),
        "output_suffix": config.final_output_suffix,
        "common_nodata": config.common_nodata,
        "output_dtype": config.output_dtype,
        "sample_limit": config.sample_limit,
        "intermediate_retention": config.intermediate_retention,
        "split_config": _split_configuration_document(config.split_config),
        "acquisition_groups": groups,
        "output_modalities": [
            {
                "acquisition_group": modality.acquisition_group,
                "source_name": modality.source_name,
                "alias": modality.alias,
                "band_names": (
                    None
                    if modality.band_names is None
                    else list(modality.band_names)
                ),
                "band_indices": (
                    None
                    if modality.band_indices is None
                    else list(modality.band_indices)
                ),
                "output_band_names": (
                    None
                    if modality.output_band_names is None
                    else list(modality.output_band_names)
                ),
                "resampling": modality.resampling,
            }
            for modality in config.output_modalities
        ],
    }


def _split_configuration_document(config: SplitConfigType) -> dict[str, Any]:
    document: dict[str, Any] = {
        "type": _split_type(config),
        "dataset_creation_seed": config.seed,
        "hash_algorithm": SPLIT_HASH_ALGORITHM,
        "hash_version": config.hash_version,
        "group_key_policy": config.group_key_policy,
        "prior_manifest_path": (
            None
            if config.prior_manifest_path is None
            else str(config.prior_manifest_path)
        ),
    }
    if isinstance(config, SimpleSplitConfig):
        document["percentages"] = _percentages_document(config.percentages)
    elif isinstance(config, MixedPercentageNumberSplitConfig):
        document.update(
            {
                "fixed_targets": {
                    name: getattr(config.fixed_counts, name)
                    for name in ("train", "val", "test")
                },
                "fixed_priority": list(config.fixed_priority),
                "remaining_percentages": _percentages_document(
                    config.remaining_percentages
                ),
            }
        )
    elif isinstance(config, NumberSplitConfig):
        document.update(
            {
                "fixed_targets": {
                    name: getattr(config.fixed_counts, name)
                    for name in ("train", "val", "test")
                },
                "fixed_priority": list(config.fixed_priority),
                "remainder_split": config.remainder_split,
            }
        )
    return document


def _canonical_json(document: Mapping[str, Any], *, pretty: bool) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _configuration_id(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(document, pretty=False)).hexdigest()
    return f"{CONFIGURATION_ID_ALGORITHM}:{digest}"


def _ordered_inputs(
    prepared_requests: Sequence[PreparedChipRequest],
    results: Sequence[ChipResult],
) -> tuple[tuple[PreparedChipRequest, ChipResult], ...]:
    prepared_by_id: dict[str, PreparedChipRequest] = {}
    for prepared in prepared_requests:
        if not isinstance(prepared, PreparedChipRequest):
            raise TypeError(
                "prepared_requests must contain PreparedChipRequest objects."
            )
        key = prepared.request.sample_id.casefold()
        if key in prepared_by_id:
            raise ValueError("Prepared sample IDs must be case-insensitively unique.")
        prepared_by_id[key] = prepared
    results_by_id: dict[str, ChipResult] = {}
    for result in results:
        if not isinstance(result, ChipResult):
            raise TypeError("results must contain ChipResult objects.")
        key = result.request.sample_id.casefold()
        if key in results_by_id:
            raise ValueError("Result sample IDs must be case-insensitively unique.")
        results_by_id[key] = result
    if set(prepared_by_id) != set(results_by_id):
        missing_results = sorted(set(prepared_by_id) - set(results_by_id))
        missing_prepared = sorted(set(results_by_id) - set(prepared_by_id))
        raise ValueError(
            "Prepared requests and results cover different samples: "
            f"missing results={missing_results}, "
            f"missing prepared requests={missing_prepared}."
        )
    ordered = tuple(
        (prepared_by_id[key], results_by_id[key])
        for key in sorted(prepared_by_id)
    )
    for prepared, result in ordered:
        if result.request != prepared.request or result.preflight != prepared.preflight:
            raise ValueError(
                f"Result does not retain prepared request state for "
                f"{prepared.request.sample_id!r}."
            )
    return ordered


def _expected_paths(
    prepared: PreparedChipRequest,
    config: ChipConfig,
) -> tuple[Path, Path] | None:
    split = prepared.assignment.assigned_split
    source_label = prepared.preflight.resolved_label_path
    if split is None or source_label is None:
        return None
    split_root = config.output_root / split
    return (
        split_root
        / "chips"
        / f"{prepared.request.sample_id}{config.final_output_suffix}",
        split_root
        / "labels"
        / _label_output_name(prepared.request.sample_id, source_label),
    )


def _actual_split_files(config: ChipConfig) -> dict[tuple[str, str], set[Path]]:
    files: dict[tuple[str, str], set[Path]] = {}
    for split in ("train", "val", "test"):
        for role in ("chips", "labels"):
            directory = config.output_root / split / role
            entries = set(directory.iterdir()) if directory.is_dir() else set()
            nonfiles = sorted(
                path for path in entries if not path.is_file() and not path.is_symlink()
            )
            if nonfiles:
                raise ValueError(
                    f"Dataset artifact directory {directory} contains non-files: "
                    f"{nonfiles}."
                )
            files[(split, role)] = entries
    return files


def _validate_membership(
    ordered: tuple[tuple[PreparedChipRequest, ChipResult], ...],
    plan: SplitPlan,
    config: ChipConfig,
) -> DatasetPublicationValidation:
    plan_by_sample = {
        assignment.sample_id.casefold(): assignment for assignment in plan.assignments
    }
    group_splits: dict[str, str] = {}
    expected_files = {
        (split, role): set()
        for split in ("train", "val", "test")
        for role in ("chips", "labels")
    }
    status_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for prepared, result in ordered:
        request = prepared.request
        assignment = prepared.assignment
        planned = plan_by_sample.get(request.sample_id.casefold())
        if planned is not None and planned != assignment:
            raise ValueError(
                f"Prepared assignment for {request.sample_id!r} differs from SplitPlan."
            )
        split = assignment.assigned_split
        if prepared.preflight.assigned_split != split and not (
            prepared.preflight.status == "failed" and split is None
        ):
            raise ValueError(
                f"Preflight split disagrees with assignment for {request.sample_id!r}."
            )
        if split is not None:
            group_key = assignment.split_group_key.casefold()
            previous = group_splits.setdefault(group_key, split)
            if previous != split:
                raise ValueError(
                    f"Split group {assignment.split_group_key!r} crosses splits."
                )
        status_counts[result.status] += 1
        if result.status == "success":
            if split is None or prepared.preflight.status != "passed":
                raise ValueError(
                    f"Successful sample {request.sample_id!r} was not eligible."
                )
            expected = _expected_paths(prepared, config)
            if expected is None or (result.chip_path, result.label_path) != expected:
                raise ValueError(
                    f"Successful sample {request.sample_id!r} has unexpected paths."
                )
            chip_path, label_path = expected
            if not chip_path.is_file() or not label_path.is_file():
                raise ValueError(
                    f"Successful sample {request.sample_id!r} is missing an artifact."
                )
            expected_files[(split, "chips")].add(chip_path)
            expected_files[(split, "labels")].add(label_path)
            split_counts[split] += 1
        elif result.chip_path is not None or result.label_path is not None:
            raise ValueError(
                f"Non-success sample {request.sample_id!r} exposes final paths."
            )

    actual_files = _actual_split_files(config)
    for key, expected in expected_files.items():
        actual = actual_files[key]
        if actual != expected:
            missing = sorted(str(path) for path in expected - actual)
            unexpected = sorted(str(path) for path in actual - expected)
            raise ValueError(
                f"Dataset membership mismatch for {key[0]}/{key[1]}: "
                f"missing={missing}, unexpected={unexpected}."
            )
    return DatasetPublicationValidation(
        sample_count=len(ordered),
        successful_count=status_counts["success"],
        status_counts=tuple(sorted(status_counts.items())),
        split_counts=tuple(
            (name, split_counts[name]) for name in ("train", "val", "test")
        ),
    )


def _sample_document(
    prepared: PreparedChipRequest,
    result: ChipResult,
) -> dict[str, Any]:
    request = prepared.request
    assignment = prepared.assignment
    target_grid = _target_grid_document(request.target_grid)
    return {
        "sample_id": request.sample_id,
        "normalized_sample_key": request.sample_id.casefold(),
        "split_group_key": assignment.split_group_key,
        "requested_split": request.assigned_split,
        "assigned_split": assignment.assigned_split,
        "assignment_status": (
            "assigned" if assignment.assigned_split is not None else "unassigned"
        ),
        "assignment_source": assignment.source,
        "processing_status": result.status,
        "preflight_status": prepared.preflight.status,
        "preflight_assigned_split": prepared.preflight.assigned_split,
        "target_grid_id": _configuration_id(target_grid),
        "target_grid": target_grid,
        "geographic_aoi": _geographic_aoi_document(request.geographic_aoi),
        "reference_path": (
            None if request.reference_path is None else str(request.reference_path)
        ),
        "source_label_path": (
            None
            if prepared.preflight.resolved_label_path is None
            else str(prepared.preflight.resolved_label_path)
        ),
        "chip_path": None if result.chip_path is None else str(result.chip_path),
        "label_path": None if result.label_path is None else str(result.label_path),
        "source_selectors": [
            {
                "acquisition_group": selector.acquisition_group,
                "source_name": selector.source_name,
                "product_id": selector.product_id,
            }
            for selector in (result.effective_selectors or request.source_selectors)
        ],
        "label_diagnostics": [
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "severity": diagnostic.severity,
                "expected": diagnostic.expected,
                "actual": diagnostic.actual,
            }
            for diagnostic in prepared.preflight.label_diagnostics
        ],
        "diagnostic_path": (
            None if result.diagnostic_path is None else str(result.diagnostic_path)
        ),
        "message": result.message,
    }


def build_dataset_manifest(
    prepared_requests: Sequence[PreparedChipRequest],
    results: Sequence[ChipResult],
    split_plan: SplitPlan,
    config: ChipConfig,
) -> dict[str, Any]:
    """Build a deterministic, audit-complete manifest document."""
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    ordered = _ordered_inputs(prepared_requests, results)
    validation = _validate_membership(ordered, split_plan, config)
    configuration = _configuration_document(config)
    return {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "configuration_id": _configuration_id(configuration),
        "configuration": configuration,
        "split_policy": _split_policy_document(config.split_config, split_plan),
        "publication_counts": {
            "samples": validation.sample_count,
            "successful": validation.successful_count,
            "statuses": dict(validation.status_counts),
            "splits": dict(validation.split_counts),
        },
        "samples": [
            _sample_document(prepared, result) for prepared, result in ordered
        ],
    }


def validate_dataset_publication(
    prepared_requests: Sequence[PreparedChipRequest],
    results: Sequence[ChipResult],
    split_plan: SplitPlan,
    config: ChipConfig,
    *,
    manifest_path: Path | None = None,
) -> DatasetPublicationValidation:
    """Validate exact artifacts, group isolation, and an optional manifest."""
    ordered = _ordered_inputs(prepared_requests, results)
    validation = _validate_membership(ordered, split_plan, config)
    if manifest_path is None:
        return validation
    path = Path(manifest_path)
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset manifest {path}.") from exc
    expected = build_dataset_manifest(
        prepared_requests,
        results,
        split_plan,
        config,
    )
    if actual != expected:
        raise ValueError("Dataset manifest does not match validated publication state.")
    return DatasetPublicationValidation(
        sample_count=validation.sample_count,
        successful_count=validation.successful_count,
        status_counts=validation.status_counts,
        split_counts=validation.split_counts,
        manifest_path=path,
    )


def write_dataset_manifest(
    prepared_requests: Sequence[PreparedChipRequest],
    results: Sequence[ChipResult],
    split_plan: SplitPlan,
    config: ChipConfig,
    *,
    output_path: Path | None = None,
) -> Path:
    """Atomically write a deterministic manifest without replacing one."""
    path = (
        config.output_root / "dataset_manifest.json"
        if output_path is None
        else Path(output_path)
    )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite dataset manifest {path}.")
    document = build_dataset_manifest(
        prepared_requests,
        results,
        split_plan,
        config,
    )
    payload = _canonical_json(document, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.publishing")
    published = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            if published:
                path.unlink(missing_ok=True)
            raise
    try:
        validate_dataset_publication(
            prepared_requests,
            results,
            split_plan,
            config,
            manifest_path=path,
        )
    except Exception:
        if published:
            path.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "CONFIGURATION_ID_ALGORITHM",
    "DATASET_MANIFEST_VERSION",
    "SPLIT_HASH_ALGORITHM",
    "ChipPublicationError",
    "DatasetPublicationValidation",
    "build_dataset_manifest",
    "publish_chip_pair",
    "validate_dataset_publication",
    "write_dataset_manifest",
]
