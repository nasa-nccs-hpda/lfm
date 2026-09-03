"""Batch preflight orchestration that never invokes tiling or writes chips."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .chip_config import ChipConfig
from .chip_labels import preflight_label
from .chip_requests import (
    AmbiguousLongitudeError,
    UnsupportedCoverageError,
    materialize_requests,
    validate_request_geographic_aoi,
    validate_target_grid_consistency,
)
from .chip_splits import SplitAssignment, SplitPlan, plan_splits
from .chip_types import (
    ChipPreflight,
    ChipRequest,
    LabelMismatchError,
    LabelValidationDiagnostic,
)


@dataclass(frozen=True)
class PreparedChipRequest:
    """One request paired with its split and non-writing preflight outcome."""

    request: ChipRequest
    assignment: SplitAssignment
    preflight: ChipPreflight

    @property
    def eligible_for_acquisition(self) -> bool:
        """Return whether C3 may invoke tiling for this request."""
        return (
            self.assignment.assigned_split is not None
            and self.preflight.status == "passed"
        )


@dataclass(frozen=True)
class BatchPreflightResult:
    """All deterministic split and label outcomes for a request batch."""

    requests: tuple[PreparedChipRequest, ...]
    split_plan: SplitPlan


def preflight_chip_requests(
    requests: Sequence[ChipRequest],
    config: ChipConfig,
) -> BatchPreflightResult:
    """Validate, split, and label-check a batch without acquiring or writing."""
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    ordered = materialize_requests(requests, sample_limit=config.sample_limit)
    geographic_failures: dict[str, ChipPreflight] = {}
    valid_requests: list[ChipRequest] = []
    for request in ordered:
        try:
            validate_target_grid_consistency(request.target_grid)
            validate_request_geographic_aoi(request)
            valid_requests.append(request)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, UnsupportedCoverageError):
                code = exc.status
            elif isinstance(exc, AmbiguousLongitudeError):
                code = exc.status
            else:
                code = "invalid_target_grid"
            geographic_failures[request.sample_id.casefold()] = ChipPreflight(
                status="failed",
                assigned_split=request.assigned_split,
                label_diagnostics=(
                    LabelValidationDiagnostic(
                        code=code,
                        message=str(exc),
                        severity="error",
                    ),
                ),
            )
    split_plan = plan_splits(tuple(valid_requests), config.split_config)
    prepared: list[PreparedChipRequest] = []
    for request in ordered:
        geographic_failure = geographic_failures.get(request.sample_id.casefold())
        if geographic_failure is not None:
            prepared.append(
                PreparedChipRequest(
                    request=request,
                    assignment=SplitAssignment(
                        sample_id=request.sample_id,
                        split_group_key=request.split_group_key,
                        assigned_split=None,
                        source="unassigned",
                    ),
                    preflight=geographic_failure,
                )
            )
            continue
        assignment = split_plan.assignment_for(request.sample_id)
        if assignment.assigned_split is None:
            preflight = ChipPreflight(
                status="skipped",
                assigned_split=None,
                label_diagnostics=(
                    LabelValidationDiagnostic(
                        code="unassigned_split",
                        message=(
                            "The number-only split policy left this request "
                            "unassigned; it is ineligible for acquisition."
                        ),
                        severity="warning",
                    ),
                ),
            )
        else:
            try:
                preflight = preflight_label(
                    request,
                    label_source=config.label_source,
                    assigned_split=assignment.assigned_split,
                )
            except LabelMismatchError as exc:
                preflight = ChipPreflight(
                    status="failed",
                    assigned_split=assignment.assigned_split,
                    resolved_label_path=exc.label_path or request.label_path,
                    label_diagnostics=exc.diagnostics,
                )
        prepared.append(
            PreparedChipRequest(
                request=request,
                assignment=assignment,
                preflight=preflight,
            )
        )
    return BatchPreflightResult(
        requests=tuple(prepared),
        split_plan=split_plan,
    )


__all__ = [
    "BatchPreflightResult",
    "PreparedChipRequest",
    "preflight_chip_requests",
]
