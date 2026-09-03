"""Structured request, preflight, result, and diagnostic contracts for chips."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Literal

from .chip_config import SPLIT_NAMES, SplitName
from .tiling_results import TileCubeRecord


PreflightStatus = Literal["pending", "passed", "failed"]
ChipStatus = Literal["pending", "success", "skipped", "partial", "failed"]
DiagnosticSeverity = Literal["info", "warning", "error"]

PREFLIGHT_STATUSES = ("pending", "passed", "failed")
CHIP_STATUSES = ("pending", "success", "skipped", "partial", "failed")
DIAGNOSTIC_SEVERITIES = ("info", "warning", "error")

_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _sample_id(value: object) -> str:
    text = _non_empty_text(value, field_name="sample_id")
    if not _SAFE_SAMPLE_ID.fullmatch(text):
        raise ValueError(
            "sample_id must contain only letters, numbers, underscores, and "
            "hyphens, and must start with a letter or number."
        )
    return text


def _finite_tuple(
    value: object,
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a numeric sequence.")
    if len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values.")
    if any(isinstance(item, bool) for item in value):
        raise TypeError(f"{field_name} values must be numeric, not booleans.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} values must be numeric.") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} values must be finite.")
    return result


@dataclass(frozen=True)
class GeographicAOI:
    """One logical geographic query AOI in upper-left/lower-right order."""

    upper_left_latitude: float
    upper_left_longitude: float
    lower_right_latitude: float
    lower_right_longitude: float

    def __post_init__(self) -> None:
        values = _finite_tuple(
            (
                self.upper_left_latitude,
                self.upper_left_longitude,
                self.lower_right_latitude,
                self.lower_right_longitude,
            ),
            length=4,
            field_name="geographic AOI",
        )
        (
            upper_left_latitude,
            upper_left_longitude,
            lower_right_latitude,
            lower_right_longitude,
        ) = values
        if upper_left_latitude <= lower_right_latitude:
            raise ValueError(
                "upper_left_latitude must be greater than lower_right_latitude."
            )
        if upper_left_longitude == lower_right_longitude:
            raise ValueError("Geographic AOI longitude extent must be nonzero.")
        for name, value in zip(
            (
                "upper_left_latitude",
                "upper_left_longitude",
                "lower_right_latitude",
                "lower_right_longitude",
            ),
            values,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TargetGrid:
    """The complete authoritative output raster grid for one chip."""

    crs_wkt: str
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "crs_wkt",
            _non_empty_text(self.crs_wkt, field_name="target grid crs_wkt"),
        )
        transform = _finite_tuple(
            self.transform,
            length=6,
            field_name="target grid transform",
        )
        determinant = transform[1] * transform[5] - transform[2] * transform[4]
        if math.isclose(determinant, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("target grid transform must be invertible.")
        object.__setattr__(self, "transform", transform)
        bounds = _finite_tuple(
            self.bounds,
            length=4,
            field_name="target grid bounds",
        )
        left, bottom, right, top = bounds
        if left >= right or bottom >= top:
            raise ValueError(
                "target grid bounds must satisfy left < right and bottom < top."
            )
        object.__setattr__(self, "bounds", bounds)
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"target grid {name} must be an integer.")
            if value < 1:
                raise ValueError(f"target grid {name} must be positive.")


@dataclass(frozen=True)
class SourceSelector:
    """A product selector for one product-scoped source in one group."""

    acquisition_group: str
    source_name: str
    product_id: str

    def __post_init__(self) -> None:
        for name in ("acquisition_group", "source_name", "product_id"):
            object.__setattr__(
                self,
                name,
                _non_empty_text(getattr(self, name), field_name=name),
            )


@dataclass(frozen=True)
class ReferenceSample:
    """Metadata extracted from a reference TIFF before request processing."""

    path: Path
    sample_id: str
    target_grid: TargetGrid
    geographic_aoi: GeographicAOI

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "sample_id", _sample_id(self.sample_id))
        if not isinstance(self.target_grid, TargetGrid):
            raise TypeError("target_grid must be a TargetGrid.")
        if not isinstance(self.geographic_aoi, GeographicAOI):
            raise TypeError("geographic_aoi must be a GeographicAOI.")


@dataclass(frozen=True)
class LabelValidationDiagnostic:
    """One typed observation produced while resolving or validating a label."""

    code: str
    message: str
    severity: DiagnosticSeverity = "error"
    expected: str | None = None
    actual: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _non_empty_text(self.code, field_name="diagnostic code"),
        )
        object.__setattr__(
            self,
            "message",
            _non_empty_text(self.message, field_name="diagnostic message"),
        )
        if self.severity not in DIAGNOSTIC_SEVERITIES:
            valid = ", ".join(DIAGNOSTIC_SEVERITIES)
            raise ValueError(f"diagnostic severity must be one of {valid}.")
        if self.expected is not None:
            object.__setattr__(self, "expected", str(self.expected))
        if self.actual is not None:
            object.__setattr__(self, "actual", str(self.actual))


@dataclass(frozen=True)
class ChipRequest:
    """One intended final chip and its complete target/query identity."""

    sample_id: str
    target_grid: TargetGrid
    geographic_aoi: GeographicAOI
    split_group_key: str
    label_path: Path | None = None
    assigned_split: SplitName | None = None
    source_selectors: tuple[SourceSelector, ...] = ()
    reference_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _sample_id(self.sample_id))
        if not isinstance(self.target_grid, TargetGrid):
            raise TypeError("target_grid must be a TargetGrid.")
        if not isinstance(self.geographic_aoi, GeographicAOI):
            raise TypeError("geographic_aoi must be a GeographicAOI.")
        object.__setattr__(
            self,
            "split_group_key",
            _non_empty_text(
                self.split_group_key,
                field_name="split_group_key",
            ),
        )
        if self.label_path is not None:
            object.__setattr__(self, "label_path", Path(self.label_path))
        if self.reference_path is not None:
            object.__setattr__(self, "reference_path", Path(self.reference_path))
        if self.assigned_split is not None:
            split = str(self.assigned_split).strip().lower()
            if split not in SPLIT_NAMES:
                valid = ", ".join(SPLIT_NAMES)
                raise ValueError(f"assigned_split must be one of {valid}.")
            object.__setattr__(self, "assigned_split", split)
        selectors = tuple(self.source_selectors)
        if any(not isinstance(item, SourceSelector) for item in selectors):
            raise TypeError("source_selectors must contain SourceSelector objects.")
        selector_keys = [
            (item.acquisition_group.casefold(), item.source_name.casefold())
            for item in selectors
        ]
        if len(set(selector_keys)) != len(selector_keys):
            raise ValueError(
                "source_selectors must be unique by acquisition group and source."
            )
        object.__setattr__(self, "source_selectors", selectors)


def validate_request_contracts(requests: Sequence[ChipRequest]) -> None:
    """Validate cross-request identity and explicit split invariants."""
    if any(not isinstance(request, ChipRequest) for request in requests):
        raise TypeError("requests must contain ChipRequest objects.")
    sample_keys = [request.sample_id.casefold() for request in requests]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Chip request sample IDs must be case-insensitively unique.")
    explicit_group_splits: dict[str, SplitName] = {}
    for request in requests:
        if request.assigned_split is None:
            continue
        group_key = request.split_group_key.casefold()
        previous = explicit_group_splits.setdefault(group_key, request.assigned_split)
        if previous != request.assigned_split:
            raise ValueError(
                f"Split group {request.split_group_key!r} has conflicting explicit "
                f"assignments: {previous!r} and {request.assigned_split!r}."
            )


@dataclass(frozen=True)
class ChipPreflight:
    """Resolved split and label state established before acquisition."""

    status: PreflightStatus
    assigned_split: SplitName | None = None
    resolved_label_path: Path | None = None
    label_diagnostics: tuple[LabelValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PREFLIGHT_STATUSES:
            valid = ", ".join(PREFLIGHT_STATUSES)
            raise ValueError(f"preflight status must be one of {valid}.")
        if self.assigned_split is not None:
            split = str(self.assigned_split).strip().lower()
            if split not in SPLIT_NAMES:
                valid = ", ".join(SPLIT_NAMES)
                raise ValueError(f"assigned_split must be one of {valid}.")
            object.__setattr__(self, "assigned_split", split)
        if self.resolved_label_path is not None:
            object.__setattr__(
                self,
                "resolved_label_path",
                Path(self.resolved_label_path),
            )
        diagnostics = tuple(self.label_diagnostics)
        if any(
            not isinstance(item, LabelValidationDiagnostic) for item in diagnostics
        ):
            raise TypeError(
                "label_diagnostics must contain LabelValidationDiagnostic objects."
            )
        object.__setattr__(self, "label_diagnostics", diagnostics)


@dataclass(frozen=True)
class ChipResult:
    """Structured outcome for one request; no status-string tuples required."""

    request: ChipRequest
    status: ChipStatus
    preflight: ChipPreflight
    chip_path: Path | None = None
    label_path: Path | None = None
    cube_records: tuple[TileCubeRecord, ...] = ()
    diagnostic_path: Path | None = None
    message: str | None = None
    elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ChipRequest):
            raise TypeError("request must be a ChipRequest.")
        if self.status not in CHIP_STATUSES:
            valid = ", ".join(CHIP_STATUSES)
            raise ValueError(f"chip status must be one of {valid}.")
        if not isinstance(self.preflight, ChipPreflight):
            raise TypeError("preflight must be a ChipPreflight.")
        for name in ("chip_path", "label_path", "diagnostic_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        records = tuple(self.cube_records)
        if any(not isinstance(item, TileCubeRecord) for item in records):
            raise TypeError("cube_records must contain TileCubeRecord objects.")
        object.__setattr__(self, "cube_records", records)
        if self.message is not None:
            object.__setattr__(self, "message", str(self.message))
        if self.elapsed_seconds is not None:
            if isinstance(self.elapsed_seconds, bool):
                raise TypeError("elapsed_seconds must be numeric or None.")
            try:
                elapsed = float(self.elapsed_seconds)
            except (TypeError, ValueError) as exc:
                raise TypeError("elapsed_seconds must be numeric or None.") from exc
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise ValueError("elapsed_seconds must be finite and nonnegative.")
            object.__setattr__(self, "elapsed_seconds", elapsed)


class LabelMismatchError(ValueError):
    """A label failed identity, archive, shape, or grid preflight for one sample."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        diagnostics: tuple[LabelValidationDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.sample_id = _sample_id(sample_id)
        self.diagnostics = tuple(diagnostics)
        if any(
            not isinstance(item, LabelValidationDiagnostic)
            for item in self.diagnostics
        ):
            raise TypeError(
                "diagnostics must contain LabelValidationDiagnostic objects."
            )


__all__ = [
    "CHIP_STATUSES",
    "DIAGNOSTIC_SEVERITIES",
    "PREFLIGHT_STATUSES",
    "ChipPreflight",
    "ChipRequest",
    "ChipResult",
    "ChipStatus",
    "DiagnosticSeverity",
    "GeographicAOI",
    "LabelMismatchError",
    "LabelValidationDiagnostic",
    "PreflightStatus",
    "ReferenceSample",
    "SourceSelector",
    "TargetGrid",
    "validate_request_contracts",
]
