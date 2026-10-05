"""Structured request, preflight, result, and diagnostic contracts for chips."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from pathlib import Path
import re
from typing import Literal

from .chip_config import ASSIGNMENT_NAMES, SPLIT_NAMES, AssignmentName, SplitName
from .tiling_results import TileCubeRecord


PreflightStatus = Literal["pending", "passed", "failed", "skipped"]
ChipStatus = Literal["pending", "success", "skipped", "partial", "failed"]
DiagnosticSeverity = Literal["info", "warning", "error"]
ChipDiagnosticStage = Literal[
    "preflight",
    "label_preparation",
    "acquisition",
    "reprojection",
    "assembly",
    "publication",
    "cleanup",
    "orchestration",
]

PREFLIGHT_STATUSES = ("pending", "passed", "failed", "skipped")
CHIP_STATUSES = ("pending", "success", "skipped", "partial", "failed")
DIAGNOSTIC_SEVERITIES = ("info", "warning", "error")
CHIP_DIAGNOSTIC_STAGES = (
    "preflight",
    "label_preparation",
    "acquisition",
    "reprojection",
    "assembly",
    "publication",
    "cleanup",
    "orchestration",
)

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


class _DictionaryRecord:
    """JSON-safe metadata only; reconstruction runs the normal validators."""

    def to_dict(self) -> dict:
        return chip_contract_to_dict(self)

    @classmethod
    def from_dict(cls, document: Mapping):
        return _contract_from_dict(cls, document)


@dataclass(frozen=True)
class GeographicAOI(_DictionaryRecord):
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
class TargetGrid(_DictionaryRecord):
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
    band_count: int
    band_descriptions: tuple[str | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "sample_id", _sample_id(self.sample_id))
        if not isinstance(self.target_grid, TargetGrid):
            raise TypeError("target_grid must be a TargetGrid.")
        if not isinstance(self.geographic_aoi, GeographicAOI):
            raise TypeError("geographic_aoi must be a GeographicAOI.")
        if isinstance(self.band_count, bool) or not isinstance(self.band_count, int):
            raise TypeError("band_count must be an integer.")
        if self.band_count < 1:
            raise ValueError("band_count must be positive.")
        descriptions = tuple(self.band_descriptions)
        if len(descriptions) != self.band_count:
            raise ValueError(
                "band_descriptions length must equal band_count."
            )
        object.__setattr__(
            self,
            "band_descriptions",
            tuple(
                None if item is None or not str(item).strip() else str(item).strip()
                for item in descriptions
            ),
        )


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
class LabelInput(_DictionaryRecord):
    """Explicit label association, not a filename/product matching rule.

    ``auto`` infers known file kinds; legacy unsupported suffixes are left for
    preflight to reject, preserving per-sample rather than constructor failure.
    No file is opened by this record.
    """

    path: Path
    kind: Literal["auto", "semantic", "raster_instance", "vector_instance"] = "auto"
    source_grid: TargetGrid | None = None
    relation: Literal["exact", "clip_to_target"] = "exact"
    layer: str | None = None
    source_id: str | None = None
    sidecar_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("label path must be a nonempty path.")
        object.__setattr__(self, "path", Path(self.path))
        if self.kind not in ("auto", "semantic", "raster_instance", "vector_instance"):
            raise ValueError("Unsupported label kind.")
        if self.kind == "auto":
            inferred = {".npy": "semantic", ".tif": "semantic", ".tiff": "semantic",
                        ".npz": "raster_instance", ".gpkg": "vector_instance"}
            object.__setattr__(self, "kind", inferred.get(self.path.suffix.lower(), "auto"))
        if self.source_grid is not None and not isinstance(self.source_grid, TargetGrid):
            raise TypeError("source_grid must be a TargetGrid or None.")
        if self.relation not in ("exact", "clip_to_target"):
            raise ValueError("label relation must be exact or clip_to_target.")
        if self.kind == "vector_instance":
            if self.source_grid is not None:
                raise ValueError("Vector labels embed a CRS, not a raster source_grid.")
            object.__setattr__(self, "layer", "craters" if self.layer is None else self.layer)
        elif self.layer is not None:
            raise ValueError("layer is only applicable to vector-instance labels.")
        for name in ("layer", "source_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_empty_text(value, field_name=name))
        if self.sidecar_path is not None:
            if not str(self.sidecar_path).strip():
                raise ValueError("sidecar_path must not be empty.")
            object.__setattr__(self, "sidecar_path", Path(self.sidecar_path))


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest.")
    return value.lower()


def _label_diagnostics(value) -> tuple[LabelValidationDiagnostic, ...]:
    diagnostics = tuple(value)
    if any(not isinstance(item, LabelValidationDiagnostic) for item in diagnostics):
        raise TypeError("diagnostics must contain LabelValidationDiagnostic objects.")
    return diagnostics


@dataclass(frozen=True)
class LabelPreparationPlan(_DictionaryRecord):
    """Non-writing, picklable plan; coverage must be established by preflight."""

    source: LabelInput
    target_grid: TargetGrid
    method: Literal["exact", "aligned_window", "nearest_warp", "vector_rasterize"]
    source_sha256: str
    source_window: tuple[int, int, int, int] | None = None  # column, row, width, height
    diagnostics: tuple[LabelValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, LabelInput) or not isinstance(self.target_grid, TargetGrid):
            raise TypeError("A label plan requires LabelInput and TargetGrid records.")
        if self.method not in ("exact", "aligned_window", "nearest_warp", "vector_rasterize"):
            raise ValueError("Unsupported label preparation method.")
        if self.source.kind == "auto":
            raise ValueError("A label plan requires a resolved label kind.")
        if self.source.relation == "exact" and self.method != "exact":
            raise ValueError("An exact label input cannot request a clipping method.")
        if (self.source.kind == "vector_instance") != (self.method == "vector_rasterize"):
            raise ValueError("Vector labels require vector_rasterize; raster labels cannot use it.")
        if self.method in ("aligned_window", "nearest_warp") and self.source.source_grid is None:
            raise ValueError("Raster clipping plans require a resolved source_grid.")
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))
        if self.source_window is not None:
            window = tuple(self.source_window)
            if len(window) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in window):
                raise ValueError("source_window must contain four integers.")
            if min(window[:2]) < 0 or min(window[2:]) < 1:
                raise ValueError("source_window requires nonnegative offsets and positive dimensions.")
            if self.method != "aligned_window":
                raise ValueError("source_window is only valid for aligned_window plans.")
            source_grid = self.source.source_grid
            if window[0] + window[2] > source_grid.width or window[1] + window[3] > source_grid.height:
                raise ValueError("source_window extends beyond the source grid.")
            if window[2:] != (self.target_grid.width, self.target_grid.height):
                raise ValueError("aligned source_window dimensions must match the target grid.")
            object.__setattr__(self, "source_window", window)
        elif self.method == "aligned_window":
            raise ValueError("aligned_window requires source_window.")
        object.__setattr__(self, "diagnostics", _label_diagnostics(self.diagnostics))

    @property
    def output_kind(self) -> str:
        return "semantic" if self.source.kind == "semantic" else "raster_instance"

    @property
    def output_suffix(self) -> str:
        return ".npy" if self.output_kind == "semantic" else ".npz"

    @property
    def requires_materialization(self) -> bool:
        return self.method != "exact" or self.source.path.suffix.lower() != self.output_suffix


@dataclass(frozen=True)
class PreparedLabelArtifact(_DictionaryRecord):
    """Validated target-sized label metadata; never carries an in-memory mask."""

    path: Path
    plan: LabelPreparationPlan
    sha256: str
    instance_id_map: tuple[tuple[int, int], ...] = ()
    diagnostics: tuple[LabelValidationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("artifact path must be a nonempty path.")
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.plan, LabelPreparationPlan):
            raise TypeError("plan must be a LabelPreparationPlan.")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        # Exact array archives retain the legacy byte-copy path. An exact-grid
        # GeoTIFF still needs format conversion to the training NPY contract.
        exact_archive = (self.plan.method == "exact"
                         and self.plan.source.path.suffix.lower() in (".npy", ".npz"))
        if exact_archive and self.sha256 != self.plan.source_sha256:
            raise ValueError("Exact-label artifacts must preserve source bytes/hash.")
        expected_suffix = ".npy" if self.kind == "semantic" else ".npz"
        if self.path.suffix.lower() != expected_suffix:
            raise ValueError(f"Prepared {self.kind} labels must use {expected_suffix}.")
        mapping = tuple(tuple(pair) for pair in self.instance_id_map)
        if any(len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in pair)
               for pair in mapping):
            raise ValueError("instance_id_map must contain positive integer ID pairs.")
        if [pair[0] for pair in mapping] != sorted(set(pair[0] for pair in mapping)):
            raise ValueError("Source instance IDs must be unique and ascending.")
        if [pair[1] for pair in mapping] != list(range(1, len(mapping) + 1)):
            raise ValueError("Target instance IDs must be compact 1..N.")
        if mapping and self.plan.source.kind == "semantic":
            raise ValueError("Semantic labels do not have an instance ID map.")
        object.__setattr__(self, "instance_id_map", mapping)
        object.__setattr__(self, "diagnostics", _label_diagnostics(self.diagnostics))

    @property
    def target_grid(self) -> TargetGrid:
        return self.plan.target_grid

    @property
    def kind(self) -> str:
        return "semantic" if self.plan.source.kind == "semantic" else "raster_instance"


@dataclass(frozen=True)
class ChipRequest(_DictionaryRecord):
    """One intended final chip and its complete target/query identity."""

    sample_id: str
    target_grid: TargetGrid
    geographic_aoi: GeographicAOI
    split_group_key: str
    label_path: Path | None = None
    label_grid: TargetGrid | None = None
    assigned_split: SplitName | None = None
    source_selectors: tuple[SourceSelector, ...] = ()
    reference_path: Path | None = None
    label_input: LabelInput | None = None
    requested_aoi: GeographicAOI | None = None

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
        if self.label_grid is not None and not isinstance(self.label_grid, TargetGrid):
            raise TypeError("label_grid must be a TargetGrid or None.")
        if self.requested_aoi is not None and not isinstance(self.requested_aoi, GeographicAOI):
            raise TypeError("requested_aoi must be a GeographicAOI or None.")
        if self.label_input is not None:
            if not isinstance(self.label_input, LabelInput):
                raise TypeError("label_input must be a LabelInput or None.")
            if self.label_path is not None and self.label_path != self.label_input.path:
                raise ValueError("label_path conflicts with label_input.path.")
            if self.label_grid is not None and self.label_grid != self.label_input.source_grid:
                raise ValueError("label_grid conflicts with label_input.source_grid.")
            object.__setattr__(self, "label_path", self.label_input.path)
            object.__setattr__(self, "label_grid", self.label_input.source_grid)
        elif self.label_path is not None:
            object.__setattr__(self, "label_input", LabelInput(self.label_path, source_grid=self.label_grid))
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
    assigned_split: AssignmentName | None = None
    resolved_label_path: Path | None = None
    label_diagnostics: tuple[LabelValidationDiagnostic, ...] = ()
    label_plan: LabelPreparationPlan | None = None

    def __post_init__(self) -> None:
        if self.label_plan is not None and not isinstance(self.label_plan, LabelPreparationPlan):
            raise TypeError("label_plan must be a LabelPreparationPlan or None.")
        if self.status not in PREFLIGHT_STATUSES:
            valid = ", ".join(PREFLIGHT_STATUSES)
            raise ValueError(f"preflight status must be one of {valid}.")
        if self.assigned_split is not None:
            split = str(self.assigned_split).strip().lower()
            if split not in ASSIGNMENT_NAMES:
                valid = ", ".join(ASSIGNMENT_NAMES)
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
class ChipDiagnostic:
    """One typed observation retained across the complete chip lifecycle."""

    stage: ChipDiagnosticStage
    code: str
    message: str
    severity: DiagnosticSeverity = "error"
    acquisition_group: str | None = None
    source_name: str | None = None
    zone: str | None = None
    zoom_level: int | None = None
    tile_x: int | None = None
    tile_y: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in CHIP_DIAGNOSTIC_STAGES:
            valid = ", ".join(CHIP_DIAGNOSTIC_STAGES)
            raise ValueError(f"chip diagnostic stage must be one of {valid}.")
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
        for name in ("acquisition_group", "source_name", "zone"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value).strip() or None)
        for name in ("zoom_level", "tile_x", "tile_y"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None.")


@dataclass(frozen=True)
class ChipResult:
    """Structured outcome for one request; no status-string tuples required."""

    request: ChipRequest
    status: ChipStatus
    preflight: ChipPreflight
    chip_path: Path | None = None
    label_path: Path | None = None
    cube_records: tuple[TileCubeRecord, ...] = ()
    effective_selectors: tuple[SourceSelector, ...] = ()
    diagnostic_path: Path | None = None
    message: str | None = None
    elapsed_seconds: float | None = None
    diagnostics: tuple[ChipDiagnostic, ...] = ()
    prepared_label: PreparedLabelArtifact | None = None

    def __post_init__(self) -> None:
        if self.prepared_label is not None:
            if not isinstance(self.prepared_label, PreparedLabelArtifact):
                raise TypeError("prepared_label must be a PreparedLabelArtifact or None.")
            if self.prepared_label.target_grid != self.request.target_grid:
                raise ValueError("Prepared label must match the request target grid.")
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
        selectors = tuple(self.effective_selectors)
        if any(not isinstance(item, SourceSelector) for item in selectors):
            raise TypeError(
                "effective_selectors must contain SourceSelector objects."
            )
        selector_keys = [
            (item.acquisition_group.casefold(), item.source_name.casefold())
            for item in selectors
        ]
        if len(set(selector_keys)) != len(selector_keys):
            raise ValueError(
                "effective_selectors must be unique by acquisition group and source."
            )
        object.__setattr__(self, "effective_selectors", selectors)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, ChipDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain ChipDiagnostic objects.")
        object.__setattr__(self, "diagnostics", diagnostics)
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


def chip_contract_to_dict(record) -> dict:
    """Encode a chip metadata record for JSON manifests and worker configuration."""
    supported = (GeographicAOI, TargetGrid, SourceSelector, LabelValidationDiagnostic,
                 LabelInput, LabelPreparationPlan, PreparedLabelArtifact, ChipRequest)

    def encode(value):
        if isinstance(value, supported):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported chip metadata value: {type(value).__name__}.")

    if not isinstance(record, supported):
        raise TypeError("Expected a chip metadata record.")
    return encode(record)


def _contract_from_dict(cls, document: Mapping):
    if not isinstance(document, Mapping):
        raise TypeError("Contract document must be a mapping.")
    nested = {
        LabelInput: {"source_grid": TargetGrid},
        LabelPreparationPlan: {"source": LabelInput, "target_grid": TargetGrid},
        PreparedLabelArtifact: {"plan": LabelPreparationPlan},
        ChipRequest: {"target_grid": TargetGrid, "label_grid": TargetGrid,
                      "geographic_aoi": GeographicAOI, "requested_aoi": GeographicAOI,
                      "label_input": LabelInput},
    }
    values = dict(document)
    for name, child_type in nested.get(cls, {}).items():
        if values.get(name) is not None:
            values[name] = _contract_from_dict(child_type, values[name])
    if "diagnostics" in values and cls in (LabelPreparationPlan, PreparedLabelArtifact):
        values["diagnostics"] = tuple(LabelValidationDiagnostic(**item) for item in values["diagnostics"])
    if cls is ChipRequest and "source_selectors" in values:
        values["source_selectors"] = tuple(SourceSelector(**item) for item in values["source_selectors"])
    return cls(**values)


def label_preparation_provenance(request: ChipRequest, preflight: ChipPreflight,
                                 result: ChipResult) -> dict:
    """Additive provenance shared by sample diagnostics and dataset manifests."""
    return {
        "requested_aoi": None if request.requested_aoi is None else request.requested_aoi.to_dict(),
        "label_input": None if request.label_input is None else request.label_input.to_dict(),
        "label_preparation_plan": None if preflight.label_plan is None else preflight.label_plan.to_dict(),
        "prepared_label": None if result.prepared_label is None else result.prepared_label.to_dict(),
    }


class LabelMismatchError(ValueError):
    """A label failed identity, archive, shape, or grid preflight for one sample."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        diagnostics: tuple[LabelValidationDiagnostic, ...] = (),
        label_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.sample_id = _sample_id(sample_id)
        self.label_path = None if label_path is None else Path(label_path)
        self.diagnostics = tuple(diagnostics)
        if any(
            not isinstance(item, LabelValidationDiagnostic)
            for item in self.diagnostics
        ):
            raise TypeError(
                "diagnostics must contain LabelValidationDiagnostic objects."
            )


__all__ = [
    "CHIP_DIAGNOSTIC_STAGES",
    "CHIP_STATUSES",
    "DIAGNOSTIC_SEVERITIES",
    "PREFLIGHT_STATUSES",
    "ChipDiagnostic",
    "ChipDiagnosticStage",
    "ChipPreflight",
    "ChipRequest",
    "ChipResult",
    "ChipStatus",
    "DiagnosticSeverity",
    "GeographicAOI",
    "LabelMismatchError",
    "LabelInput",
    "LabelPreparationPlan",
    "PreparedLabelArtifact",
    "LabelValidationDiagnostic",
    "PreflightStatus",
    "ReferenceSample",
    "SourceSelector",
    "TargetGrid",
    "validate_request_contracts",
    "chip_contract_to_dict",
    "label_preparation_provenance",
]
