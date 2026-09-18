"""Strict label resolution and validation before chip acquisition."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from .chip_config import AssignmentName
from .chip_requests import (
    normalize_sample_id,
    validate_target_grid_consistency,
)
from .chip_types import (
    ChipPreflight,
    ChipRequest,
    LabelMismatchError,
    LabelValidationDiagnostic,
    TargetGrid,
)


LABEL_SUFFIXES = (".npy", ".npz")


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to validate chip labels.") from exc
    return np


def _label_error(
    request: ChipRequest,
    *,
    code: str,
    message: str,
    expected: object | None = None,
    actual: object | None = None,
) -> LabelMismatchError:
    diagnostic = LabelValidationDiagnostic(
        code=code,
        message=message,
        expected=None if expected is None else str(expected),
        actual=None if actual is None else str(actual),
    )
    return LabelMismatchError(
        message,
        sample_id=request.sample_id,
        diagnostics=(diagnostic,),
    )


def _validate_label_identity(request: ChipRequest, path: Path) -> None:
    actual = normalize_sample_id(path)
    if actual.casefold() != request.sample_id.casefold():
        raise _label_error(
            request,
            code="label_identity_mismatch",
            message=(
                f"Label {path} normalizes to sample {actual!r}, not request "
                f"{request.sample_id!r}."
            ),
            expected=request.sample_id,
            actual=actual,
        )


def resolve_label_path(
    request: ChipRequest,
    label_source: str | Path,
) -> Path:
    """Resolve exactly one full-sample-ID label for a request."""
    if request.label_path is not None:
        path = request.label_path
        if not path.is_file():
            raise _label_error(
                request,
                code="missing_label",
                message=f"Explicit label does not exist: {path}",
                expected=request.sample_id,
                actual=path,
            )
        candidates = (path,)
    else:
        source = Path(label_source)
        if source.is_file():
            candidates = (source,)
        elif source.is_dir():
            matching_candidates: list[Path] = []
            for path in source.iterdir():
                if not path.is_file() or path.suffix.casefold() not in LABEL_SUFFIXES:
                    continue
                try:
                    candidate_id = normalize_sample_id(path)
                except ValueError:
                    continue
                if candidate_id.casefold() == request.sample_id.casefold():
                    matching_candidates.append(path)
            candidates = tuple(
                sorted(
                    matching_candidates,
                    key=lambda path: str(path).casefold(),
                )
            )
        else:
            raise _label_error(
                request,
                code="missing_label_source",
                message=f"Label source does not exist: {source}",
                actual=source,
            )
    if not candidates:
        raise _label_error(
            request,
            code="missing_label",
            message=f"No label matched full sample ID {request.sample_id!r}.",
            expected=request.sample_id,
        )
    if len(candidates) > 1:
        raise _label_error(
            request,
            code="duplicate_labels",
            message=(
                f"Multiple labels matched full sample ID {request.sample_id!r}: "
                + ", ".join(str(path) for path in candidates)
            ),
            expected="exactly one label",
            actual=len(candidates),
        )
    path = candidates[0]
    if path.suffix.casefold() not in LABEL_SUFFIXES:
        raise _label_error(
            request,
            code="unsupported_label_type",
            message=f"Unsupported label file type: {path.suffix}",
            expected=LABEL_SUFFIXES,
            actual=path.suffix,
        )
    _validate_label_identity(request, path)
    return path


def _validate_mask(request: ChipRequest, mask: Any) -> None:
    np = _numpy()
    if mask.ndim != 2:
        raise _label_error(
            request,
            code="label_shape_mismatch",
            message=f"Label mask must be two-dimensional, got shape {mask.shape}.",
            expected=(request.target_grid.height, request.target_grid.width),
            actual=mask.shape,
        )
    expected_shape = (request.target_grid.height, request.target_grid.width)
    if tuple(mask.shape) != expected_shape:
        raise _label_error(
            request,
            code="label_shape_mismatch",
            message=(
                f"Label mask shape {mask.shape} does not match target-grid shape "
                f"{expected_shape}."
            ),
            expected=expected_shape,
            actual=mask.shape,
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise _label_error(
            request,
            code="invalid_label_dtype",
            message=f"Label mask must have an integer dtype, got {mask.dtype}.",
            expected="integer dtype",
            actual=mask.dtype,
        )


def _validate_instance_archive(
    request: ChipRequest,
    archive: Mapping[str, Any],
) -> tuple[LabelValidationDiagnostic, ...]:
    """Validate archive structure and return nonfatal occlusion diagnostics."""
    np = _numpy()
    required = {"mask", "bboxes", "num_craters"}
    missing = sorted(required - set(archive))
    if missing:
        raise _label_error(
            request,
            code="malformed_instance_label",
            message=f"Instance label is missing required arrays: {', '.join(missing)}.",
            expected=sorted(required),
            actual=sorted(archive),
        )
    mask = np.asarray(archive["mask"])
    _validate_mask(request, mask)
    bboxes = np.asarray(archive["bboxes"])
    count_array = np.asarray(archive["num_craters"])
    if count_array.ndim != 0 or not np.issubdtype(count_array.dtype, np.integer):
        raise _label_error(
            request,
            code="malformed_instance_label",
            message="num_craters must be one scalar integer.",
            expected="scalar integer",
            actual=f"shape={count_array.shape}, dtype={count_array.dtype}",
        )
    count = int(count_array.item())
    if count < 0:
        raise _label_error(
            request,
            code="malformed_instance_label",
            message="num_craters must be nonnegative.",
            expected=">= 0",
            actual=count,
        )
    if bboxes.shape != (count, 4):
        raise _label_error(
            request,
            code="malformed_instance_label",
            message=(
                f"bboxes shape {bboxes.shape} does not match num_craters={count}."
            ),
            expected=(count, 4),
            actual=bboxes.shape,
        )
    if not np.issubdtype(bboxes.dtype, np.number) or not np.isfinite(bboxes).all():
        raise _label_error(
            request,
            code="malformed_instance_label",
            message="bboxes must contain only finite numeric values.",
        )
    if count:
        x, y, width, height = (bboxes[:, index] for index in range(4))
        inside = (
            (x >= 0)
            & (y >= 0)
            & (width > 0)
            & (height > 0)
            & (x + width <= request.target_grid.width)
            & (y + height <= request.target_grid.height)
        )
        if not bool(np.all(inside)):
            raise _label_error(
                request,
                code="malformed_instance_label",
                message=(
                    "bboxes must use finite COCO x/y/width/height values within "
                    "the target pixel extent."
                ),
                expected=(request.target_grid.width, request.target_grid.height),
            )
    positive_ids = np.unique(mask[mask > 0])
    if bool(np.any(mask < 0)):
        raise _label_error(
            request,
            code="malformed_instance_label",
            message="Instance masks may contain only background 0 and IDs 1..N.",
        )
    present_ids = {int(value) for value in positive_ids}
    expected_ids = set(range(1, count + 1))
    out_of_range = tuple(sorted(present_ids - expected_ids))
    if out_of_range:
        raise _label_error(
            request,
            code="malformed_instance_label",
            message=(
                "Positive mask IDs must fall within 1..num_craters; found "
                f"out-of-range IDs {out_of_range}."
            ),
            expected=tuple(sorted(expected_ids)),
            actual=tuple(sorted(present_ids)),
        )

    missing_ids = tuple(sorted(expected_ids - present_ids))
    occluding_ids: dict[int, tuple[int, ...]] = {}
    unexplained_ids: list[int] = []
    for instance_id in missing_ids:
        x, y, width, height = bboxes[instance_id - 1]
        left = max(0, int(math.floor(float(x))))
        top = max(0, int(math.floor(float(y))))
        right = min(mask.shape[1], int(math.ceil(float(x + width))))
        bottom = min(mask.shape[0], int(math.ceil(float(y + height))))
        overlapping = tuple(
            sorted(
                int(value)
                for value in np.unique(mask[top:bottom, left:right])
                if int(value) > 0
            )
        )
        if overlapping:
            occluding_ids[instance_id] = overlapping
        else:
            unexplained_ids.append(instance_id)
    if unexplained_ids:
        unexplained = tuple(unexplained_ids)
        raise _label_error(
            request,
            code="malformed_instance_label",
            message=(
                "Mask omits annotated instance IDs whose bounding-box regions "
                f"contain no other instance pixels: {unexplained}."
            ),
            expected="every absent ID to have overlap evidence",
            actual=unexplained,
        )
    if not occluding_ids:
        return ()
    overlap_summary = ", ".join(
        f"{instance_id}->{values}"
        for instance_id, values in sorted(occluding_ids.items())
    )
    return (
        LabelValidationDiagnostic(
            code="occluded_instance_ids",
            message=(
                f"Mask omits annotated instance IDs {missing_ids}; their "
                "bounding-box regions contain pixels assigned to other "
                f"instances ({overlap_summary}), consistent with overlap "
                "during mask rasterization."
            ),
            severity="warning",
            expected=str(tuple(sorted(expected_ids))),
            actual=str(tuple(sorted(present_ids))),
        ),
    )


def _sidecar_path(label_path: Path) -> Path | None:
    candidates = tuple(
        path
        for path in (
            label_path.with_suffix(label_path.suffix + ".json"),
            label_path.with_suffix(".json"),
        )
        if path.is_file()
    )
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(
            f"Multiple geospatial sidecars found for label {label_path}: {unique}"
        )
    return unique[0] if unique else None


def _grid_from_metadata(metadata: Mapping[str, Any]) -> TargetGrid:
    grid_values = metadata.get("target_grid", metadata)
    if not isinstance(grid_values, Mapping):
        raise ValueError("Label sidecar target_grid must be an object.")
    required = ("crs_wkt", "transform", "bounds", "width", "height")
    missing = tuple(name for name in required if name not in grid_values)
    if missing:
        raise ValueError(
            "Label sidecar target_grid is missing: " + ", ".join(missing)
        )
    grid = TargetGrid(
        crs_wkt=grid_values["crs_wkt"],
        transform=grid_values["transform"],
        bounds=grid_values["bounds"],
        width=grid_values["width"],
        height=grid_values["height"],
    )
    validate_target_grid_consistency(grid)
    return grid


def _crs_is_same(first_wkt: str, second_wkt: str) -> bool:
    if "".join(first_wkt.split()) == "".join(second_wkt.split()):
        return True
    try:
        from osgeo import osr
    except ImportError:
        return False
    try:
        first = osr.SpatialReference()
        second = osr.SpatialReference()
        if (
            first.ImportFromWkt(first_wkt) != 0
            or second.ImportFromWkt(second_wkt) != 0
        ):
            return False
        return bool(first.IsSame(second))
    except Exception:
        return False


def _validate_grid_match(
    request: ChipRequest,
    label_grid: TargetGrid,
) -> None:
    target = request.target_grid
    if not _crs_is_same(target.crs_wkt, label_grid.crs_wkt):
        raise _label_error(
            request,
            code="label_grid_mismatch",
            message="Label CRS does not match the target-grid CRS.",
            expected=target.crs_wkt,
            actual=label_grid.crs_wkt,
        )
    if (label_grid.width, label_grid.height) != (target.width, target.height):
        raise _label_error(
            request,
            code="label_grid_mismatch",
            message="Label grid dimensions do not match the target grid.",
            expected=(target.width, target.height),
            actual=(label_grid.width, label_grid.height),
        )
    comparisons = (
        (label_grid.transform, target.transform, "transform"),
        (label_grid.bounds, target.bounds, "bounds"),
    )
    for actual, expected, name in comparisons:
        scale = max(*(abs(value) for value in (*actual, *expected)), 1.0)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=scale * 1e-10)
            for left, right in zip(actual, expected)
        ):
            raise _label_error(
                request,
                code="label_grid_mismatch",
                message=f"Label grid {name} does not match the target grid.",
                expected=expected,
                actual=actual,
            )


def _validate_sidecar(request: ChipRequest, label_path: Path) -> bool:
    try:
        sidecar = _sidecar_path(label_path)
        if sidecar is None:
            return False
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError("Label sidecar must contain a JSON object.")
        sidecar_sample = metadata.get("sample_id")
        if (
            sidecar_sample is not None
            and str(sidecar_sample).casefold() != request.sample_id.casefold()
        ):
            raise ValueError(
                f"Label sidecar sample_id {sidecar_sample!r} does not match "
                f"{request.sample_id!r}."
            )
        _validate_grid_match(request, _grid_from_metadata(metadata))
        return True
    except LabelMismatchError:
        raise
    except Exception as exc:
        raise _label_error(
            request,
            code="malformed_label_metadata",
            message=f"Could not validate label sidecar metadata: {exc}",
        ) from exc


def validate_label(
    request: ChipRequest,
    path: str | Path,
) -> tuple[LabelValidationDiagnostic, ...]:
    """Validate label identity, contents, shape, and available grid metadata."""
    np = _numpy()
    label_path = Path(path)
    _validate_label_identity(request, label_path)
    content_diagnostics: tuple[LabelValidationDiagnostic, ...] = ()
    try:
        has_grid_metadata = request.label_grid is not None
        if request.label_grid is not None:
            validate_target_grid_consistency(request.label_grid)
            _validate_grid_match(request, request.label_grid)
        if label_path.suffix.casefold() == ".npy":
            mask = np.load(label_path, mmap_mode="r", allow_pickle=False)
            _validate_mask(request, np.asarray(mask))
            has_grid_metadata = (
                _validate_sidecar(request, label_path) or has_grid_metadata
            )
        elif label_path.suffix.casefold() == ".npz":
            with np.load(label_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            content_diagnostics = _validate_instance_archive(request, arrays)
            has_grid_metadata = (
                _validate_sidecar(request, label_path) or has_grid_metadata
            )
        else:
            raise _label_error(
                request,
                code="unsupported_label_type",
                message=f"Unsupported label file type: {label_path.suffix}",
            )
    except LabelMismatchError:
        raise
    except Exception as exc:
        raise _label_error(
            request,
            code="malformed_label",
            message=f"Could not read or validate label {label_path}: {exc}",
        ) from exc
    diagnostics = [
        LabelValidationDiagnostic(
            code="label_validated",
            message="Label identity, contents, and spatial shape are valid.",
            severity="info",
        )
    ]
    diagnostics.extend(content_diagnostics)
    if not has_grid_metadata:
        diagnostics.append(
            LabelValidationDiagnostic(
                code="label_grid_unverified",
                message=(
                    "The array label has no independent geospatial metadata; "
                    "identity and spatial shape are the verifiable boundary."
                ),
                severity="warning",
            )
        )
    return tuple(diagnostics)


def preflight_label(
    request: ChipRequest,
    *,
    label_source: str | Path,
    assigned_split: AssignmentName,
) -> ChipPreflight:
    """Resolve and validate one label, raising a typed per-sample error."""
    path = resolve_label_path(request, label_source)
    try:
        diagnostics = validate_label(request, path)
    except LabelMismatchError as exc:
        if exc.label_path is None:
            exc.label_path = path
        raise
    return ChipPreflight(
        status="passed",
        assigned_split=assigned_split,
        resolved_label_path=path,
        label_diagnostics=diagnostics,
    )


__all__ = [
    "LABEL_SUFFIXES",
    "preflight_label",
    "resolve_label_path",
    "validate_label",
]
