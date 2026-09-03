"""Shared configuration helpers and defaults for lunar tiling workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from model import (
    BandNoDataOverride,
    MINIRF_SOURCE_NODATA,
    MINIRF_SOURCE_NODATA_BANDS,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
    TileSourceConfig,
)


# Shared notebook-workflow defaults. Raster band numbers are 1-based.
DEFAULT_ZOOM_LEVEL = 5
DEFAULT_WAC_BAND_NUMBER = 3  # First WAC VIS band after the two UV bands.
DEFAULT_NAC_BAND_NUMBER = 1


def create_tiling_run_id(now: datetime | None = None) -> str:
    """Return a sortable timestamp identifier for one tiling notebook run."""
    timestamp = now if now is not None else datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


# Importing the helper starts one timestamped notebook run.
RUN_ID = create_tiling_run_id()


def validate_path_pairs(
    label_path_pairs: Mapping[str, str | Path]
    | Iterable[tuple[str, str | Path]],
    *,
    path_type: Literal["file", "directory"],
) -> dict[str, Path]:
    """Validate labeled configured paths and return normalized ``Path`` values."""
    if path_type not in {"file", "directory"}:
        raise ValueError("path_type must be 'file' or 'directory'.")
    items = (
        label_path_pairs.items()
        if isinstance(label_path_pairs, Mapping)
        else label_path_pairs
    )
    normalized: dict[str, Path] = {}
    errors: list[str] = []
    for raw_label, raw_path in items:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("Configured path labels must not be empty.")
        if label in normalized:
            raise ValueError(f"Duplicate configured path label: {label!r}.")
        path = Path(raw_path)
        normalized[label] = path
        if path_type == "file" and not path.is_file():
            errors.append(f"{label} is not an existing file: {path}")
        if path_type == "directory" and not path.is_dir():
            errors.append(f"{label} is not an existing directory: {path}")
    if not normalized:
        raise ValueError("At least one labeled path must be provided.")
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise FileNotFoundError(f"Invalid configured {path_type} paths:\n{details}")
    return normalized


def make_static_source(
    *,
    data_dir: str | Path,
    index_path: str | Path,
    index_layer: str | None = None,
    location_field: str = "location",
    required: bool = True,
) -> TileSourceConfig:
    """Create the canonical 63-band static lunar tiling source config."""
    return TileSourceConfig(
        name="static",
        data_dir=Path(data_dir),
        index_path=Path(index_path),
        index_layer=index_layer,
        location_field=location_field,
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
        required=required,
    )


__all__ = [
    "DEFAULT_NAC_BAND_NUMBER",
    "DEFAULT_WAC_BAND_NUMBER",
    "DEFAULT_ZOOM_LEVEL",
    "RUN_ID",
    "create_tiling_run_id",
    "make_static_source",
    "validate_path_pairs",
]
