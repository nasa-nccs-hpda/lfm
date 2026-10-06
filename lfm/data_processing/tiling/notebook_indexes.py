"""Notebook-facing shared-index and custom-cache path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal


DEFAULT_PROJECT_DATA_DIR = Path("/explore/nobackup/projects/lfm")
DEFAULT_SHARED_INDEX_NAME = "output_index.gpkg"
DEFAULT_SOURCE_DATA_DIRS = MappingProxyType(
    {
        "wac": (
            DEFAULT_PROJECT_DATA_DIR
            / "processed_data/Lunar/LRO_WAC_Pho_Sites"
        ),
        "nac": (
            DEFAULT_PROJECT_DATA_DIR
            / "processed_data/Lunar/LRO_NAC_Pho_Sites"
        ),
        "static": DEFAULT_PROJECT_DATA_DIR / "staticLinks",
    }
)


@dataclass(frozen=True)
class NotebookIndexResolution:
    """Resolved shared or user-owned index policy for one notebook source."""

    source_name: str
    data_dir: Path
    index_path: Path
    uses_shared_default: bool
    rebuild_invalid_index: bool


def _normalized_path(path: str | Path) -> Path:
    """Return a comparison-safe absolute path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


def resolve_notebook_source_index(
    *,
    source_name: Literal["wac", "nac", "static"],
    data_dir: str | Path,
    cache_dir: str | Path,
    default_data_dir: str | Path | None = None,
    shared_index_name: str = DEFAULT_SHARED_INDEX_NAME,
) -> NotebookIndexResolution:
    """Choose a protected shared index or a replaceable per-clone cache.

    The repository's canonical WAC, NAC, and static directories already own a
    validated ``output_index.gpkg``. Those indexes are shared read-only and
    must exist. A caller-supplied data directory instead receives a persistent
    GeoPackage path beneath ``cache_dir``; that application-owned cache may be
    created or rebuilt by the high-level preparation workflow.

    ``default_data_dir`` is primarily useful to applications that mirror the
    canonical directory layout at another explicitly declared location and to
    focused tests. When omitted, the canonical Explore path for the modality
    is used.
    """
    normalized_name = str(source_name).strip().casefold()
    if normalized_name not in DEFAULT_SOURCE_DATA_DIRS:
        valid = ", ".join(DEFAULT_SOURCE_DATA_DIRS)
        raise ValueError(
            f"source_name must be one of {valid}, got {source_name!r}."
        )
    index_name = Path(shared_index_name)
    if (
        index_name.name != shared_index_name
        or index_name.suffix.lower() != ".gpkg"
    ):
        raise ValueError("shared_index_name must be a .gpkg filename.")

    source_dir = Path(data_dir)
    canonical_dir = Path(
        DEFAULT_SOURCE_DATA_DIRS[normalized_name]
        if default_data_dir is None
        else default_data_dir
    )
    uses_shared_default = _normalized_path(source_dir) == _normalized_path(
        canonical_dir
    )
    if uses_shared_default:
        index_path = source_dir / shared_index_name
        if not index_path.is_file():
            raise FileNotFoundError(
                f"The default {normalized_name.upper()} directory requires its "
                f"shared raster index, but it was not found: {index_path}"
            )
        rebuild_invalid_index = False
    else:
        index_path = Path(cache_dir) / f"{normalized_name}_index.gpkg"
        rebuild_invalid_index = True

    return NotebookIndexResolution(
        source_name=normalized_name,
        data_dir=source_dir,
        index_path=index_path,
        uses_shared_default=uses_shared_default,
        rebuild_invalid_index=rebuild_invalid_index,
    )


__all__ = [
    "DEFAULT_PROJECT_DATA_DIR",
    "DEFAULT_SHARED_INDEX_NAME",
    "DEFAULT_SOURCE_DATA_DIRS",
    "NotebookIndexResolution",
    "resolve_notebook_source_index",
]
