"""Dependency-free source selection and NoData policies for lunar tiling."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

from .product_ids import (
    ProductIdResolutionError,
    lunar_product_id_from_raster_path,
)
from .tiling_config import TileSourceConfig
from .vector_index import IndexedRaster


def product_id_from_raster_path(path: str | Path) -> str:
    """Return the legacy lunar product ID prefix from a raster filename."""
    return lunar_product_id_from_raster_path(path)


def resolve_source_product_id(
    source: TileSourceConfig,
    path: str | Path,
) -> str:
    """Resolve and validate one product ID using its source's declared policy."""
    raster_path = Path(path)
    try:
        resolved = source.product_id_resolver(raster_path)
    except Exception as exc:
        raise ProductIdResolutionError(
            f"Source {source.name!r} could not resolve a product ID for "
            f"{raster_path}: {exc}"
        ) from exc
    product_id = str(resolved).strip()
    if not product_id:
        raise ProductIdResolutionError(
            f"Source {source.name!r} resolved an empty product ID for "
            f"{raster_path}."
        )
    return product_id


def group_source_rasters_by_product(
    source: TileSourceConfig,
    records: Sequence[IndexedRaster],
) -> dict[str, tuple[IndexedRaster, ...]]:
    """Group companion rasters by resolved PID in deterministic order."""
    grouped: dict[str, list[IndexedRaster]] = {}
    ordered = sorted(
        records,
        key=lambda item: (str(item.path), item.feature_id or -1),
    )
    for record in ordered:
        product_id = resolve_source_product_id(source, record.path)
        grouped.setdefault(product_id, []).append(record)
    return {
        product_id: tuple(grouped[product_id])
        for product_id in sorted(grouped)
    }


def select_source_rasters(
    source: TileSourceConfig,
    records: list[IndexedRaster],
    *,
    selector: str | None = None,
) -> list[IndexedRaster]:
    """Apply a source's declared selection policy to indexed rasters."""
    if source.selection_mode == "all_intersecting":
        if selector is not None:
            raise ValueError(
                f"Source {source.name!r} uses all_intersecting and does not "
                "accept a product selector."
            )
        return list(records)

    if selector is None or not str(selector).strip():
        raise ValueError(
            f"Source {source.name!r} uses product_id selection and requires "
            "a non-empty per-query selector."
        )
    product_id = str(selector).strip()
    return [
        record
        for record in records
        if resolve_source_product_id(source, record.path) == product_id
    ]


def band_nodata_values(
    source: TileSourceConfig,
    *,
    band_name: str,
    metadata_source_nodata: float | None,
) -> tuple[float | None, float | None]:
    """Resolve source and output NoData values for one named band."""
    override = source.nodata_override_for(band_name)
    source_nodata = (
        override.source_value
        if override is not None and override.source_value is not None
        else source.source_nodata
        if source.source_nodata is not None
        else metadata_source_nodata
    )
    preserve_source = (
        override.preserve_source
        if override is not None
        else source.preserve_source_nodata
    )
    if preserve_source:
        return source_nodata, source_nodata
    output_nodata = (
        override.output_value
        if override is not None and override.output_value is not None
        else source.output_nodata
        if source.output_nodata is not None
        else source_nodata
    )
    return source_nodata, output_nodata


def validate_source_selectors(
    sources: Sequence[TileSourceConfig],
    selectors: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate per-query selectors against configured source policies."""
    normalized = {
        str(name): str(value).strip() for name, value in (selectors or {}).items()
    }
    source_names = {source.name for source in sources}
    unknown = sorted(set(normalized) - source_names)
    if unknown:
        raise KeyError(f"Selectors reference unknown tile sources: {unknown}")
    for source in sources:
        selector = normalized.get(source.name)
        if source.selection_mode == "product_id" and not selector:
            raise ValueError(f"Source {source.name!r} requires a product selector.")
        if source.selection_mode == "all_intersecting" and selector is not None:
            raise ValueError(
                f"Source {source.name!r} does not accept a product selector."
            )
    return normalized


__all__ = [
    "band_nodata_values",
    "group_source_rasters_by_product",
    "product_id_from_raster_path",
    "resolve_source_product_id",
    "select_source_rasters",
    "validate_source_selectors",
]
