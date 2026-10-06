"""High-level AOI tiling with optional per-source product discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import logging

from .tiling import create_tiles_for_aoi as _create_tiles_for_aoi_strict
from .tiling_config import TileConfig, TileSourceConfig
from .tiling_policy import group_source_rasters_by_product
from .tiling_results import (
    TileCubeRecord,
    TileSourceError,
    safe_filename_component,
)
from .vector_index import query_source_index


LOGGER = logging.getLogger(__name__)


class MissingRequiredProductError(RuntimeError):
    """A required product-scoped source produced no selectable product."""

    def __init__(
        self,
        message: str,
        *,
        source_name: str,
        product_id: str | None = None,
        completed_records: tuple[TileCubeRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.source_name = source_name
        self.product_id = product_id
        self.completed_records = completed_records


def _normalize_product_requests(
    config: TileConfig,
    product_ids: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {}
    for raw_name, raw_product_id in (product_ids or {}).items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Product-ID source names must not be empty.")
        if name in normalized:
            raise ValueError(f"Duplicate product-ID source name: {name!r}.")
        if raw_product_id is None:
            normalized[name] = None
            continue
        product_id = str(raw_product_id).strip()
        if not product_id:
            raise ValueError(
                f"Product ID for source {name!r} must be non-empty or None."
            )
        normalized[name] = product_id

    sources = {source.name: source for source in config.sources}
    unknown = sorted(set(normalized) - set(sources))
    if unknown:
        raise KeyError(f"Product IDs reference unknown tile sources: {unknown}")
    contextual = sorted(
        name
        for name in normalized
        if sources[name].selection_mode != "product_id"
    )
    if contextual:
        raise ValueError(
            "Product IDs may only target product_id sources; invalid sources: "
            f"{contextual}."
        )
    return normalized


def _validate_filename_product_ids(
    source: TileSourceConfig,
    product_ids: tuple[str, ...],
) -> None:
    by_component: dict[str, str] = {}
    for product_id in product_ids:
        component = safe_filename_component(product_id)
        previous = by_component.get(component)
        if previous is not None and previous != product_id:
            raise ValueError(
                f"Source {source.name!r} product IDs {previous!r} and "
                f"{product_id!r} map to the same filename component "
                f"{component!r}."
            )
        by_component[component] = product_id


def discover_products_for_aoi(
    config: TileConfig,
    *,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    product_ids: Mapping[str, str | None] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, tuple[str, ...]]:
    """Resolve explicit or discovered PIDs for every product-scoped source."""
    requested = _normalize_product_requests(config, product_ids)
    active_logger = logger or LOGGER
    selections: dict[str, tuple[str, ...]] = {}

    for source in config.sources:
        if source.selection_mode != "product_id":
            continue
        indexed = query_source_index(
            source,
            ul_lat=ul_lat,
            ul_lon=ul_lon,
            lr_lat=lr_lat,
            lr_lon=lr_lon,
        )
        grouped = group_source_rasters_by_product(source, indexed)
        explicit_product_id = requested.get(source.name)
        if explicit_product_id is None:
            selected = tuple(grouped)
            active_logger.info(
                "Discovered %d product ID(s) for source %r: %s",
                len(selected),
                source.name,
                ", ".join(selected) if selected else "none",
            )
        else:
            selected = (
                (explicit_product_id,)
                if explicit_product_id in grouped
                else ()
            )
            active_logger.info(
                "Using explicit product ID %r for source %r.",
                explicit_product_id,
                source.name,
            )

        if not selected and source.required:
            if explicit_product_id is None:
                detail = "no products intersect the AOI"
            else:
                detail = (
                    f"product {explicit_product_id!r} does not intersect the AOI"
                )
            raise MissingRequiredProductError(
                f"Required source {source.name!r} has {detail}.",
                source_name=source.name,
                product_id=explicit_product_id,
            )
        _validate_filename_product_ids(source, selected)
        selections[source.name] = selected
    return selections


def _ordered_records(
    config: TileConfig,
    records: list[TileCubeRecord],
) -> list[TileCubeRecord]:
    source_order = {
        source.name: position for position, source in enumerate(config.sources)
    }
    return sorted(
        records,
        key=lambda record: (
            record.zone,
            record.tile_y,
            record.tile_x,
            source_order[record.source_name],
            record.product_id or "",
        ),
    )


def create_tiles_for_aoi_by_product(
    config: TileConfig,
    *,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    product_ids: Mapping[str, str | None] | None = None,
    logger: logging.Logger | None = None,
) -> list[TileCubeRecord]:
    """Tile an AOI with explicit PIDs or per-source PID discovery.

    This is the high-level optional-product path. The strict low-level
    :func:`create_tiles_for_aoi` API still requires selectors for every
    ``product_id`` source. Product-scoped sources run once per selected PID;
    ``all_intersecting`` sources run once total so contextual outputs are not
    duplicated or overwritten.
    """
    bounds = {
        "ul_lat": ul_lat,
        "ul_lon": ul_lon,
        "lr_lat": lr_lat,
        "lr_lon": lr_lon,
    }
    selections = discover_products_for_aoi(
        config,
        **bounds,
        product_ids=product_ids,
        logger=logger,
    )
    records: list[TileCubeRecord] = []

    for source in config.sources:
        if source.selection_mode != "product_id":
            continue
        for product_id in selections[source.name]:
            run_source = replace(source, required=False)
            run_config = TileConfig(
                output_dir=config.output_dir,
                zoom_level=config.zoom_level,
                sources=(run_source,),
                debug=config.debug,
            )
            try:
                product_records = _create_tiles_for_aoi_strict(
                    run_config,
                    **bounds,
                    selectors={source.name: product_id},
                )
            except TileSourceError as exc:
                exc.completed_records = (
                    tuple(records) + tuple(exc.completed_records)
                )
                raise
            if source.required and not product_records:
                raise MissingRequiredProductError(
                    f"Required source {source.name!r} product {product_id!r} "
                    "produced no tile cubes for the AOI.",
                    source_name=source.name,
                    product_id=product_id,
                    completed_records=tuple(records),
                )
            for record in product_records:
                if record.source_name != source.name or record.product_id != product_id:
                    raise RuntimeError(
                        "Product-scoped tiling returned inconsistent structured "
                        f"metadata for source {source.name!r}, product "
                        f"{product_id!r}: {record}."
                    )
            records.extend(product_records)

    contextual_sources = tuple(
        source
        for source in config.sources
        if source.selection_mode == "all_intersecting"
    )
    if contextual_sources:
        contextual_config = TileConfig(
            output_dir=config.output_dir,
            zoom_level=config.zoom_level,
            sources=contextual_sources,
            debug=config.debug,
        )
        try:
            contextual_records = _create_tiles_for_aoi_strict(
                contextual_config,
                **bounds,
            )
        except TileSourceError as exc:
            exc.completed_records = tuple(records) + tuple(exc.completed_records)
            raise
        if any(record.product_id is not None for record in contextual_records):
            raise RuntimeError(
                "Contextual tiling returned an unexpected product-scoped record."
            )
        records.extend(contextual_records)

    return _ordered_records(config, records)


__all__ = [
    "MissingRequiredProductError",
    "create_tiles_for_aoi_by_product",
    "discover_products_for_aoi",
]
