"""Dynamic/static source-mode composition for high-level tiling workflows."""

from __future__ import annotations

from collections.abc import Iterable

from .tiling_config import TileSourceConfig


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _enabled_source_tuple(
    values: Iterable[TileSourceConfig] | None,
    *,
    role: str,
) -> tuple[TileSourceConfig, ...]:
    sources = tuple(() if values is None else values)
    if not sources:
        raise ValueError(
            f"include_{role}=True requires at least one {role} source."
        )
    if any(not isinstance(source, TileSourceConfig) for source in sources):
        raise TypeError(f"{role}_sources must contain TileSourceConfig objects.")
    if role == "static":
        product_scoped = sorted(
            source.name
            for source in sources
            if source.selection_mode != "all_intersecting"
        )
        if product_scoped:
            raise ValueError(
                "Static sources must use selection_mode='all_intersecting'; "
                f"product-scoped static sources: {product_scoped}."
            )
    return sources


def compose_tile_sources(
    *,
    dynamic_sources: Iterable[TileSourceConfig] | None = None,
    static_sources: Iterable[TileSourceConfig] | None = None,
    include_dynamic: bool = True,
    include_static: bool = True,
) -> tuple[TileSourceConfig, ...]:
    """Return enabled sources in deterministic dynamic-then-static order.

    Disabled collections are deliberately not iterated or validated. Callers
    therefore do not need to supply paths, indexes, selectors, or even source
    objects for a disabled class. Required/optional behavior remains attached
    to each returned :class:`TileSourceConfig` and is enforced by the existing
    tiling APIs.
    """
    use_dynamic = _require_bool(
        include_dynamic,
        field_name="include_dynamic",
    )
    use_static = _require_bool(
        include_static,
        field_name="include_static",
    )
    if not use_dynamic and not use_static:
        raise ValueError(
            "At least one of include_dynamic or include_static must be True."
        )

    dynamic = (
        _enabled_source_tuple(dynamic_sources, role="dynamic")
        if use_dynamic
        else ()
    )
    static = (
        _enabled_source_tuple(static_sources, role="static")
        if use_static
        else ()
    )
    sources = dynamic + static
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        duplicates = sorted(
            name for name in set(names) if names.count(name) > 1
        )
        raise ValueError(
            f"Enabled tile source names must be unique: {duplicates}."
        )
    return sources


__all__ = ["compose_tile_sources"]
