"""Public configuration-driven lunar tiling API."""

from __future__ import annotations

from collections.abc import Mapping

from .tiling_config import TileConfig
from .tiling_results import TileCubeRecord


def _tiler_cls():
    from .configured_tiler import ConfiguredTiler

    return ConfiguredTiler


def _grid_id(*, zone: str | None, grid_id: str | None) -> str:
    if zone is None and grid_id is None:
        raise ValueError("A zone or grid_id is required.")
    if zone is not None and grid_id is not None and zone != grid_id:
        raise ValueError(
            f"zone {zone!r} and grid_id {grid_id!r} identify different grids."
        )
    selected = grid_id if grid_id is not None else zone
    if selected is None or not str(selected).strip():
        raise ValueError("zone/grid_id must not be empty.")
    return str(selected).strip()


def create_tiles_for_index(
    config: TileConfig,
    *,
    tile_x: int,
    tile_y: int,
    zone: str | None = None,
    grid_id: str | None = None,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create source cubes for one explicit grid tile address."""
    selected_grid = _grid_id(zone=zone, grid_id=grid_id)
    return _tiler_cls()(config, selectors=selectors).run_tile_index(
        tile_x,
        tile_y,
        selected_grid,
    )


def create_tiles_for_point(
    config: TileConfig,
    *,
    lat: float,
    lon: float,
    zone: str | None = None,
    grid_id: str | None = None,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create source cubes for the explicit grid tile containing a point."""
    selected_grid = _grid_id(zone=zone, grid_id=grid_id)
    return _tiler_cls()(config, selectors=selectors).run_point(
        lat,
        lon,
        selected_grid,
    )


def create_tiles_for_aoi(
    config: TileConfig,
    *,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create configured source cubes for every routed tile intersecting an AOI."""
    return _tiler_cls()(config, selectors=selectors).run_aoi(
        ul_lat,
        ul_lon,
        lr_lat,
        lr_lon,
    )


__all__ = [
    "create_tiles_for_aoi",
    "create_tiles_for_index",
    "create_tiles_for_point",
]
