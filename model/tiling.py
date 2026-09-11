"""Public configuration-driven lunar tiling API."""

from __future__ import annotations

from collections.abc import Mapping

from .tiling_config import TileConfig
from .tiling_results import TileCubeRecord


def _tiler_cls():
    from .configured_tiler import ConfiguredTiler

    return ConfiguredTiler


def create_tiles_for_index(
    config: TileConfig,
    *,
    tile_x: int,
    tile_y: int,
    zone: str,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create configured source cubes for one explicit LTM tile index."""
    return _tiler_cls()(config, selectors=selectors).run_tile_index(
        tile_x,
        tile_y,
        zone,
    )


def create_tiles_for_point(
    config: TileConfig,
    *,
    lat: float,
    lon: float,
    zone: str,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create configured source cubes for the LTM tile containing a point."""
    return _tiler_cls()(config, selectors=selectors).run_point(lat, lon, zone)


def create_tiles_for_aoi(
    config: TileConfig,
    *,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    selectors: Mapping[str, str] | None = None,
) -> list[TileCubeRecord]:
    """Create configured source cubes for every LTM tile intersecting an AOI."""
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
