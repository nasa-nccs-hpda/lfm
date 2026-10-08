"""Geographic routing across numbered LTM and lunar polar grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .grid_registry import GridFamily, GridRegistry, default_grid_registry


POLAR_LATITUDE_THRESHOLD = 82.0


class GeographicRoutingError(ValueError):
    """A geographic point or AOI does not satisfy the routing contract."""


@dataclass(frozen=True)
class GridPointRoute:
    """Canonical grid assignment for one lunar geographic point."""

    grid_id: str
    family: GridFamily
    lat: float
    lon: float


@dataclass(frozen=True)
class GridQueryPart:
    """One positive-area, non-wrapping AOI part assigned to one grid."""

    grid_id: str
    family: GridFamily
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not -90 <= self.south < self.north <= 90:
            raise ValueError("Grid query part has invalid latitude bounds.")
        if not -180 <= self.west < self.east <= 180:
            raise ValueError("Grid query part has invalid longitude bounds.")

    @property
    def ul_lat(self) -> float:
        return self.north

    @property
    def ul_lon(self) -> float:
        return self.west

    @property
    def lr_lat(self) -> float:
        return self.south

    @property
    def lr_lon(self) -> float:
        return self.east


def _finite_coordinate(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GeographicRoutingError(f"{name} must be a finite number.")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise GeographicRoutingError(
            f"{name} must be a finite number."
        ) from exc
    if not math.isfinite(normalized):
        raise GeographicRoutingError(f"{name} must be a finite number.")
    return normalized


def _latitude(value: Real, *, name: str) -> float:
    latitude = _finite_coordinate(value, name=name)
    if not -90 <= latitude <= 90:
        raise GeographicRoutingError(f"{name} must be inside [-90, 90].")
    return latitude


def normalize_lunar_longitude(value: Real) -> float:
    """Normalize a finite longitude to [-180, 180], preserving +180."""
    longitude = _finite_coordinate(value, name="longitude")
    normalized = (longitude + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and longitude > 0:
        return 180.0
    if normalized == 0:
        return 0.0
    return normalized


def _ltm_grid_id(lat: float, lon: float) -> str:
    if lon == 180.0:
        zone = 45
    else:
        zone = math.floor((lon + 180.0) / 8.0) + 1
        zone = min(45, max(1, zone))
    hemisphere = "N" if lat >= 0 else "S"
    return f"{zone}{hemisphere}"


def route_point(
    *,
    lat: Real,
    lon: Real,
    registry: GridRegistry | None = None,
) -> GridPointRoute:
    """Route a lunar point using the inclusive +/-82-degree threshold."""
    point_lat = _latitude(lat, name="lat")
    point_lon = normalize_lunar_longitude(lon)
    if abs(point_lat) == 90:
        point_lon = 0.0
    if point_lat >= POLAR_LATITUDE_THRESHOLD:
        grid_id = "LPS_N"
    elif point_lat <= -POLAR_LATITUDE_THRESHOLD:
        grid_id = "LPS_S"
    else:
        grid_id = _ltm_grid_id(point_lat, point_lon)
    active_registry = registry or default_grid_registry()
    definition = active_registry[grid_id]
    return GridPointRoute(
        grid_id=grid_id,
        family=definition.family,
        lat=point_lat,
        lon=point_lon,
    )


def _longitude_parts(
    raw_west: Real,
    raw_east: Real,
    *,
    allow_full_longitude: bool,
) -> tuple[tuple[float, float], ...]:
    west_input = _finite_coordinate(raw_west, name="ul_lon")
    east_input = _finite_coordinate(raw_east, name="lr_lon")
    if west_input == -180.0 and east_input == 180.0:
        if not allow_full_longitude:
            raise GeographicRoutingError(
                "Full-longitude [-180, 180] AOIs are limited to one polar cap."
            )
        return ((-180.0, 180.0),)

    west = normalize_lunar_longitude(west_input)
    east = normalize_lunar_longitude(east_input)
    if west == east:
        raise GeographicRoutingError("AOI longitude width must be positive.")
    if west < east:
        span = east - west
        if span == 180.0:
            raise GeographicRoutingError(
                "An AOI longitude span of exactly 180 degrees is ambiguous."
            )
        if span > 180.0:
            raise GeographicRoutingError(
                "A non-polar AOI longitude span may not exceed 180 degrees."
            )
        return ((west, east),)

    span = (180.0 - west) + (east + 180.0)
    if span == 180.0:
        raise GeographicRoutingError(
            "An AOI longitude span of exactly 180 degrees is ambiguous."
        )
    if span > 180.0:
        raise GeographicRoutingError(
            "An antimeridian AOI longitude span may not exceed 180 degrees."
        )
    parts = []
    if west < 180.0:
        parts.append((west, 180.0))
    if east > -180.0:
        parts.append((-180.0, east))
    if not parts:
        raise GeographicRoutingError("AOI longitude width must be positive.")
    return tuple(parts)


def _append_part(
    parts: list[GridQueryPart],
    registry: GridRegistry,
    *,
    grid_id: str,
    south: float,
    west: float,
    north: float,
    east: float,
) -> None:
    if north <= south or east <= west:
        return
    definition = registry[grid_id]
    parts.append(
        GridQueryPart(
            grid_id=grid_id,
            family=definition.family,
            south=south,
            west=west,
            north=north,
            east=east,
        )
    )


def _append_ltm_parts(
    parts: list[GridQueryPart],
    registry: GridRegistry,
    *,
    south: float,
    north: float,
    longitude_parts: tuple[tuple[float, float], ...],
) -> None:
    hemisphere_intervals = (
        ("S", south, min(north, 0.0)),
        ("N", max(south, 0.0), north),
    )
    for hemisphere, part_south, part_north in hemisphere_intervals:
        if part_north <= part_south:
            continue
        definitions = (
            definition
            for definition in registry.by_family(GridFamily.LTM)
            if definition.grid_id.endswith(hemisphere)
        )
        for definition in definitions:
            coverage = definition.geographic_coverage
            for west, east in longitude_parts:
                part_west = max(west, coverage.west)
                part_east = min(east, coverage.east)
                _append_part(
                    parts,
                    registry,
                    grid_id=definition.grid_id,
                    south=part_south,
                    west=part_west,
                    north=part_north,
                    east=part_east,
                )


def route_aoi(
    *,
    ul_lat: Real,
    ul_lon: Real,
    lr_lat: Real,
    lr_lon: Real,
    registry: GridRegistry | None = None,
) -> tuple[GridQueryPart, ...]:
    """Partition and route an AOI into non-wrapping canonical grid parts."""
    north = _latitude(ul_lat, name="ul_lat")
    south = _latitude(lr_lat, name="lr_lat")
    if north <= south:
        raise GeographicRoutingError("AOI requires ul_lat > lr_lat.")
    full_longitude_polar_cap = (
        north == 90.0 and south >= POLAR_LATITUDE_THRESHOLD
    ) or (
        south == -90.0 and north <= -POLAR_LATITUDE_THRESHOLD
    )
    longitude_parts = _longitude_parts(
        ul_lon,
        lr_lon,
        allow_full_longitude=full_longitude_polar_cap,
    )
    active_registry = registry or default_grid_registry()
    parts: list[GridQueryPart] = []

    polar_north_south = max(south, POLAR_LATITUDE_THRESHOLD)
    if north > polar_north_south:
        for west, east in longitude_parts:
            _append_part(
                parts,
                active_registry,
                grid_id="LPS_N",
                south=polar_north_south,
                west=west,
                north=north,
                east=east,
            )

    ltm_south = max(south, -POLAR_LATITUDE_THRESHOLD)
    ltm_north = min(north, POLAR_LATITUDE_THRESHOLD)
    if ltm_north > ltm_south:
        _append_ltm_parts(
            parts,
            active_registry,
            south=ltm_south,
            north=ltm_north,
            longitude_parts=longitude_parts,
        )

    polar_south_north = min(north, -POLAR_LATITUDE_THRESHOLD)
    if polar_south_north > south:
        for west, east in longitude_parts:
            _append_part(
                parts,
                active_registry,
                grid_id="LPS_S",
                south=south,
                west=west,
                north=polar_south_north,
                east=east,
            )

    unique = {part: None for part in parts}
    return tuple(
        sorted(
            unique,
            key=lambda part: (
                part.grid_id,
                part.south,
                part.west,
                part.north,
                part.east,
            ),
        )
    )


__all__ = [
    "GeographicRoutingError",
    "GridPointRoute",
    "GridQueryPart",
    "POLAR_LATITUDE_THRESHOLD",
    "normalize_lunar_longitude",
    "route_aoi",
    "route_point",
]
