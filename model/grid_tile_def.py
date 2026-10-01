"""Grid-neutral tile-definition factory and polar tile geometry."""

from __future__ import annotations

import math
from typing import Iterable

from osgeo import ogr, osr

from .TmsTileDef import TmsTileDef
from .grid_registry import (
    GeographicCoverage,
    GridDefinition,
    GridFamily,
    GridRegistry,
    default_grid_registry,
)
from .lunar_crs import load_lunar_geographic_wkt
from .tile_matrix import (
    TileMatrixGeometry,
    candidate_tile_range,
    projected_to_tile_index,
    tile_bounds,
    validate_tile_index,
)


GEOGRAPHIC_DENSIFY_DEGREES = 0.05
PROJECTED_EDGE_SEGMENTS = 128
GEOGRAPHIC_ENVELOPE_PADDING_DEGREES = 0.05


def _coordinate_transformation(source, target, *, description: str):
    with osr.ExceptionMgr(useExceptions=False):
        transform = osr.CoordinateTransformation(source, target)
    if transform is None:
        raise RuntimeError(
            f"Could not construct lunar coordinate transformation: {description}."
        )
    return transform


def _transform_point(transform, x: float, y: float, *, description: str):
    result = transform.TransformPoint(x, y)
    if result is None or len(result) < 2:
        raise RuntimeError(f"Lunar coordinate transformation failed: {description}.")
    out_x, out_y = float(result[0]), float(result[1])
    if not math.isfinite(out_x) or not math.isfinite(out_y):
        raise RuntimeError(
            "Lunar coordinate transformation returned non-finite values for "
            f"{description}: ({out_x}, {out_y})."
        )
    return out_x, out_y


def _edge_values(start: float, stop: float, max_step: float) -> list[float]:
    count = max(1, math.ceil(abs(stop - start) / max_step))
    return [start + (stop - start) * index / count for index in range(count)]


def _geographic_perimeter(
    *,
    north: float,
    west: float,
    south: float,
    east: float,
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    points.extend(
        (lon, north)
        for lon in _edge_values(west, east, GEOGRAPHIC_DENSIFY_DEGREES)
    )
    points.extend(
        (east, lat)
        for lat in _edge_values(north, south, GEOGRAPHIC_DENSIFY_DEGREES)
    )
    points.extend(
        (lon, south)
        for lon in _edge_values(east, west, GEOGRAPHIC_DENSIFY_DEGREES)
    )
    points.extend(
        (west, lat)
        for lat in _edge_values(south, north, GEOGRAPHIC_DENSIFY_DEGREES)
    )
    points.append((west, north))
    return tuple(points)


def _full_polar_cap_perimeter(
    *,
    boundary_latitude: float,
) -> tuple[tuple[float, float], ...]:
    points = [
        (lon, boundary_latitude)
        for lon in _edge_values(
            -180.0,
            180.0,
            GEOGRAPHIC_DENSIFY_DEGREES,
        )
    ]
    points.append((-180.0, boundary_latitude))
    return tuple(points)


def _polygon_from_points(points: Iterable[tuple[float, float]]):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in points:
        ring.AddPoint_2D(x, y)
    ring.CloseRings()
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    if polygon.IsEmpty() or not polygon.IsValid():
        repaired = polygon.MakeValid()
        if repaired is None or repaired.IsEmpty():
            raise ValueError("Projected geographic AOI produced invalid geometry.")
        polygon = repaired
    return polygon


def _rectangle_polygon(
    ulx: float,
    uly: float,
    lrx: float,
    lry: float,
):
    return _polygon_from_points(
        ((ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry), (ulx, uly))
    )


def _projected_edge_points(
    bounds: tuple[float, float, float, float],
    *,
    segments: int = PROJECTED_EDGE_SEGMENTS,
) -> tuple[tuple[float, float], ...]:
    ulx, uly, lrx, lry = bounds
    points: list[tuple[float, float]] = []
    for index in range(segments):
        fraction = index / segments
        points.append((ulx + (lrx - ulx) * fraction, uly))
    for index in range(segments):
        fraction = index / segments
        points.append((lrx, uly + (lry - uly) * fraction))
    for index in range(segments):
        fraction = index / segments
        points.append((lrx + (ulx - lrx) * fraction, lry))
    for index in range(segments):
        fraction = index / segments
        points.append((ulx, lry + (uly - lry) * fraction))
    return tuple(points)


def _longitude_envelopes(
    longitudes: Iterable[float],
) -> tuple[tuple[float, float], ...]:
    expanded = []
    for longitude in longitudes:
        expanded.extend(
            (
                longitude - GEOGRAPHIC_ENVELOPE_PADDING_DEGREES,
                longitude + GEOGRAPHIC_ENVELOPE_PADDING_DEGREES,
            )
        )
    values = sorted(
        set((float(longitude) + 180.0) % 360.0 - 180.0 for longitude in expanded)
    )
    if not values:
        raise ValueError("Cannot derive a geographic envelope without longitudes.")
    if len(values) == 1:
        epsilon = 1e-9
        west = max(-180.0, values[0] - epsilon)
        east = min(180.0, values[0] + epsilon)
        return ((west, east),)
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    gaps.append(values[0] + 360.0 - values[-1])
    largest_index = max(range(len(gaps)), key=gaps.__getitem__)
    west = values[(largest_index + 1) % len(values)]
    east = values[largest_index]
    if largest_index == len(values) - 1:
        return ((west, east),)
    result = []
    if west < 180.0:
        result.append((west, 180.0))
    if east > -180.0:
        result.append((-180.0, east))
    return tuple(result)


class PolarTileDef:
    """Projected tile geometry for one `LPS_N` or `LPS_S` zoom matrix."""

    def __init__(self, definition: GridDefinition, zoom_level: int) -> None:
        if definition.family not in (GridFamily.LPS_N, GridFamily.LPS_S):
            raise ValueError(
                f"PolarTileDef requires LPS_N or LPS_S, got {definition.grid_id!r}."
            )
        selected = definition.matrix(zoom_level)
        self._definition = definition
        self._zoom_level = zoom_level
        self._matrix = TileMatrixGeometry(
            origin_x=selected.origin_x,
            origin_y=selected.origin_y,
            cell_size=selected.cell_size,
            tile_width=selected.tile_width,
            tile_height=selected.tile_height,
            matrix_width=selected.matrix_width,
            matrix_height=selected.matrix_height,
        )
        self._srs = osr.SpatialReference()
        if self._srs.ImportFromWkt(definition.crs_wkt) != 0:
            raise ValueError(f"Could not parse CRS for grid {definition.grid_id!r}.")
        self._geo_srs = osr.SpatialReference()
        if self._geo_srs.ImportFromWkt(load_lunar_geographic_wkt()) != 0:
            raise ValueError("Could not parse repository IAU:30100 WKT.")
        projected_geo_srs = self._srs.CloneGeogCS()
        if projected_geo_srs is None or not projected_geo_srs.IsSame(self._geo_srs):
            raise ValueError(
                f"Grid {definition.grid_id!r} does not use repository IAU:30100."
            )
        self._srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self._geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self._geo_to_projected = _coordinate_transformation(
            self._geo_srs,
            self._srs,
            description=f"IAU:30100 to {definition.grid_id}",
        )
        self._projected_to_geo = _coordinate_transformation(
            self._srs,
            self._geo_srs,
            description=f"{definition.grid_id} to IAU:30100",
        )

    @property
    def grid_id(self) -> str:
        return self._definition.grid_id

    @property
    def zone(self) -> str:
        return self.grid_id

    @property
    def zoomLevel(self) -> int:
        return self._zoom_level

    @property
    def srs(self):
        return self._srs

    @property
    def geoSrs(self):
        return self._geo_srs

    @property
    def cellSize(self) -> float:
        return self._matrix.cell_size

    @property
    def pointOfOrigin(self) -> tuple[float, float]:
        return self._matrix.origin_x, self._matrix.origin_y

    @property
    def tileWidth(self) -> int:
        return self._matrix.tile_width

    @property
    def tileHeight(self) -> int:
        return self._matrix.tile_height

    @property
    def matrixWidth(self) -> int:
        return self._matrix.matrix_width

    @property
    def matrixHeight(self) -> int:
        return self._matrix.matrix_height

    def validateTileIndex(self, tile_x: int, tile_y: int) -> None:
        validate_tile_index(
            self._matrix,
            tile_x,
            tile_y,
            grid_id=self.grid_id,
            zoom_level=self.zoomLevel,
        )

    def getTileBbox(self, tile_x: int, tile_y: int) -> list[float]:
        self.validateTileIndex(tile_x, tile_y)
        return list(tile_bounds(self._matrix, tile_x, tile_y))

    def latLonToProjected(self, lat: float, lon: float) -> tuple[float, float]:
        return _transform_point(
            self._geo_to_projected,
            lon,
            lat,
            description=f"longitude/latitude ({lon}, {lat}) to {self.grid_id}",
        )

    def projectedToLatLon(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = _transform_point(
            self._projected_to_geo,
            x,
            y,
            description=f"{self.grid_id} ({x}, {y}) to longitude/latitude",
        )
        return lat, lon

    def llToTileIndex(self, lat: float, lon: float) -> tuple[int, int] | None:
        x, y = self.latLonToProjected(lat, lon)
        return projected_to_tile_index(self._matrix, x, y)

    def getOverlappingTiles(
        self,
        ulLat: float,
        ulLon: float,
        lrLat: float,
        lrLon: float,
        minOverlapMeters: float = 10.0,
    ) -> list[tuple[int, int]]:
        if ulLon == -180.0 and lrLon == 180.0:
            boundary_latitude = (
                lrLat
                if self._definition.family is GridFamily.LPS_N
                else ulLat
            )
            perimeter = _full_polar_cap_perimeter(
                boundary_latitude=boundary_latitude,
            )
        else:
            perimeter = _geographic_perimeter(
                north=ulLat,
                west=ulLon,
                south=lrLat,
                east=lrLon,
            )
        projected = tuple(
            self.latLonToProjected(lat, lon) for lon, lat in perimeter
        )
        query_polygon = _polygon_from_points(projected)
        min_x, max_x, min_y, max_y = query_polygon.GetEnvelope()
        candidates = candidate_tile_range(
            self._matrix,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        if candidates is None:
            return []
        min_col, max_col, min_row, max_row = candidates
        indices: list[tuple[int, int]] = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_polygon = _rectangle_polygon(*self.getTileBbox(col, row))
                if not query_polygon.Intersects(tile_polygon):
                    continue
                intersection = query_polygon.Intersection(tile_polygon)
                if intersection is None or intersection.IsEmpty():
                    continue
                ix_min, ix_max, iy_min, iy_max = intersection.GetEnvelope()
                if ix_max - ix_min < minOverlapMeters:
                    continue
                if iy_max - iy_min < minOverlapMeters:
                    continue
                indices.append((col, row))
        return indices

    def geographic_query_envelopes(
        self,
        tile_x: int,
        tile_y: int,
    ) -> tuple[GeographicCoverage, ...]:
        bounds = tuple(self.getTileBbox(tile_x, tile_y))
        projected_points = _projected_edge_points(bounds)
        geographic_points = tuple(
            self.projectedToLatLon(x, y) for x, y in projected_points
        )
        latitudes = [lat for lat, _ in geographic_points]
        longitudes = [lon for _, lon in geographic_points]
        ulx, uly, lrx, lry = bounds
        pole_x, pole_y = self.latLonToProjected(
            90.0 if self._definition.family is GridFamily.LPS_N else -90.0,
            0.0,
        )
        contains_pole = ulx <= pole_x <= lrx and lry <= pole_y <= uly
        if contains_pole:
            longitude_ranges = ((-180.0, 180.0),)
            if self._definition.family is GridFamily.LPS_N:
                latitudes.append(90.0)
            else:
                latitudes.append(-90.0)
        else:
            longitude_ranges = _longitude_envelopes(longitudes)
        south = max(
            -90.0,
            min(latitudes) - GEOGRAPHIC_ENVELOPE_PADDING_DEGREES,
        )
        north = min(
            90.0,
            max(latitudes) + GEOGRAPHIC_ENVELOPE_PADDING_DEGREES,
        )
        return tuple(
            GeographicCoverage(
                south=south,
                west=west,
                north=north,
                east=east,
            )
            for west, east in longitude_ranges
        )


def tile_definition_for_grid(
    grid_id: str,
    zoom_level: int,
    *,
    registry: GridRegistry | None = None,
):
    """Return the proven LTM definition or the polar grid implementation."""
    active_registry = registry or default_grid_registry()
    definition = active_registry[grid_id]
    definition.matrix(zoom_level)
    if definition.family is GridFamily.LTM:
        return TmsTileDef.initFromParams(grid_id, zoom_level)
    return PolarTileDef(definition, zoom_level)


__all__ = [
    "GEOGRAPHIC_DENSIFY_DEGREES",
    "GEOGRAPHIC_ENVELOPE_PADDING_DEGREES",
    "PROJECTED_EDGE_SEGMENTS",
    "PolarTileDef",
    "tile_definition_for_grid",
]
