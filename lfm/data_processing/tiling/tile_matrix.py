"""Grid-neutral projected tile-matrix arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TileMatrixGeometry:
    """Projected layout values needed to address one tile matrix."""

    origin_x: float
    origin_y: float
    cell_size: float
    tile_width: int
    tile_height: int
    matrix_width: int
    matrix_height: int

    @property
    def tile_span_x(self) -> float:
        return self.cell_size * self.tile_width

    @property
    def tile_span_y(self) -> float:
        return self.cell_size * self.tile_height


def validate_tile_index(
    matrix: TileMatrixGeometry,
    tile_x: int,
    tile_y: int,
    *,
    grid_id: str,
    zoom_level: int,
) -> None:
    """Reject an address outside the selected matrix."""
    if isinstance(tile_x, bool) or not isinstance(tile_x, int):
        raise TypeError("tile_x must be an integer.")
    if isinstance(tile_y, bool) or not isinstance(tile_y, int):
        raise TypeError("tile_y must be an integer.")
    if not 0 <= tile_x < matrix.matrix_width:
        raise IndexError(
            f"Grid {grid_id!r} zoom {zoom_level} tile_x {tile_x} is outside "
            f"[0, {matrix.matrix_width - 1}]."
        )
    if not 0 <= tile_y < matrix.matrix_height:
        raise IndexError(
            f"Grid {grid_id!r} zoom {zoom_level} tile_y {tile_y} is outside "
            f"[0, {matrix.matrix_height - 1}]."
        )


def tile_bounds(
    matrix: TileMatrixGeometry,
    tile_x: int,
    tile_y: int,
) -> tuple[float, float, float, float]:
    """Return ``(ulx, uly, lrx, lry)`` projected tile bounds."""
    xmin = matrix.origin_x + tile_x * matrix.tile_span_x
    xmax = matrix.origin_x + (tile_x + 1) * matrix.tile_span_x
    ymax = matrix.origin_y - tile_y * matrix.tile_span_y
    ymin = matrix.origin_y - (tile_y + 1) * matrix.tile_span_y
    return xmin, ymax, xmax, ymin


def projected_to_tile_index(
    matrix: TileMatrixGeometry,
    x: float,
    y: float,
    *,
    epsilon: float = 1e-6,
) -> tuple[int, int] | None:
    """Resolve one projected point to a zero-based matrix address."""
    tile_x = math.floor((x - matrix.origin_x) / matrix.tile_span_x + epsilon)
    tile_y = math.floor((matrix.origin_y - y) / matrix.tile_span_y + epsilon)
    if not 0 <= tile_x < matrix.matrix_width:
        return None
    if not 0 <= tile_y < matrix.matrix_height:
        return None
    return tile_x, tile_y


def candidate_tile_range(
    matrix: TileMatrixGeometry,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    epsilon: float = 1e-6,
) -> tuple[int, int, int, int] | None:
    """Return a clipped inclusive candidate range for projected bounds."""
    min_col = max(
        0,
        math.floor((min_x - matrix.origin_x) / matrix.tile_span_x + epsilon),
    )
    max_col = min(
        matrix.matrix_width - 1,
        math.floor((max_x - matrix.origin_x) / matrix.tile_span_x + epsilon),
    )
    min_row = max(
        0,
        math.floor((matrix.origin_y - max_y) / matrix.tile_span_y + epsilon),
    )
    max_row = min(
        matrix.matrix_height - 1,
        math.floor((matrix.origin_y - min_y) / matrix.tile_span_y + epsilon),
    )
    if min_col > max_col or min_row > max_row:
        return None
    return min_col, max_col, min_row, max_row


__all__ = [
    "TileMatrixGeometry",
    "candidate_tile_range",
    "projected_to_tile_index",
    "tile_bounds",
    "validate_tile_index",
]
