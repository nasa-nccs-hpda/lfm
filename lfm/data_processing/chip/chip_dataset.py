"""Native raster-window planning for the full-dataset notebook."""

from dataclasses import dataclass
import math
import warnings

from .chip_requests import (
    geographic_aoi_from_target_grid, target_grid_from_pixel_bounds,
    validate_target_grid_consistency,
)
from .chip_types import ChipRequest, LabelInput, SourceSelector, TargetGrid


@dataclass(frozen=True)
class RasterChipPlan:
    requests: tuple[ChipRequest, ...]
    chip_size: int
    block_size_chips: int
    window_rows: int
    window_columns: int
    dropped_rows: int
    dropped_columns: int
    dropped_pixels: int


def plan_raster_chips(
    source_grid: TargetGrid, *, product_id: str, label_input: LabelInput,
    source_selectors: tuple[SourceSelector, ...], chip_size: int = 256,
    block_size_chips: int = 4,
) -> RasterChipPlan:
    """Cover the full raster with complete, disjoint square pixel windows.

    Block membership is anchored to raster pixel (0, 0), never to feature
    order or crater extents. Geographic envelopes only select acquisition;
    they are never round-tripped back into enlarged output windows.
    """
    for name, value in (("chip_size", chip_size), ("block_size_chips", block_size_chips)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    validate_target_grid_consistency(source_grid)
    _, a, b, _, c, d = source_grid.transform
    if not math.isclose(a * b + c * d, 0, rel_tol=0,
                        abs_tol=1e-10 * math.hypot(a, c) * math.hypot(b, d)):
        raise ValueError("Source pixel lattice is sheared; chip footprints must be rectangular.")
    columns, right = divmod(source_grid.width, chip_size)
    rows, bottom = divmod(source_grid.height, chip_size)
    dropped = source_grid.width * source_grid.height - rows * columns * chip_size**2
    if bottom or right:
        warnings.warn(f"Dropping incomplete windows: {bottom} bottom rows, {right} right columns "
                      f"({dropped} pixels total, corner counted once).", UserWarning, stacklevel=2)
    if not rows or not columns:
        raise ValueError("Source raster contains no complete chip window.")
    requests = []
    for row in range(rows):
        for column in range(columns):
            r, c = row * chip_size, column * chip_size
            grid = target_grid_from_pixel_bounds(source_grid, (c, r, c + chip_size, r + chip_size))
            group = (f"{product_id}_block{block_size_chips}_chip{chip_size}"
                     f"_r{row // block_size_chips}_c{column // block_size_chips}")
            requests.append(ChipRequest(
                sample_id=f"{product_id}_r{r}_c{c}", target_grid=grid,
                geographic_aoi=geographic_aoi_from_target_grid(grid),
                split_group_key=group, source_selectors=tuple(source_selectors), label_input=label_input,
            ))
    return RasterChipPlan(tuple(requests), chip_size, block_size_chips, rows, columns,
                          bottom, right, dropped)
