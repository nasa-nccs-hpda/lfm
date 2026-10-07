"""Native raster-window planning for the full-dataset notebook."""

from dataclasses import dataclass
from contextlib import contextmanager
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
    rejected_mask_windows: int = 0


@contextmanager
def _window_validity(source_grid, source_raster, valid_mask):
    """Read source masks window-by-window, avoiding a full-scene NAC allocation."""
    import numpy as np

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != (source_grid.height, source_grid.width):
            raise ValueError("valid_mask must be Boolean and match the source grid shape.")

    def manual_valid(r, c, size):
        return valid_mask is None or bool(valid_mask[r:r+size, c:c+size].all())

    if source_raster is None:
        yield manual_valid
        return
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(source_raster) as dataset:
        if (dataset.width != source_grid.width or dataset.height != source_grid.height
                or dataset.transform.to_gdal() != source_grid.transform
                or dataset.crs != rasterio.crs.CRS.from_user_input(source_grid.crs_wkt)):
            raise ValueError("Source raster must match the planning grid exactly.")

        def source_valid(r, c, size):
            if not manual_valid(r, c, size):
                return False
            for band in dataset.indexes:
                pixels = dataset.read(band, window=Window(c, r, size, size), masked=True)
                if np.ma.getmaskarray(pixels).any() or not np.isfinite(pixels.data).all():
                    return False
            return True

        yield source_valid


def plan_raster_chips(
    source_grid: TargetGrid, *, product_id: str, label_input: LabelInput,
    source_selectors: tuple[SourceSelector, ...], chip_size: int = 256,
    block_size_chips: int = 4,
    source_raster=None, valid_mask=None,
) -> RasterChipPlan:
    """Cover the full raster with complete, disjoint square pixel windows.

    Block membership is anchored to raster pixel (0, 0), never to feature
    order or crater extents. Geographic envelopes only select acquisition;
    they are never round-tripped back into enlarged output windows.
    When supplied, source_raster requires all source bands to be valid throughout
    each window. An optional native-grid Boolean valid_mask further restricts
    coverage (True means valid). Neither filter moves windows or block boundaries.
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
    rejected = 0
    with _window_validity(source_grid, source_raster, valid_mask) as window_valid:
        for row in range(rows):
            for column in range(columns):
                r, c = row * chip_size, column * chip_size
                if not window_valid(r, c, chip_size):
                    rejected += 1
                    continue
                grid = target_grid_from_pixel_bounds(source_grid, (c, r, c + chip_size, r + chip_size))
                group = (f"{product_id}_block{block_size_chips}_chip{chip_size}"
                         f"_r{row // block_size_chips}_c{column // block_size_chips}")
                requests.append(ChipRequest(
                    sample_id=f"{product_id}_r{r}_c{c}", target_grid=grid,
                    geographic_aoi=geographic_aoi_from_target_grid(grid),
                    split_group_key=group, source_selectors=tuple(source_selectors), label_input=label_input,
                ))
    if rejected:
        warnings.warn(f"Rejected {rejected} complete windows outside the valid source mask.",
                      UserWarning, stacklevel=2)
    if not requests:
        raise ValueError("No complete chip windows remain inside the valid source mask.")
    return RasterChipPlan(tuple(requests), chip_size, block_size_chips, rows, columns,
                          bottom, right, dropped, rejected)
