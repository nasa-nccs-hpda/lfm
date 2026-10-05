"""Shared visualization helpers for model-agnostic workflows."""

from .tiling_viz import (
    SUBPLOT_TITLE_MAX_CHARACTERS,
    display_array,
    pair_dynamic_and_static,
    plot_cube_pairs,
    plot_cube_records,
    plot_modern_legacy_cube_comparison,
    plot_tiling_records,
    print_record_summary,
    read_raster_band,
    read_record_band,
    robust_limits,
    tile_key,
    wrap_subplot_title,
)

__all__ = [
    "SUBPLOT_TITLE_MAX_CHARACTERS",
    "display_array",
    "pair_dynamic_and_static",
    "plot_cube_pairs",
    "plot_cube_records",
    "plot_modern_legacy_cube_comparison",
    "plot_tiling_records",
    "print_record_summary",
    "read_raster_band",
    "read_record_band",
    "robust_limits",
    "tile_key",
    "wrap_subplot_title",
]
