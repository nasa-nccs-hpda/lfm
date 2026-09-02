"""Visualization and compact inspection helpers for tiled lunar datacubes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def tile_key(record: Any) -> tuple[str, int, int, int]:
    """Return a structured spatial key without parsing a cube filename."""
    return record.zone, record.zoom_level, record.tile_x, record.tile_y


def pair_dynamic_and_static(
    records: Sequence[Any],
    dynamic_source: str,
) -> list[tuple[Any, Any]]:
    """Pair one named dynamic source with static context on matching tiles."""
    records_by_tile: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for record in records:
        sources = records_by_tile.setdefault(tile_key(record), {})
        if record.source_name in sources:
            raise ValueError(
                f"Duplicate source {record.source_name!r} for tile "
                f"{tile_key(record)}."
            )
        sources[record.source_name] = record
    return [
        (sources[dynamic_source], sources["static"])
        for _, sources in sorted(records_by_tile.items())
        if dynamic_source in sources and "static" in sources
    ]


def print_record_summary(records: Sequence[Any]) -> None:
    """Print cube identity, a band-name preview, and compact NoData counts."""
    for record in records:
        nodata_counts: list[int] = []
        with rasterio.open(record.path) as source:
            for band_number in range(1, source.count + 1):
                band = source.read(band_number, masked=True)
                nodata_counts.append(int(np.ma.getmaskarray(band).sum()))
        if len(record.band_names) <= 7:
            band_preview = list(record.band_names)
        else:
            band_preview = [
                *record.band_names[:3],
                "...",
                *record.band_names[-3:],
            ]
        print(
            f"{record.source_name:>6} | LTM{record.zone} | z{record.zoom_level} | "
            f"tile=({record.tile_x}, {record.tile_y}) | "
            f"bands={len(record.band_names):>2} | {record.path.name}"
        )
        print(f"         band names: {band_preview}")
        print(
            f"         NoData pixels per band: min={min(nodata_counts):,}, "
            f"max={max(nodata_counts):,}, total={sum(nodata_counts):,}"
        )


def robust_limits(image: np.ma.MaskedArray) -> tuple[float, float]:
    """Return finite 2nd/98th percentile limits for a masked raster band."""
    values = np.asarray(image.compressed(), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    lower, upper = np.percentile(values, (2.0, 98.0))
    if np.isclose(lower, upper):
        padding = max(abs(float(lower)) * 0.01, 1.0)
        return float(lower - padding), float(upper + padding)
    return float(lower), float(upper)


def read_record_band(
    record: Any,
    *,
    band_number: int | None = None,
    band_name: str | None = None,
) -> tuple[np.ma.MaskedArray, str, int]:
    """Read one masked band selected by 1-based number or exact record name."""
    if (band_number is None) == (band_name is None):
        raise ValueError("Provide exactly one of band_number or band_name.")
    if band_name is not None:
        try:
            band_number = record.band_names.index(band_name) + 1
        except ValueError as exc:
            raise KeyError(
                f"Band {band_name!r} is not present in {record.path}"
            ) from exc
    if band_number is None or band_number < 1:
        raise ValueError("band_number must be a positive 1-based index.")
    with rasterio.open(record.path) as source:
        if band_number > source.count:
            raise IndexError(
                f"Band {band_number} exceeds the {source.count} bands in "
                f"{record.path}."
            )
        image = source.read(band_number, masked=True)
        name = source.tags(band_number).get("Name", f"band_{band_number}")
    return image, name, band_number


def read_raster_band(
    path: str | Path,
    *,
    band_number: int | None = None,
    band_name: str | None = None,
) -> tuple[np.ma.MaskedArray, str, int]:
    """Read a masked raster band by 1-based number or ``Name`` metadata."""
    if (band_number is None) == (band_name is None):
        raise ValueError("Provide exactly one of band_number or band_name.")
    path = Path(path)
    with rasterio.open(path) as source:
        if band_name is not None:
            matching_numbers = [
                number
                for number in range(1, source.count + 1)
                if source.tags(number).get("Name") == band_name
                or source.descriptions[number - 1] == band_name
            ]
            if not matching_numbers:
                raise KeyError(f"Band {band_name!r} is not present in {path}")
            if len(matching_numbers) > 1:
                raise ValueError(f"Band {band_name!r} is duplicated in {path}")
            band_number = matching_numbers[0]
        if band_number is None or band_number < 1:
            raise ValueError("band_number must be a positive 1-based index.")
        if band_number > source.count:
            raise IndexError(
                f"Band {band_number} exceeds the {source.count} bands in {path}."
            )
        image = source.read(band_number, masked=True)
        name = (
            source.tags(band_number).get("Name")
            or source.descriptions[band_number - 1]
            or f"band_{band_number}"
        )
    return image, name, band_number


def plot_cube_pairs(
    pairs: Sequence[tuple[Any, Any]],
    *,
    dynamic_label: str,
    dynamic_band_number: int,
    static_band_name: str,
    output_path: str | Path,
    max_tiles: int = 2,
):
    """Plot paired dynamic/static cube bands with masks and colorbars."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be positive.")
    selected_pairs = list(pairs)[:max_tiles]
    if not selected_pairs:
        raise ValueError(
            f"No {dynamic_label}/static tile pairs are available to plot."
        )

    figure = plt.figure(
        figsize=(6 * len(selected_pairs), 10),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        2 * len(selected_pairs),
        width_ratios=[
            value for _ in selected_pairs for value in (1.0, 0.05)
        ],
    )

    for column, (dynamic_record, static_record) in enumerate(selected_pairs):
        dynamic, dynamic_name, dynamic_number = read_record_band(
            dynamic_record,
            band_number=dynamic_band_number,
        )
        static, static_name, static_number = read_record_band(
            static_record,
            band_name=static_band_name,
        )
        tile_title = (
            f"LTM{dynamic_record.zone} z{dynamic_record.zoom_level} "
            f"tile ({dynamic_record.tile_x}, {dynamic_record.tile_y})"
        )

        for row, (image, name, number, cmap, label) in enumerate(
            (
                (dynamic, dynamic_name, dynamic_number, "gray", dynamic_label),
                (static, static_name, static_number, "terrain", "STATIC"),
            )
        ):
            axis = figure.add_subplot(grid[row, 2 * column])
            colorbar_axis = figure.add_subplot(grid[row, 2 * column + 1])
            vmin, vmax = robust_limits(image)
            rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"{tile_title}\n{label} band {number}: {name}")
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(image.shape[0] - 0.5, -0.5)
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            figure.colorbar(rendered, cax=colorbar_axis)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"{dynamic_label} + STATIC LTM cubes", y=1.01)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved visualization: {output_path}")
    return figure


def plot_modern_legacy_cube_comparison(
    comparisons: Sequence[tuple[Any, Any, Path, Path]],
    *,
    product_id: str,
    dynamic_band_number: int,
    static_band_name: str,
    output_path: str | Path,
    max_tiles: int = 2,
):
    """Plot current and legacy WAC/static cubes as four ordered rows."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be positive.")
    selected = list(comparisons)[:max_tiles]
    if not selected:
        raise ValueError("No matched current/legacy cube sets are available to plot.")

    figure = plt.figure(
        figsize=(6 * len(selected), 20),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        4,
        2 * len(selected),
        width_ratios=[value for _ in selected for value in (1.0, 0.05)],
    )

    for column, comparison in enumerate(selected):
        current_wac, current_static, legacy_wac, legacy_static = comparison
        images = (
            (*read_record_band(current_wac, band_number=dynamic_band_number),
             "gray", "Current WAC"),
            (*read_record_band(current_static, band_name=static_band_name),
             "terrain", "Current STATIC"),
            (*read_raster_band(legacy_wac, band_number=dynamic_band_number),
             "gray", "Legacy WAC"),
            (*read_raster_band(legacy_static, band_name=static_band_name),
             "terrain", "Legacy STATIC"),
        )
        tile_title = (
            f"LTM{current_wac.zone} z{current_wac.zoom_level} "
            f"tile ({current_wac.tile_x}, {current_wac.tile_y})"
        )

        for row, (image, name, number, cmap, label) in enumerate(images):
            axis = figure.add_subplot(grid[row, 2 * column])
            colorbar_axis = figure.add_subplot(grid[row, 2 * column + 1])
            vmin, vmax = robust_limits(image)
            rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"{tile_title}\n{label} band {number}: {name}")
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(image.shape[0] - 0.5, -0.5)
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            figure.colorbar(rendered, cax=colorbar_axis)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"Modern vs legacy WAC/static cubes: {product_id}", y=1.005)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison visualization: {output_path}")
    return figure


__all__ = [
    "pair_dynamic_and_static",
    "plot_cube_pairs",
    "plot_modern_legacy_cube_comparison",
    "print_record_summary",
    "read_raster_band",
    "read_record_band",
    "robust_limits",
    "tile_key",
]
