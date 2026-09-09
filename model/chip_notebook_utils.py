"""Small loading and plotting helpers for chip-creation notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chip_types import ChipResult


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for chip notebook helpers.") from exc
    return np


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Rasterio is required for chip notebook helpers.") from exc
    return rasterio


def read_display_band(
    path: str | Path,
    *,
    keyword: str = "vis",
) -> tuple[Any, str]:
    """Read the first raster band whose description contains ``keyword``.

    Matching is case-insensitive. If no description matches, the first band is
    returned so rasters without band descriptions remain inspectable.
    """
    raster_path = Path(path)
    normalized_keyword = str(keyword).strip().casefold()
    if not normalized_keyword:
        raise ValueError("keyword must contain non-whitespace text.")

    rasterio = _rasterio()
    with rasterio.open(raster_path) as dataset:
        names = tuple(
            description or f"band {index + 1}"
            for index, description in enumerate(dataset.descriptions)
        )
        index = next(
            (
                index
                for index, name in enumerate(names)
                if normalized_keyword in name.casefold()
            ),
            0,
        )
        band = dataset.read(index + 1, masked=True)
    return band, names[index]


def read_label(path: str | Path) -> tuple[Any, int | None]:
    """Load a semantic ``.npy`` label or an instance ``.npz`` label archive."""
    np = _numpy()
    label_path = Path(path)
    if label_path.suffix.casefold() == ".npz":
        with np.load(label_path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"])
            instance_count = (
                int(archive["num_craters"])
                if "num_craters" in archive
                else None
            )
        return mask, instance_count
    return np.asarray(np.load(label_path, allow_pickle=False)), None


def absent_instance_ids(
    mask: Any,
    instance_count: int | None,
) -> tuple[int, ...]:
    """Return declared instance IDs absent from the rasterized mask.

    An absent ID is diagnostic information, not necessarily an invalid label:
    a declared instance can be fully occluded by another instance while its
    bounding box remains valid.
    """
    if instance_count is None:
        return ()
    if isinstance(instance_count, bool) or not isinstance(instance_count, int):
        raise TypeError("instance_count must be an integer or None.")
    if instance_count < 0:
        raise ValueError("instance_count must be nonnegative.")

    np = _numpy()
    values = np.asarray(mask)
    present = {int(value) for value in np.unique(values[values > 0])}
    return tuple(sorted(set(range(1, instance_count + 1)) - present))


def _display_limits(image: Any) -> tuple[float, float]:
    np = _numpy()
    values = np.ma.asarray(image).compressed()
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(values, (2, 98))
    if lower == upper:
        padding = max(abs(float(lower)) * 0.01, 1.0)
        return float(lower - padding), float(upper + padding)
    return float(lower), float(upper)


def plot_chip_result(
    result: ChipResult,
    *,
    display_band_keyword: str = "vis",
    figure_path: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> tuple[Any, Any]:
    """Build the standard six-panel notebook inspection for one chip result.

    The returned figure and axes remain available for notebook-specific edits.
    When ``figure_path`` is supplied, the figure is also saved to that path.
    """
    if not isinstance(result, ChipResult):
        raise TypeError("result must be a ChipResult.")
    if result.status != "success":
        raise RuntimeError(
            f"Cannot plot {result.request.sample_id!r} with status "
            f"{result.status!r}."
        )
    if result.chip_path is None or result.label_path is None:
        raise ValueError("A successful result must contain chip and label paths.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to plot chip notebook results."
        ) from exc

    np = _numpy()
    rasterio = _rasterio()
    request = result.request
    generated, generated_name = read_display_band(
        result.chip_path,
        keyword=display_band_keyword,
    )
    label, instance_count = read_label(result.label_path)
    if label.shape != generated.shape:
        raise ValueError(
            f"Chip display band shape {generated.shape} does not match label "
            f"shape {label.shape}."
        )
    with rasterio.open(result.chip_path) as dataset:
        output_band_count = dataset.count

    instances = np.ma.masked_where(label == 0, (label - 1) % 20)
    absent_ids = absent_instance_ids(label, instance_count)
    vmin, vmax = _display_limits(generated)

    figure, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    axes[0, 0].imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"generated | {generated_name}")

    if request.reference_path is not None:
        reference, reference_name = read_display_band(
            request.reference_path,
            keyword=display_band_keyword,
        )
        reference_min, reference_max = _display_limits(reference)
        axes[0, 1].imshow(
            reference,
            cmap="gray",
            vmin=reference_min,
            vmax=reference_max,
        )
        axes[0, 1].set_title(f"reference | {reference_name}")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "explicit AOI\n(no reference TIFF)",
            ha="center",
            va="center",
        )

    valid_mask = ~np.ma.getmaskarray(generated)
    axes[0, 2].imshow(valid_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"valid data: {valid_mask.mean():.1%}")

    axes[1, 0].imshow(instances, cmap="tab20", vmin=-0.5, vmax=19.5)
    axes[1, 0].set_title(
        f"label | IDs absent from mask: {list(absent_ids) or 'none'}"
    )
    axes[1, 1].imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(
        instances,
        cmap="tab20",
        vmin=-0.5,
        vmax=19.5,
        alpha=0.45,
    )
    axes[1, 1].set_title("diagnostic overlay")
    axes[1, 2].text(
        0.0,
        1.0,
        f"shape: {request.target_grid.height} x {request.target_grid.width}\n"
        f"bounds: {tuple(round(value, 3) for value in request.target_grid.bounds)}\n"
        f"output bands: {output_band_count}\n"
        f"split: {result.preflight.assigned_split}",
        ha="left",
        va="top",
        family="monospace",
    )
    axes[1, 2].set_title("target contract")

    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(request.sample_id)
    figure.tight_layout()

    if figure_path is not None:
        destination = Path(figure_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


__all__ = [
    "absent_instance_ids",
    "plot_chip_result",
    "read_display_band",
    "read_label",
]
