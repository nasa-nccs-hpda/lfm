"""Small loading and plotting helpers for chip-creation notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._paths import REPO_ROOT
from .chip_types import ChipResult, TargetGrid


def chip_batch_fingerprints(batch) -> dict[str, tuple[str, str, str | None]]:
    """Hash successful published pairs and split assignments for replay checks.

    Deliberately exclude manifest paths/timings, which vary between isolated
    runs. Failures, empty batches and duplicate IDs cannot count as equality.
    """
    import hashlib

    def digest(path):
        with Path(path).open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        return checksum.hexdigest()

    result = {}
    for item in batch.results:
        sample_id = item.request.sample_id
        if item.status != "success":
            raise ValueError(f"Cannot compare unsuccessful sample {sample_id}: {item.message}")
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        result[sample_id] = (digest(item.chip_path), digest(item.label_path), item.preflight.assigned_split)
    if not result:
        raise ValueError("Cannot compare an empty batch.")
    return result


def latest_crater_label_path(label_dir: str | Path | None = None) -> Path:
    """Find the most recently modified crater-notebook export, read-only.

    Defaults to this checkout's notebooks/outputs/labels. Search only immediate
    *_label_craters.gpkg files: the producer's temporary GeoPackages and unrelated
    indexes are excluded. Equal mtimes use ascending filename order. Discovery
    does not validate labels or imply a match to the selected imagery/AOI.
    """
    directory = (REPO_ROOT / "notebooks/outputs/labels"
                 if label_dir is None else Path(label_dir).expanduser().resolve())
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {directory}. "
                                "Accept/export craters first, or supply the labeling notebook's LABEL_DIR.")
    candidates = [(path.stat().st_mtime_ns, path.name, path)
                  for path in directory.glob("*_label_craters.gpkg") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No *_label_craters.gpkg exports found in {directory}. "
                                "Accept/export craters in the labeling notebook first.")
    return min(candidates, key=lambda item: (-item[0], item[1]))[2]


def read_source_grid(path: str | Path, *, expected_band_count: int | None = None) -> TargetGrid:
    """Read original imagery metadata only, without loading its raster pixels."""
    from .chip_requests import raster_bounds, validate_target_grid_consistency

    with _rasterio().open(Path(path)) as dataset:
        if expected_band_count is not None and dataset.count != expected_band_count:
            raise ValueError(f"Expected {expected_band_count} source bands, found {dataset.count}: {path}")
        if dataset.crs is None:
            raise ValueError(f"Source raster has no CRS: {path}")
        transform = dataset.transform.to_gdal()
        grid = TargetGrid(dataset.crs.to_wkt(), transform,
                          raster_bounds(transform, dataset.width, dataset.height),
                          dataset.width, dataset.height)
    validate_target_grid_consistency(grid)
    return grid


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
    """Plot chip, label and overlay, plus a reference only when one exists.

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
    if instance_count is None:
        classes, inverse = np.unique(label, return_inverse=True)
        instances = inverse.reshape(label.shape)
        color_options = dict(cmap="tab20", vmin=-0.5, vmax=max(0.5, len(classes) - 0.5))
    else:
        instances = np.ma.masked_where(label == 0, (label - 1) % 20)
        color_options = dict(cmap="tab20", vmin=-0.5, vmax=19.5)
    absent_ids = absent_instance_ids(label, instance_count)
    vmin, vmax = _display_limits(generated)

    has_reference = request.reference_path is not None
    figure, axes = plt.subplots(2 if has_reference else 1, 2 if has_reference else 3,
                               figsize=(10, 9) if has_reference else (14, 4), squeeze=False)
    generated_axis = axes.flat[0]
    label_axis, overlay_axis = axes.flat[-2], axes.flat[-1]
    generated_axis.imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
    generated_axis.set_title(f"generated | {generated_name}")

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
    label_axis.imshow(instances, **color_options)
    label_axis.set_title(
        f"label | IDs absent from mask: {list(absent_ids) or 'none'}" if instance_count is not None
        else f"semantic classes: {classes.tolist()}"
    )
    overlay_axis.imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
    overlay_axis.imshow(
        instances,
        **color_options,
        alpha=0.45,
    )
    overlay_axis.set_title("diagnostic overlay")

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
    "latest_crater_label_path",
    "plot_chip_result",
    "read_display_band",
    "read_label",
    "read_source_grid",
]
