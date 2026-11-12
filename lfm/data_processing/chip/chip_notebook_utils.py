"""Small loading and plotting helpers for chip-creation notebooks."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import os
import re
from typing import Any

from .._paths import REPO_ROOT
from .chip_types import ChipResult, TargetGrid


def notebook_index_workers() -> int:
    """Use allocated Slurm CPUs, otherwise the process's available CPUs."""
    from ..tiling.vector_index_builder import resolve_index_worker_count

    if os.environ.get("SLURM_CPUS_PER_TASK", "").strip():
        return resolve_index_worker_count()
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


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


def read_source_grid(
    path: str | Path, *, expected_band_count: int | tuple[int, ...] | None = None
) -> TargetGrid:
    """Read native grid metadata, optionally requiring one of the band counts.

    Pass ``(2, 5)`` for a WAC UV or VIS raster. No raster pixels are loaded.
    """
    from .chip_requests import raster_bounds, validate_target_grid_consistency

    with _rasterio().open(Path(path)) as dataset:
        if expected_band_count is not None:
            allowed = ((expected_band_count,) if isinstance(expected_band_count, int)
                       else expected_band_count)
            if dataset.count not in allowed:
                expected = " or ".join(str(count) for count in allowed)
                raise ValueError(f"Expected {expected} source bands, found {dataset.count}: {path}")
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
    band_index: int | None = None,
) -> tuple[Any, str]:
    """Read the first raster band whose description contains ``keyword``.

    Matching is case-insensitive. If no description matches, the first band is
    returned so rasters without band descriptions remain inspectable.
    An explicit zero-based ``band_index`` overrides description matching.
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
        if band_index is not None:
            if isinstance(band_index, bool) or not isinstance(band_index, int) or not 0 <= band_index < dataset.count:
                raise ValueError("band_index must be a valid zero-based band index.")
            index = band_index
        band = _numpy().ma.masked_invalid(dataset.read(index + 1, masked=True).astype("float64"))
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
    values = np.ma.masked_invalid(np.ma.asarray(image, dtype="float64")).compressed()
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(values, (2, 98))
    if lower == upper:
        padding = max(abs(float(lower)) * 0.01, 1.0)
        return float(lower - padding), float(upper + padding)
    return float(lower), float(upper)


def plot_chip_tiling(
    result: ChipResult,
    *,
    display_band_keyword: str = "vis",
    static_band_keyword: str = "elv",
    max_tiles: int = 4,
    figure_path: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot retained tile cubes before chip reprojection and clipping.

    Show one band per source cube for up to ``max_tiles`` structured tile
    addresses, including static context when present. Missing cubes
    require recreating the chip with ``intermediate_retention="always"``.
    Display limits apply only to plotting, not acquisition or saved cubes.
    """
    if not isinstance(result, ChipResult):
        raise TypeError("result must be a ChipResult.")
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or max_tiles < 1:
        raise ValueError("max_tiles must be a positive integer.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    if not str(display_band_keyword).strip() or not str(static_band_keyword).strip():
        raise ValueError("Display band keywords must contain non-whitespace text.")
    if not result.cube_records:
        raise ValueError("No tiling cube records are available for this result.")

    addresses = list(dict.fromkeys((r.zone, r.zoom_level, r.tile_x, r.tile_y)
                                   for r in result.cube_records))
    selected = set(addresses[:max_tiles])
    records = list({r.path: r for r in result.cube_records
                    if (r.zone, r.zoom_level, r.tile_x, r.tile_y) in selected}.values())
    for record in records:
        if not record.path.is_file():
            raise FileNotFoundError(
                f"Tiling cube is unavailable: {record.path}. Set "
                'intermediate_retention="always" and rerun chip creation before plotting.'
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to plot tiling cubes.") from exc
    import textwrap

    np = _numpy()
    columns = min(4, len(records))
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for axis, record in zip(axes.flat, records):
        is_static = record.source_name.casefold() == "static"
        keyword = static_band_keyword if is_static else display_band_keyword
        index = next((i for i, name in enumerate(record.band_names)
                      if keyword.casefold() in name.casefold()), 0)
        with _rasterio().open(record.path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Tiling cube has no CRS: {record.path}")
            image = dataset.read(index + 1, masked=True).astype("float64")
            nodata = record.nodata_values[index]
            if nodata is not None:
                image = np.ma.masked_where(image == nodata, image)
            image = np.ma.masked_invalid(image)
        vmin, vmax = _display_limits(image)
        cmap = plt.get_cmap("terrain" if is_static else "gray").with_extremes(bad="black")
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_xlim(-.5, image.shape[1] - .5)
        axis.set_ylim(image.shape[0] - .5, -.5)
        title = (f"{record.source_name} | {record.band_names[index]}\n"
                 f"{record.zone} z{record.zoom_level} tile ({record.tile_x}, {record.tile_y})")
        if record.product_id:
            title += f"\n{record.product_id}"
        axis.set_title("\n".join(textwrap.fill(line, 45) for line in title.splitlines()))
        axis.axis("off")
        figure.colorbar(rendered, ax=axis, fraction=.046, pad=.04)
    for axis in list(axes.flat)[len(records):]:
        axis.set_visible(False)
    figure.suptitle(f"{result.request.sample_id}: tiling cubes\n"
                   f"Showing {len(selected)} of {len(addresses)} tile addresses")
    figure.tight_layout()
    if figure_path is not None:
        destination = Path(figure_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def save_chip_band_pngs(
    result: ChipResult,
    *,
    output_dir: str | Path,
    band_patterns: Mapping[str, str],
) -> tuple[Path, ...]:
    """Save grayscale band PNGs selected by case-insensitive metadata regexes.

    Keys in ``band_patterns`` label modalities in filenames. Match against
    Rasterio band descriptions, falling back to per-band ``Name`` tags. Each
    pattern must match at least one band, and a band must match only one
    modality. No figures are created or displayed. Images retain native pixel
    dimensions with a per-band 2–98 percentile stretch; masked/NoData and
    nonfinite pixels are transparent, including entirely uncovered bands.
    """
    if not isinstance(result, ChipResult):
        raise TypeError("result must be a ChipResult.")
    if result.status != "success" or result.chip_path is None:
        raise ValueError("Band PNGs require a successful result with a chip path.")
    if not band_patterns:
        raise ValueError("Configure at least one band regex.")
    patterns = {}
    for modality, pattern in band_patterns.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", modality):
            raise ValueError(f"Invalid modality filename component: {modality!r}")
        if not pattern.strip():
            raise ValueError(f"Band regex for {modality!r} must not be empty.")
        patterns[modality] = re.compile(pattern, flags=re.IGNORECASE)

    from matplotlib.image import imsave

    np = _numpy()
    chip_path = Path(result.chip_path)
    destination = Path(output_dir)
    saved = []
    with _rasterio().open(chip_path) as dataset:
        selections = []
        matched = set()
        for index in dataset.indexes:
            tags = dataset.tags(index)
            name = (dataset.descriptions[index - 1] or tags.get("Name")
                    or tags.get("name") or f"band_{index}")
            modalities = [key for key, regex in patterns.items() if regex.search(name)]
            if len(modalities) > 1:
                raise ValueError(f"Band {name!r} matches multiple modalities: {modalities}")
            if modalities:
                modality = modalities[0]
                selections.append((index, name, modality))
                matched.add(modality)
        missing = patterns.keys() - matched
        if missing:
            raise ValueError(f"No raster band metadata matches regexes for: {sorted(missing)}")

        destination.mkdir(parents=True, exist_ok=True)
        for index, name, modality in selections:
            band = np.ma.masked_invalid(dataset.read(index, masked=True).astype("float64"))
            vmin, vmax = _display_limits(band)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or f"band_{index}"
            path = destination / f"{chip_path.stem}_{modality}_band_{index:02d}_{safe_name}.png"
            imsave(path, band, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
            saved.append(path)
    return tuple(saved)


def plot_chip_result(
    result: ChipResult,
    *,
    display_band_keyword: str = "vis",
    display_band_index: int | None = None,
    figure_path: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot chip, label and overlay, plus a reference only when one exists.

    The returned figure and axes remain available for notebook-specific edits.
    When ``figure_path`` is supplied, the figure is also saved to that path.
    ``display_band_index`` selects an exact zero-based band instead of a keyword.
    Class zero is black in the label panel and transparent over the imagery.
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
        band_index=display_band_index,
    )
    label, instance_count = read_label(result.label_path)
    if label.shape != generated.shape:
        raise ValueError(
            f"Chip display band shape {generated.shape} does not match label "
            f"shape {label.shape}."
        )
    if instance_count is None:
        from matplotlib.colors import ListedColormap

        classes, inverse = np.unique(label, return_inverse=True)
        instances = inverse.reshape(label.shape)
        palette = plt.get_cmap("tab20")(np.arange(len(classes)) % 20)
        palette[classes == 0] = (0, 0, 0, 1)
        label_cmap = ListedColormap(palette)
        color_options = dict(vmin=-0.5, vmax=max(0.5, len(classes) - 0.5))
    else:
        instances = np.ma.masked_where(label == 0, (label - 1) % 20)
        label_cmap = plt.get_cmap("tab20").with_extremes(bad="black")
        color_options = dict(vmin=-0.5, vmax=19.5)
    overlay_cmap = label_cmap.with_extremes(bad=(0, 0, 0, 0))
    overlay_instances = np.ma.masked_where(label == 0, instances)
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
            band_index=display_band_index,
        )
        reference_min, reference_max = _display_limits(reference)
        axes[0, 1].imshow(
            reference,
            cmap="gray",
            vmin=reference_min,
            vmax=reference_max,
        )
        axes[0, 1].set_title(f"reference | {reference_name}")
    label_axis.imshow(instances, cmap=label_cmap, interpolation="nearest", **color_options)
    label_axis.set_title(
        f"label | IDs absent from mask: {list(absent_ids) or 'none'}" if instance_count is not None
        else f"semantic classes: {classes.tolist()}"
    )
    overlay_axis.imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
    overlay_axis.imshow(
        overlay_instances,
        cmap=overlay_cmap,
        **color_options,
        alpha=0.45,
        interpolation="nearest",
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
    "plot_chip_tiling",
    "read_display_band",
    "read_label",
    "read_source_grid",
    "save_chip_band_pngs",
]
