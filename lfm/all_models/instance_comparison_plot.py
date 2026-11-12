"""Shared instance comparison figures requiring only NumPy and Matplotlib.

Kept outside the training packages so local previews do not import model weights,
Torch, geospatial readers, or Lightning. These ID rasters are for visualization;
instance AP must be calculated separately from the original detection masks.
"""

from __future__ import annotations

import colorsys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lfm.all_models.plotting import blend_overlay, draw_boxes, figure_grid
from lfm.all_models.prediction_artifacts import ChipMetadata


@dataclass(frozen=True)
class InstanceComparisonChip:
    key: str
    display_image: np.ndarray
    ground_truth: np.ndarray
    predictions: Mapping[str, np.ndarray]
    valid_mask: np.ndarray | None = None
    prediction_boxes: Mapping[str, np.ndarray] = field(default_factory=dict)
    ground_truth_boxes: np.ndarray | None = None
    display_note: str = ""


def instance_colormap(instance_mask: np.ndarray) -> np.ndarray:
    """Use the validation callback's deterministic palette for positive IDs."""
    mask = np.asarray(instance_mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for instance_id in np.unique(mask):
        if instance_id <= 0:
            continue
        hue = (int(instance_id) * 0.618033988749895) % 1.0
        saturation = 0.9 if int(instance_id) % 2 == 0 else 0.7
        value = 0.95 if (int(instance_id) // 2) % 2 == 0 else 0.75
        rgb[mask == instance_id] = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb


def instance_partitions_match(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare instance geometry even when caches renumber positive IDs."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or not np.array_equal(first > 0, second > 0):
        return False
    first_ids = np.unique(first[first > 0])
    second_ids = np.unique(second[second > 0])
    if len(first_ids) != len(second_ids):
        return False
    return all(len(np.unique(second[first == value])) == 1 for value in first_ids)


def validate_instance_comparison_sources(
    samples_by_model: Mapping[str, Mapping[str, dict]],
    sample_keys: Sequence[str],
) -> None:
    """Check chip identity and target geometry, allowing independent normalization."""
    if not samples_by_model:
        raise ValueError("Provide at least one model's cached samples.")
    first_model = next(iter(samples_by_model))
    for key in sample_keys:
        reference = samples_by_model[first_model][key]
        reference_gt = np.asarray(reference["gt_mask"])
        for model, samples in samples_by_model.items():
            sample = samples[key]
            gt = np.asarray(sample["gt_mask"])
            if not instance_partitions_match(reference_gt, gt):
                raise ValueError(
                    f"{key}: {model} ground truth differs from the reference cache."
                )
            ChipMetadata(
                key, str(reference.get("image_path", "")), "", reference_gt >= 0
            ).validate_pair(
                ChipMetadata(key, str(sample.get("image_path", "")), "", gt >= 0),
                paths="name",
            )
            if (
                np.asarray(sample["image"]).shape[-2:] != reference_gt.shape
                or np.asarray(sample["pred_mask"]).shape != reference_gt.shape
            ):
                raise ValueError(
                    f"{key}: {model} image and predictions must align with ground truth."
                )


def _boxes_from_mask(mask: np.ndarray) -> np.ndarray:
    boxes = []
    for value in np.unique(mask):
        if value <= 0:
            continue
        ys, xs = np.where(mask == value)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def plot_instance_comparison(
    chips: Sequence[InstanceComparisonChip],
    output_path: str | Path,
    *,
    model_labels: Mapping[str, str],
    title: str = "Instance segmentation on the same chips",
    synthetic: bool = False,
    overlay: bool = False,
    dpi: int = 150,
) -> Path:
    """Plot input, one prediction row per model, then ground truth.

    ``display_image`` is one canonical HW grayscale or HWC RGB image in [0, 1],
    independent of either model's normalization. The two-model case has four rows.
    """
    if not chips or not model_labels:
        raise ValueError("At least one chip and one model are required.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if len({chip.key for chip in chips}) != len(chips):
        raise ValueError("Comparison chip keys must be unique.")
    prepared = []
    for chip in chips:
        image = np.asarray(chip.display_image, dtype=np.float32)
        gt = np.asarray(chip.ground_truth)
        if gt.ndim != 2:
            raise ValueError(f"{chip.key}: ground truth must be a 2D instance raster.")
        if image.shape != gt.shape and image.shape != (*gt.shape, 3):
            raise ValueError(f"{chip.key}: display image and masks must align.")
        valid = (
            np.asarray(chip.valid_mask, dtype=bool)
            if chip.valid_mask is not None
            else gt >= 0
        )
        if valid.shape != gt.shape:
            raise ValueError(f"{chip.key}: validity mask and ground truth must align.")
        if not np.isfinite(image[valid]).all():
            raise ValueError(f"{chip.key}: valid display pixels must be finite.")
        if np.any((image[valid] < 0) | (image[valid] > 1)):
            raise ValueError(f"{chip.key}: display image must be scaled to [0, 1].")
        for model in model_labels:
            if model not in chip.predictions:
                raise ValueError(f"{chip.key}: missing predictions for {model}.")
            if np.asarray(chip.predictions[model]).shape != gt.shape:
                raise ValueError(
                    f"{chip.key}: {model} prediction and ground truth must align."
                )
        rgb = (
            np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image.copy()
        )
        rgb[~valid] = 0.3
        prepared.append((chip, rgb, gt, valid))

    n_rows = 2 + len(model_labels)
    row_labels = ["Input chip", *model_labels.values(), "Ground truth"]
    with figure_grid(
        n_rows, len(chips), figsize=(4 * len(chips) + 0.5, 3.3 * n_rows + 0.7)
    ) as grid:
        fig, axes = grid.figure, grid.axes
        for col, (chip, rgb, gt, valid) in enumerate(prepared):
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(
                chip.key + (f"\n{chip.display_note}" if chip.display_note else ""),
                fontsize=10,
            )
            masks = [chip.predictions[name] for name in model_labels] + [gt]
            boxes = [chip.prediction_boxes.get(name) for name in model_labels]
            boxes.append(chip.ground_truth_boxes)
            for row, (mask, supplied_boxes) in enumerate(zip(masks, boxes), start=1):
                mask = np.asarray(mask).copy()
                mask[~valid] = 0
                colors = instance_colormap(mask)
                if overlay:
                    foreground = mask > 0
                    colors = blend_overlay(rgb, colors, foreground=foreground)
                colors[~valid] = 0.3
                ax = axes[row, col]
                ax.imshow(colors, vmin=0, vmax=1)
                draw_boxes(
                    ax,
                    _boxes_from_mask(mask)
                    if supplied_boxes is None
                    else supplied_boxes,
                )
                count = len(np.unique(mask[mask > 0]))
                ax.set_title(
                    f"{count} {'instance' if count == 1 else 'instances'}", fontsize=10
                )
            for row in range(n_rows):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=11, labelpad=12)
        heading = (
            f"SYNTHETIC PREVIEW — NOT MODEL RESULTS\n{title}" if synthetic else title
        )
        fig.suptitle(heading, fontsize=15, fontweight="bold", y=0.99)
        note = "Instance colors show identities; red boxes show extents."
        if any(not valid.all() for _, _, _, valid in prepared):
            note += " Gray = NoData."
        if synthetic:
            note += " All chips and predictions are generated; no training or metrics."
        fig.text(
            0.5,
            0.012,
            textwrap.fill(note, width=max(45, int(fig.get_figwidth() * 16))),
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.945 if synthetic else 0.965))
        return grid.save(output_path, dpi=dpi)
