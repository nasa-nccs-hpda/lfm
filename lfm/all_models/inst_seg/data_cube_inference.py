"""Datacube inference helpers for Graha instance segmentation.

The loading and preprocessing boundary is shared with semantic datacube
inference. This module owns instance-specific tiled prediction, duplicate
suppression, global mask assembly, and visualization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from tiler import Tiler
from tqdm import tqdm

from lfm.all_models.sem_seg.data_cube_inference import (
    WAC_NODATA_THRESHOLD,
    calculate_tile_position,
    get_datacube_data,
    load_and_configure_input_data,
    preprocess_datacubes,
)

__all__ = [
    "WAC_NODATA_THRESHOLD",
    "get_datacube_data",
    "load_and_configure_input_data",
    "preprocess_datacubes",
    "sliding_window_instance_inference",
    "plot_instance_inference_results",
]


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return IoU between one xyxy box and an array of xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure NumPy class-aware NMS for detections from overlapping tiles."""
    kept: list[int] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        order = candidates[np.argsort(scores[candidates])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[
                _box_iou(boxes[current], boxes[remaining]) <= iou_threshold
            ]
    return np.asarray(sorted(kept, key=lambda idx: scores[idx], reverse=True))


def _normalize_tile_prediction(
    prediction: dict,
    *,
    score_threshold: float,
    mask_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract filtered boxes, scores, labels, and masks from one tile."""
    boxes = _as_numpy(prediction.get("boxes", np.zeros((0, 4))), dtype=np.float32)
    scores = _as_numpy(
        prediction.get("scores", np.ones((len(boxes),))), dtype=np.float32
    ).reshape(-1)
    labels = _as_numpy(
        prediction.get("labels", np.ones((len(boxes),))), dtype=np.int64
    ).reshape(-1)
    masks = _as_numpy(
        prediction.get("masks", np.zeros((len(boxes), *mask_shape))),
        dtype=np.float32,
    )
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected instance masks with shape (N,H,W), got {masks.shape}."
        )
    if not (len(boxes) == len(scores) == len(labels) == len(masks)):
        raise ValueError(
            "Instance prediction fields disagree in length: "
            f"boxes={len(boxes)}, scores={len(scores)}, "
            f"labels={len(labels)}, masks={len(masks)}."
        )
    keep = (scores >= score_threshold) & (labels > 0)
    return boxes[keep], scores[keep], labels[keep], masks[keep]


def sliding_window_instance_inference(
    images_scaled,
    model,
    *,
    target_size: int | tuple[int, int] = 256,
    device: str | torch.device = "cuda",
    n_channels: int = 70,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    overlap: float = 0.25,
    nodata_masks=None,
    max_detections: int | None = 500,
    verbose: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Run tiled instance inference and merge duplicate global detections.

    Boxes from each tile are translated into full-image coordinates. Duplicate
    detections from overlapping tiles are removed with class-aware box NMS.
    Static NoData is mean-imputed during preprocessing; only WAC-invalid pixels
    are removed from final instance masks.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    images_scaled = np.asarray(images_scaled)
    if images_scaled.ndim == 3:
        images_scaled = images_scaled[None]
    if images_scaled.ndim != 4:
        raise ValueError(
            "images_scaled must have shape (N,H,W,C) or (H,W,C), got "
            f"{images_scaled.shape}."
        )

    was_training = model.training
    model.eval()
    all_predictions: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for image_index, image in enumerate(images_scaled):
            image_h, image_w, image_channels = image.shape
            if image_channels != n_channels:
                raise ValueError(
                    f"Expected {n_channels} channels, got {image_channels}."
                )
            tiler = Tiler(
                data_shape=image.shape,
                tile_shape=(target_size[0], target_size[1], n_channels),
                channel_dimension=-1,
                overlap=overlap,
                mode="reflect",
            )
            detections: list[dict] = []
            tile_iterator = tiler(image, batch_size=1, progress_bar=False)
            for tile_id, tile_batch in tqdm(
                tile_iterator,
                total=len(tiler),
                desc=f"Instance inference image {image_index + 1}",
                disable=not verbose,
            ):
                tile_tensor = (
                    torch.from_numpy(tile_batch[0])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                tile_predictions = model(tile_tensor)
                if len(tile_predictions) != 1:
                    raise ValueError(
                        "Tile inference expects one prediction dictionary, got "
                        f"{len(tile_predictions)}."
                    )
                boxes, scores, labels, masks = _normalize_tile_prediction(
                    tile_predictions[0],
                    score_threshold=score_threshold,
                    mask_shape=target_size,
                )
                y_start, y_end, x_start, x_end = calculate_tile_position(
                    tile_id,
                    image.shape,
                    (target_size[0], target_size[1], n_channels),
                    overlap,
                )
                visible_h, visible_w = y_end - y_start, x_end - x_start
                for box, score, label, mask in zip(
                    boxes, scores, labels, masks
                ):
                    global_box = box.copy()
                    global_box[[0, 2]] += x_start
                    global_box[[1, 3]] += y_start
                    global_box[[0, 2]] = np.clip(
                        global_box[[0, 2]], 0, image_w
                    )
                    global_box[[1, 3]] = np.clip(
                        global_box[[1, 3]], 0, image_h
                    )
                    detections.append(
                        {
                            "box": global_box,
                            "score": float(score),
                            "label": int(label),
                            "mask": mask[:visible_h, :visible_w],
                            "origin": (y_start, x_start),
                        }
                    )

            if detections:
                boxes = np.stack([item["box"] for item in detections])
                scores = np.asarray(
                    [item["score"] for item in detections], dtype=np.float32
                )
                labels = np.asarray(
                    [item["label"] for item in detections], dtype=np.int64
                )
                keep = _class_aware_nms(
                    boxes, scores, labels, nms_iou_threshold
                )
                if max_detections is not None:
                    keep = keep[:max_detections]
                selected = [detections[int(index)] for index in keep]
            else:
                selected = []

            wac_invalid = np.zeros((image_h, image_w), dtype=bool)
            if nodata_masks is not None:
                wac_channels = min(7, nodata_masks[image_index].shape[0])
                wac_invalid = np.any(
                    nodata_masks[image_index][:wac_channels], axis=0
                )

            global_masks = []
            output_boxes = []
            output_scores = []
            output_labels = []
            for item in selected:
                y_start, x_start = item["origin"]
                local_mask = item["mask"] > mask_threshold
                y_stop = min(y_start + local_mask.shape[0], image_h)
                x_stop = min(x_start + local_mask.shape[1], image_w)
                full_mask = np.zeros((image_h, image_w), dtype=bool)
                full_mask[y_start:y_stop, x_start:x_stop] = local_mask[
                    : y_stop - y_start, : x_stop - x_start
                ]
                full_mask[wac_invalid] = False
                if not full_mask.any():
                    continue
                global_masks.append(full_mask)
                output_boxes.append(item["box"])
                output_scores.append(item["score"])
                output_labels.append(item["label"])

            masks_array = (
                np.stack(global_masks)
                if global_masks
                else np.zeros((0, image_h, image_w), dtype=bool)
            )
            boxes_array = (
                np.stack(output_boxes).astype(np.float32)
                if output_boxes
                else np.zeros((0, 4), dtype=np.float32)
            )
            scores_array = np.asarray(output_scores, dtype=np.float32)
            labels_array = np.asarray(output_labels, dtype=np.int64)
            instance_map = np.zeros((image_h, image_w), dtype=np.int32)
            for instance_id, mask in enumerate(masks_array, start=1):
                instance_map[mask & (instance_map == 0)] = instance_id
            result = {
                "boxes": boxes_array,
                "scores": scores_array,
                "labels": labels_array,
                "masks": masks_array,
                "instance_map": instance_map,
            }
            all_predictions.append(result)
            if verbose:
                print(
                    f"Image {image_index + 1}: retained "
                    f"{len(scores_array)} instance(s) after tile NMS."
                )
    model.train(was_training)
    return all_predictions


def _colorize_instances(instance_map: np.ndarray) -> np.ndarray:
    colored = np.zeros((*instance_map.shape, 3), dtype=np.float32)
    cmap = plt.get_cmap("tab20")
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        colored[instance_map == instance_id] = cmap((instance_id - 1) % 20)[:3]
    return colored


def plot_instance_inference_results(
    images_raw,
    predictions: list[dict[str, np.ndarray]],
    file_pairs,
    output_dir,
    n_channels: int,
    *,
    nodata_masks=None,
    static_band_number: int = 59,
    max_samples: int = 4,
    verbose: bool = True,
):
    """Plot raw VIS, a raw static band, and merged instance predictions."""
    sample_count = min(max_samples, len(predictions), len(file_pairs))
    if sample_count == 0:
        raise ValueError("No instance predictions are available to plot.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        figsize=(6 * sample_count, 14),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        3,
        2 * sample_count,
        width_ratios=[value for _ in range(sample_count) for value in (1.0, 0.05)],
    )
    axes = np.empty((3, sample_count), dtype=object)
    colorbar_axes = np.empty((3, sample_count), dtype=object)
    for row in range(3):
        for column in range(sample_count):
            axes[row, column] = fig.add_subplot(grid[row, 2 * column])
            colorbar_axes[row, column] = fig.add_subplot(
                grid[row, 2 * column + 1]
            )
    for index in range(sample_count):
        wac_file, static_file = file_pairs[index]
        prediction = predictions[index]
        with rasterio.open(wac_file) as src:
            vis = src.read(3, masked=True)
            vis_name = src.tags(3).get("Name", "VIS band 0")
        vis_values = vis.compressed()
        vis_limits = (
            (float(vis_values.min()), float(vis_values.max()))
            if vis_values.size
            else (0.0, 1.0)
        )
        with rasterio.open(static_file) as src:
            static = src.read(static_band_number, masked=True)
            static_name = src.tags(static_band_number).get(
                "Name", f"band_{static_band_number}"
            )
        static_values = static.compressed()
        static_limits = (
            (float(static_values.min()), float(static_values.max()))
            if static_values.size
            else (0.0, 1.0)
        )
        vis_plot = axes[0, index].imshow(
            vis, cmap="gray", vmin=vis_limits[0], vmax=vis_limits[1]
        )
        axes[0, index].set_title(f"VIS band 0: {vis_name}")
        fig.colorbar(vis_plot, cax=colorbar_axes[0, index])
        static_plot = axes[1, index].imshow(
            static,
            cmap="gray",
            vmin=static_limits[0],
            vmax=static_limits[1],
        )
        axes[1, index].set_title(
            f"Static band {static_band_number}: {static_name}"
        )
        fig.colorbar(static_plot, cax=colorbar_axes[1, index])
        instance_display = _colorize_instances(prediction["instance_map"])
        if nodata_masks is not None:
            wac_channels = min(7, nodata_masks[index].shape[0])
            invalid = np.any(nodata_masks[index][:wac_channels], axis=0)
            instance_display[invalid] = (0.65, 0.65, 0.65)
        axes[2, index].imshow(instance_display)
        colorbar_axes[2, index].set_axis_off()
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            x1, y1, x2, y2 = box
            axes[2, index].add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            axes[2, index].text(
                x1,
                y1,
                f"{score:.2f}",
                color="white",
                fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )
        axes[2, index].set_title(
            f"Graha instances ({len(prediction['scores'])} detections)"
        )
        image_height, image_width = vis.shape
        for row in axes[:, index]:
            row.set_xlim(-0.5, image_width - 0.5)
            row.set_ylim(image_height - 0.5, -0.5)
            row.set_aspect("equal", adjustable="box")
            row.axis("off")
    fig.suptitle(f"Graha instance inference: {n_channels} input channels", y=1.0)
    output_path = output_dir / "graha_instance_inference_viz.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    if verbose:
        print(f"Saved instance visualization to {output_path}")
    return fig
