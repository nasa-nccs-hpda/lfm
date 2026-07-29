"""Instance segmentation plotting helpers."""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from lfm.all_models.all_tasks.data.image_io import (
    path_key,
    read_tif,
)
from lfm.all_models.all_tasks.data.tensor_utils import image_to_chw_float
from lfm.all_models.all_tasks.utils.common import (
    _display_sample_key,
    _extract_paths,
    _get_split_dataloader,
    _model_display_name,
    _sample_key,
)
from lfm.all_models.all_tasks.utils.display import prepare_image_for_display
from lfm.all_models.all_tasks.utils.metrics import _instance_metrics, calculate_f1_score
from lfm.all_models.inst_seg.prediction.instance_prediction_cache import (
    _load_instance_prediction_cache,
)


def _extract_instance_boxes(batch, sample_index: int) -> torch.Tensor:
    boxes = batch.get("crater_boxes") if isinstance(batch, dict) else None
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    if isinstance(boxes, torch.Tensor):
        return boxes[sample_index]
    return boxes[sample_index]


def plot_instance_batch_sanity(
    datamodule,
    output_dir: str | Path,
    *,
    split: str = "train",
    n_samples: int = 5,
    filename: str = "instance_batch_sanity.png",
    plots_subdir: str | Path = "plots",
    display_method: str = "minmax",
    dpi: int = 200,
    setup_datamodule: bool = True,
) -> Path:
    """Save a visual sanity check for instance masks and cropped xyxy boxes."""
    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else "test")
    dataloader = _get_split_dataloader(datamodule, split)
    batch = next(iter(dataloader))

    images = batch["image"].detach().cpu()
    masks = batch["mask"].detach().cpu()
    filenames, _ = _extract_paths(batch)
    batch_size = min(n_samples, images.shape[0])

    plots_dir = Path(output_dir) / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / filename

    fig, axes = plt.subplots(3, batch_size, figsize=(4 * batch_size, 11))
    if batch_size == 1:
        axes = axes.reshape(3, 1)

    for i in range(batch_size):
        image = images[i].numpy().transpose(1, 2, 0)
        mask = masks[i].numpy()
        boxes = _extract_instance_boxes(batch, i).detach().cpu()
        img_vis, display_note = prepare_image_for_display(image, method=display_method)
        cmap_image = "gray" if img_vis.ndim == 2 else None
        title = _display_sample_key(
            _sample_key(filenames[i] if i < len(filenames) else None, i)
        )

        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(f"{title}\n{display_note}", fontsize=10)

        axes[1, i].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1, i].set_title(
            f"Instance Mask\ninstances: {int(mask.max())}", fontsize=10
        )

        axes[2, i].imshow(img_vis, cmap=cmap_image)
        axes[2, i].set_title(f"Cropped Boxes\nboxes: {boxes.shape[0]}", fontsize=10)
        for box in boxes.tolist():
            x1, y1, x2, y2 = box[:4]
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            if width <= 0 or height <= 0:
                continue
            axes[2, i].add_patch(
                Rectangle(
                    (x1, y1),
                    width,
                    height,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                )
            )

        for row in range(3):
            axes[row, i].axis("off")

    fig.suptitle(
        f"Instance Segmentation Sanity Check - {split}",
        fontsize=16,
        fontweight="bold",
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance sanity plot to {save_path}")
    return save_path


def _to_cpu_prediction_list(predictions) -> list[dict[str, torch.Tensor]]:
    if hasattr(predictions, "output"):
        predictions = predictions.output
    result = []
    for pred in predictions:
        result.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in pred.items()
            }
        )
    return result


def _instance_union(mask_tensor: torch.Tensor) -> np.ndarray:
    if mask_tensor.numel() == 0:
        shape = mask_tensor.shape[-2:] if mask_tensor.ndim >= 2 else (1, 1)
        return np.zeros(tuple(shape), dtype=bool)
    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor[:, 0]
    if mask_tensor.ndim == 3:
        return (mask_tensor > 0).any(dim=0).numpy()
    if mask_tensor.ndim == 2:
        return (mask_tensor > 0).numpy()
    raise ValueError(f"Unsupported instance mask shape: {tuple(mask_tensor.shape)}")


def _draw_boxes(
    ax, boxes: torch.Tensor, *, color: str, scores: torch.Tensor | None = None
) -> None:
    if boxes.numel() == 0:
        return
    for i, box in enumerate(boxes.detach().cpu().tolist()):
        x1, y1, x2, y2 = box[:4]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0 or height <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        if scores is not None and i < scores.numel():
            ax.text(
                x1,
                y1,
                f"{float(scores[i]):.2f}",
                color=color,
                fontsize=7,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.45,
                    "pad": 1,
                    "edgecolor": "none",
                },
            )


def _instance_union_np(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _xywh_to_xyxy_np(boxes: np.ndarray | None) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])[:, :4].astype(np.float32)
    xyxy = boxes.copy()
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    return torch.as_tensor(xyxy, dtype=torch.float32)


def _find_unsplit_instance_pairs(root: str | Path) -> dict[str, tuple[Path, Path]]:
    root = Path(root)
    chips_dir = root / "chips"
    labels_dir = root / "labels"
    chips = {
        path_key(path, "_input_wac_static_chip"): path
        for path in sorted(chips_dir.glob("*.tif"))
    }
    labels = {
        path_key(path, "_label"): path
        for path in sorted(labels_dir.glob("*_label.npz"))
    }
    return {key: (chips[key], labels[key]) for key in sorted(set(chips) & set(labels))}


def _find_split_instance_pairs(root: str | Path) -> dict[str, tuple[Path, Path]]:
    root = Path(root)
    pairs: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        for key, pair in _find_unsplit_instance_pairs(split_root).items():
            pairs.setdefault(key, pair)
    return pairs


def _load_instance_comparison_sample(pair: tuple[Path, Path]) -> dict[str, object]:
    chip_path, label_path = pair
    image = image_to_chw_float(read_tif(chip_path))
    if image.shape[0] > 7:
        image = image[:7]
    with np.load(label_path) as data:
        mask = np.asarray(data["mask"])
        boxes = np.asarray(data["bboxes"]) if "bboxes" in data else None
    return {
        "chip_path": chip_path,
        "label_path": label_path,
        "image": image.numpy().transpose(1, 2, 0),
        "mask": mask,
        "boxes": _xywh_to_xyxy_np(boxes),
    }


def _overlay_instance_gt(
    img_vis: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float = 0.35,
) -> np.ndarray:
    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis.copy()
    union = _instance_union_np(mask)
    overlay = img_rgb.copy()
    overlay[union, 0] = color[0]
    overlay[union, 1] = color[1]
    overlay[union, 2] = color[2]
    return np.where(union[:, :, None], overlay * alpha + img_rgb * (1 - alpha), img_rgb)


def plot_instance_label_comparison(
    kaguya_root: str | Path,
    split_data_root: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 8,
    filename: str = "iseg_label_comparison.png",
    plots_subdir: str | Path = "plots",
    display_method: str = "minmax",
    dpi: int = 200,
) -> Path:
    """Compare raw Kaguya instance labels against the split instance dataset."""
    kaguya_pairs = _find_unsplit_instance_pairs(kaguya_root)
    split_pairs = _find_split_instance_pairs(split_data_root)
    sample_keys = sorted(set(kaguya_pairs) & set(split_pairs))[:n_samples]
    if not sample_keys:
        raise ValueError(
            f"No matching instance chip/label pairs found between {kaguya_root} "
            f"and {split_data_root}"
        )

    green = (0.0, 1.0, 0.0)
    red = (1.0, 0.0, 0.0)
    n_cols = len(sample_keys)
    fig, axes = plt.subplots(6, n_cols, figsize=(4 * n_cols, 22))
    if n_cols == 1:
        axes = axes.reshape(6, 1)

    for col, key in enumerate(sample_keys):
        kaguya = _load_instance_comparison_sample(kaguya_pairs[key])
        split = _load_instance_comparison_sample(split_pairs[key])
        kaguya_img, display_note = prepare_image_for_display(
            kaguya["image"], method=display_method
        )
        split_img, _ = prepare_image_for_display(split["image"], method=display_method)
        cmap_kaguya = "gray" if kaguya_img.ndim == 2 else None
        cmap_split = "gray" if split_img.ndim == 2 else None

        kaguya_mask = kaguya["mask"]
        split_mask = split["mask"]
        kaguya_boxes = kaguya["boxes"]
        split_boxes = split["boxes"]

        black = np.zeros((*kaguya_mask.shape, 3), dtype=np.float32)
        kaguya_black = _overlay_instance_gt(black, kaguya_mask, color=green, alpha=0.8)
        split_black = _overlay_instance_gt(black, split_mask, color=red, alpha=0.8)
        both_black = _overlay_instance_gt(
            kaguya_black, split_mask, color=red, alpha=0.8
        )
        both_on_kaguya = _overlay_instance_gt(
            kaguya_img, kaguya_mask, color=green, alpha=0.35
        )
        both_on_kaguya = _overlay_instance_gt(
            both_on_kaguya, split_mask, color=red, alpha=0.35
        )

        axes[0, col].imshow(kaguya_black)
        axes[0, col].set_title(
            f"{_display_sample_key(key)}\nKaguya GT base\n{display_note}",
            fontsize=9,
        )
        _draw_boxes(axes[0, col], kaguya_boxes, color="lime")

        axes[1, col].imshow(
            _overlay_instance_gt(kaguya_img, kaguya_mask, color=green), cmap=cmap_kaguya
        )
        axes[1, col].set_title("Kaguya GT on Kaguya chip", fontsize=9)
        _draw_boxes(axes[1, col], kaguya_boxes, color="lime")

        axes[2, col].imshow(split_black)
        axes[2, col].set_title("New GT base", fontsize=9)
        _draw_boxes(axes[2, col], split_boxes, color="red")

        axes[3, col].imshow(
            _overlay_instance_gt(split_img, split_mask, color=red), cmap=cmap_split
        )
        axes[3, col].set_title("New GT on new chip", fontsize=9)
        _draw_boxes(axes[3, col], split_boxes, color="red")

        axes[4, col].imshow(both_black)
        axes[4, col].set_title("Both GT on black", fontsize=9)
        _draw_boxes(axes[4, col], kaguya_boxes, color="lime")
        _draw_boxes(axes[4, col], split_boxes, color="red")

        axes[5, col].imshow(both_on_kaguya, cmap=cmap_kaguya)
        axes[5, col].set_title("Both GT on Kaguya chip", fontsize=9)
        _draw_boxes(axes[5, col], kaguya_boxes, color="lime")
        _draw_boxes(axes[5, col], split_boxes, color="red")

        for row in range(6):
            axes[row, col].axis("off")

    fig.suptitle(
        "Instance Label Comparison - Kaguya green / New split red",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plots_dir = Path(output_dir) / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance label comparison plot to {save_path}")
    return save_path


def plot_instance_predictions(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    split: str = "val",
    n_samples: int = 5,
    filename: str = "instance_predictions.png",
    plots_subdir: str | Path = "plots/instance_predictions",
    score_threshold: float = 0.5,
    display_method: str = "minmax",
    dpi: int = 200,
    setup_datamodule: bool = True,
) -> Path:
    """Save true instance prediction plots with GT and predicted masks/boxes."""
    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else "test")

    dataloader = _get_split_dataloader(datamodule, split)
    batch = next(iter(dataloader))
    device = task.device
    x = batch["image"].to(device)

    was_training = task.training
    task.eval()
    with torch.no_grad():
        predictions = task.predict_step({"image": x}, batch_idx=0)
    if was_training:
        task.train()
    predictions = _to_cpu_prediction_list(predictions)

    images = batch["image"].detach().cpu()
    filenames, _ = _extract_paths(batch)
    batch_size = min(n_samples, images.shape[0], len(predictions))

    plots_dir = Path(output_dir) / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / filename

    fig, axes = plt.subplots(4, batch_size, figsize=(4 * batch_size, 14))
    if batch_size == 1:
        axes = axes.reshape(4, 1)

    for i in range(batch_size):
        image = images[i].numpy().transpose(1, 2, 0)
        img_vis, display_note = prepare_image_for_display(image, method=display_method)
        cmap_image = "gray" if img_vis.ndim == 2 else None
        title = _display_sample_key(
            _sample_key(filenames[i] if i < len(filenames) else None, i)
        )

        gt_boxes = batch["boxes"][i].detach().cpu()
        gt_masks = batch["masks"][i].detach().cpu()
        pred = predictions[i]
        pred_scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32))
        keep = pred_scores >= score_threshold
        pred_boxes = pred.get("boxes", torch.zeros((0, 4), dtype=torch.float32))[keep]
        pred_masks = pred.get(
            "masks", torch.zeros((0, *gt_masks.shape[-2:]), dtype=torch.uint8)
        )[keep]
        pred_scores = pred_scores[keep]

        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(f"{title}\n{display_note}", fontsize=10)

        axes[1, i].imshow(img_vis, cmap=cmap_image)
        axes[1, i].imshow(
            _instance_union(gt_masks), cmap=ListedColormap(["none", "red"]), alpha=0.35
        )
        _draw_boxes(axes[1, i], gt_boxes, color="red")
        axes[1, i].set_title(f"GT Instances: {gt_boxes.shape[0]}", fontsize=10)

        axes[2, i].imshow(img_vis, cmap=cmap_image)
        axes[2, i].imshow(
            _instance_union(pred_masks),
            cmap=ListedColormap(["none", "cyan"]),
            alpha=0.35,
        )
        _draw_boxes(axes[2, i], pred_boxes, color="cyan", scores=pred_scores)
        axes[2, i].set_title(
            f"Pred >= {score_threshold:.2f}: {pred_boxes.shape[0]}", fontsize=10
        )

        axes[3, i].imshow(img_vis, cmap=cmap_image)
        axes[3, i].imshow(
            _instance_union(gt_masks), cmap=ListedColormap(["none", "red"]), alpha=0.25
        )
        axes[3, i].imshow(
            _instance_union(pred_masks),
            cmap=ListedColormap(["none", "cyan"]),
            alpha=0.35,
        )
        _draw_boxes(axes[3, i], gt_boxes, color="red")
        _draw_boxes(axes[3, i], pred_boxes, color="cyan", scores=pred_scores)
        axes[3, i].set_title("GT red / Pred cyan", fontsize=10)

        for row in range(4):
            axes[row, i].axis("off")

    fig.suptitle(
        f"Instance Predictions - {split}", fontsize=16, fontweight="bold", y=0.995
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance prediction plot to {save_path}")
    return save_path


def _instance_colormap(instance_mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Generate deterministic high-contrast colors for instance ids."""
    if torch.is_tensor(instance_mask):
        instance_mask = instance_mask.detach().cpu().numpy()
    instance_mask = np.asarray(instance_mask).astype(np.int32)
    h, w = instance_mask.shape[-2:]
    rgb_image = np.zeros((h, w, 3), dtype=np.float32)
    golden_ratio = 0.618033988749895

    for instance_id in np.unique(instance_mask):
        if int(instance_id) == 0:
            continue
        hue = (int(instance_id) * golden_ratio) % 1.0
        saturation = 0.9 if int(instance_id) % 2 == 0 else 0.7
        value = 0.95 if (int(instance_id) // 2) % 2 == 0 else 0.75
        rgb_image[instance_mask == instance_id] = colorsys.hsv_to_rgb(
            hue,
            saturation,
            value,
        )
    return rgb_image


def _instance_overlay(
    img_vis: np.ndarray,
    instance_mask: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis
    colored_mask = _instance_colormap(instance_mask)
    alpha_mask = (np.asarray(instance_mask) != 0).astype(np.float32)[:, :, None] * alpha
    return np.clip(img_rgb * (1 - alpha_mask) + colored_mask * alpha_mask, 0, 1)


def _colored_instance_overlay(
    img_vis: np.ndarray,
    instance_mask: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float = 0.35,
) -> np.ndarray:
    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis.copy()
    union = np.asarray(instance_mask) > 0
    overlay = img_rgb.copy()
    overlay[union, 0] = color[0]
    overlay[union, 1] = color[1]
    overlay[union, 2] = color[2]
    return np.where(union[:, :, None], overlay * alpha + img_rgb * (1 - alpha), img_rgb)


def plot_instance_cache_predictions(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    n_samples: int = 5,
    filename: str = "instance_predictions.png",
    display_method: str = "minmax",
    dpi: int = 200,
) -> Path:
    """Save one model's instance cache using the toy instance driver plot style."""
    samples = _load_instance_prediction_cache(cache_dir)
    sample_keys = sorted(samples)[:n_samples]
    n_cols = len(sample_keys)
    if n_cols == 0:
        raise ValueError(f"No instance cache samples found in {cache_dir}")
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))
    if n_cols == 1:
        axes = axes.reshape(4, 1)
    instance_metrics = []
    semantic_f1s = []
    display_note = None

    for col, sample_key in enumerate(sample_keys):
        sample = samples[sample_key]
        img = sample["image"].transpose(1, 2, 0)
        img_vis, display_note = prepare_image_for_display(img, method=display_method)
        cmap_image = "gray" if img_vis.ndim == 2 else None
        pred_mask = sample["pred_mask"].astype(np.int32)
        gt_mask = sample["gt_mask"].astype(np.int32)
        metrics = _instance_metrics(pred_mask, gt_mask)
        sem_f1 = calculate_f1_score(
            (pred_mask > 0).astype(np.uint8), (gt_mask > 0).astype(np.uint8)
        )
        instance_metrics.append(metrics)
        semantic_f1s.append(sem_f1)

        axes[0, col].imshow(img_vis, cmap=cmap_image)
        axes[0, col].set_title(
            f"{_display_sample_key(sample_key)}\n"
            f"Sem F1: {sem_f1:.3f}\n"
            f"Inst F1: {metrics['f1']:.3f}\n"
            f"Pred: {metrics['num_pred']} | GT: {metrics['num_gt']}",
            fontsize=12,
            fontweight="bold",
        )

        axes[1, col].imshow(_instance_colormap(pred_mask), vmin=0, vmax=1)
        _draw_boxes(axes[1, col], torch.as_tensor(sample["pred_boxes"]), color="red")
        axes[1, col].set_title(
            f"Predicted ({metrics['num_pred']} instances)\n"
            f"Precision: {metrics['precision']:.3f}",
            fontsize=11,
        )

        axes[2, col].imshow(
            _instance_overlay(img_vis, pred_mask, alpha=0.5), vmin=0, vmax=1
        )
        _draw_boxes(axes[2, col], torch.as_tensor(sample["pred_boxes"]), color="red")
        axes[2, col].set_title(
            f"Prediction Overlay\n" f"Mean IoU: {metrics['mean_iou']:.3f}",
            fontsize=11,
        )

        axes[3, col].imshow(_instance_colormap(gt_mask), vmin=0, vmax=1)
        _draw_boxes(axes[3, col], torch.as_tensor(sample["gt_boxes"]), color="red")
        axes[3, col].set_title(
            f"Ground Truth ({metrics['num_gt']} instances)\n"
            f"Recall: {metrics['recall']:.3f}",
            fontsize=11,
        )
        for row in range(4):
            axes[row, col].axis("off")

    avg_inst_f1 = (
        float(np.mean([m["f1"] for m in instance_metrics])) if instance_metrics else 0.0
    )
    avg_sem_f1 = float(np.mean(semantic_f1s)) if semantic_f1s else 0.0
    avg_precision = (
        float(np.mean([m["precision"] for m in instance_metrics]))
        if instance_metrics
        else 0.0
    )
    avg_recall = (
        float(np.mean([m["recall"] for m in instance_metrics]))
        if instance_metrics
        else 0.0
    )
    avg_iou = (
        float(np.mean([m["mean_iou"] for m in instance_metrics]))
        if instance_metrics
        else 0.0
    )
    num_channels = samples[sample_keys[0]]["image"].shape[0]
    fig.suptitle(
        f"{_model_display_name(model_name)} Instance Segmentation Results\n"
        f"Semantic F1: {avg_sem_f1:.3f} | Instance F1: {avg_inst_f1:.3f} | "
        f"Precision: {avg_precision:.3f} | Recall: {avg_recall:.3f} | "
        f"Mean IoU: {avg_iou:.3f}\n"
        f"Input: {num_channels}ch",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance cache plot to {save_path}")
    return save_path


def plot_instance_cache_comparison(
    cache_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    n_samples: int = 5,
    filename: str = "side_by_side_instance_predictions.png",
    display_method: str = "minmax",
    dpi: int = 200,
) -> Path:
    """Create side-by-side true-instance plots using driver-style visuals."""
    loaded = {
        name: _load_instance_prediction_cache(path) for name, path in cache_dirs.items()
    }
    first_model = next(iter(loaded))
    shared_keys = set(loaded[first_model])
    for samples in loaded.values():
        shared_keys &= set(samples)
    sample_keys = sorted(shared_keys)[:n_samples]
    if not sample_keys:
        raise ValueError(
            "Instance prediction caches do not contain matching sample keys."
        )

    model_names = list(loaded)
    n_rows = 2 + (2 * len(model_names))
    n_cols = len(sample_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, sample_key in enumerate(sample_keys):
        reference = loaded[first_model][sample_key]
        img = reference["image"].transpose(1, 2, 0)
        img_vis, display_note = prepare_image_for_display(img, method=display_method)
        cmap_image = "gray" if img_vis.ndim == 2 else None

        axes[0, col].imshow(img_vis, cmap=cmap_image)
        axes[0, col].set_title(
            f"{_display_sample_key(sample_key)}\n{display_note}", fontsize=10
        )

        row = 1
        for model_name in model_names:
            sample = loaded[model_name][sample_key]
            pred_mask = sample["pred_mask"].astype(np.int32)
            gt_mask = reference["gt_mask"].astype(np.int32)
            metrics = _instance_metrics(pred_mask, gt_mask)
            sem_f1 = calculate_f1_score(
                (pred_mask > 0).astype(np.uint8),
                (gt_mask > 0).astype(np.uint8),
            )

            axes[row, col].imshow(_instance_colormap(pred_mask), vmin=0, vmax=1)
            _draw_boxes(
                axes[row, col],
                torch.as_tensor(sample["pred_boxes"]),
                color="red",
            )
            axes[row, col].set_title(
                f"{_model_display_name(model_name)} Pred ({metrics['num_pred']})\n"
                f"Sem F1: {sem_f1:.3f} | Inst F1: {metrics['f1']:.3f}",
                fontsize=10,
            )
            row += 1

            axes[row, col].imshow(
                _instance_overlay(img_vis, pred_mask, alpha=0.5), vmin=0, vmax=1
            )
            _draw_boxes(
                axes[row, col],
                torch.as_tensor(sample["pred_boxes"]),
                color="red",
            )
            axes[row, col].set_title(
                f"{_model_display_name(model_name)} Overlay\n"
                f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}",
                fontsize=10,
            )
            row += 1

        gt_mask = reference["gt_mask"].astype(np.int32)
        axes[row, col].imshow(_instance_colormap(gt_mask), vmin=0, vmax=1)
        _draw_boxes(axes[row, col], torch.as_tensor(reference["gt_boxes"]), color="red")
        axes[row, col].set_title(
            f"Ground Truth ({len([x for x in np.unique(gt_mask) if int(x) != 0])} instances)",
            fontsize=10,
        )

        for row_idx in range(n_rows):
            axes[row_idx, col].axis("off")

    fig.suptitle(
        "Side-by-Side Instance Segmentation Predictions",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side instance comparison plot to {save_path}")
    return save_path
