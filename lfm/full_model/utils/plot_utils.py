"""Plotting helpers for Graha/Lunar-FM segmentation notebooks."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap


def calculate_f1_score(pred: np.ndarray, label: np.ndarray) -> float:
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * (precision * recall) / (precision + recall + 1e-8))


def prepare_image_for_display(
    img: np.ndarray,
    *,
    method: str = "minmax",
    clip_percentile: float = 2.0,
    std_clip: float = 3.0,
) -> tuple[np.ndarray, str]:
    """Convert HWC multispectral image to displayable grayscale/RGB."""
    num_channels = img.shape[2]
    if method == "percentile":
        img_normalized = np.zeros_like(img, dtype=np.float32)
        for c in range(num_channels):
            band = img[:, :, c]
            p_low, p_high = np.percentile(band, [clip_percentile, 100 - clip_percentile])
            band = np.clip(band, p_low, p_high)
            img_normalized[:, :, c] = (band - p_low) / (p_high - p_low + 1e-8)
    elif method == "std_clip":
        img_normalized = (np.clip(img, -std_clip, std_clip) + std_clip) / (2 * std_clip)
    elif method == "minmax":
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
    elif method is None:
        img_normalized = img
    else:
        raise ValueError(f"Unknown display method: {method}")
    img_normalized = np.clip(img_normalized, 0, 1)

    if num_channels == 1:
        return img_normalized[:, :, 0], "Grayscale"
    if num_channels in (5, 7):
        return img_normalized[:, :, [3, 1, 0]], f"{num_channels}ch (RGB from bands 3,1,0)"
    if num_channels == 3:
        return img_normalized[:, :, [2, 1, 0]], "RGB (BGR->RGB)"
    return img_normalized[:, :, :3], f"{num_channels}ch (first 3)"


def create_overlay_image(img_vis: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    return create_colored_overlay_image(img_vis, pred_mask, color=(1.0, 1.0, 0.0))


def create_colored_overlay_image(
    img_vis: np.ndarray,
    pred_mask: np.ndarray,
    *,
    color: tuple[float, float, float],
) -> np.ndarray:
    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis

    overlay = img_rgb.copy()
    mask_bool = pred_mask == 1
    overlay[mask_bool, 0] = color[0]
    overlay[mask_bool, 1] = color[1]
    overlay[mask_bool, 2] = color[2]
    alpha = np.where(mask_bool[:, :, None], 0.5, 0.0)
    return np.clip(overlay * alpha + img_rgb * (1 - alpha), 0, 1)


def _move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch


def _extract_image_and_mask(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported validation batch type for plotting: {type(batch)}")


def _extract_logits(model_output) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if hasattr(model_output, "output"):
        return model_output.output
    raise TypeError(f"Unsupported model output type for plotting: {type(model_output)}")


def _extract_paths(batch) -> tuple[list[str | None], list[str | None]]:
    if isinstance(batch, dict):
        filenames = batch.get("filename")
        if filenames is None:
            batch_size = batch["image"].shape[0]
            return [None] * batch_size, [None] * batch_size
        if isinstance(filenames, (str, Path)):
            filenames = [str(filenames)]
        return [str(path) for path in filenames], [None] * len(filenames)
    if isinstance(batch, (tuple, list)):
        batch_size = batch[0].shape[0]
        image_paths = batch[2] if len(batch) > 2 else [None] * batch_size
        label_paths = batch[3] if len(batch) > 3 else [None] * batch_size
        return (
            [str(path) if path is not None else None for path in image_paths],
            [str(path) if path is not None else None for path in label_paths],
        )
    return [None], [None]


def _prediction_probabilities(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] > 1:
        return torch.softmax(output, dim=1)[:, 1]
    return torch.sigmoid(output[:, 0])


def _sample_key(image_path: str | None, sample_idx: int) -> str:
    if image_path:
        return Path(image_path).stem
    return f"sample_{sample_idx:04d}"


def _display_sample_key(sample_key: str) -> str:
    """Shorten chip stem to the stable M..._r..._c... identifier."""
    return sample_key.split("_input", 1)[0]


def _get_split_dataloader(datamodule, split: str):
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split}")


def plot_validation_predictions(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    n_samples: int = 5,
    filename: str = "validation_predictions.png",
    plots_subdir: str | Path = "plots",
    display_method: str = "minmax",
    dpi: int = 300,
    setup_datamodule: bool = True,
) -> Path:
    """Save a 4-row prediction figure for the first validation samples."""
    plots_dir = Path(output_dir) / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / filename

    if setup_datamodule:
        datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    batch = _move_batch_to_device(batch, task.device)
    x, y = _extract_image_and_mask(batch)

    was_training = task.training
    task.eval()
    with torch.no_grad():
        output = _extract_logits(task(x))
        if output.shape[1] > 1:
            pred = output.argmax(dim=1)
        else:
            pred = (torch.sigmoid(output[:, 0]) > 0.5).long()

    images = x.detach().cpu()[:n_samples]
    labels = y.detach().cpu()[:n_samples]
    preds = pred.detach().cpu()[:n_samples]
    batch_size = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(4, batch_size, figsize=(4 * batch_size, 16))
    if batch_size == 1:
        axes = axes.reshape(4, 1)

    cmap_pred = ListedColormap(["black", "yellow"])
    cmap_label = ListedColormap(["black", "red"])
    f1_scores = []
    display_note = None

    for i in range(batch_size):
        img = images[i].numpy().transpose(1, 2, 0)
        label = labels[i].numpy()
        pred_i = preds[i].numpy()
        f1 = calculate_f1_score(pred_i, label)
        f1_scores.append(f1)

        img_vis, note = prepare_image_for_display(img, method=display_method)
        display_note = display_note or note
        cmap_image = "gray" if img_vis.ndim == 2 else None

        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(f"Image {i}\nF1: {f1:.3f}", fontsize=12)
        axes[1, i].imshow(pred_i, cmap=cmap_pred, vmin=0, vmax=1)
        axes[1, i].set_title(f"Prediction {i}", fontsize=12)
        axes[2, i].imshow(create_overlay_image(img_vis, pred_i))
        axes[2, i].set_title(f"Overlay {i}", fontsize=12)
        axes[3, i].imshow(label, cmap=cmap_label, vmin=0, vmax=1)
        axes[3, i].set_title(f"Ground Truth {i}", fontsize=12)
        for row in range(4):
            axes[row, i].axis("off")

    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    fig.suptitle(
        f"Validation Predictions - Mean F1: {mean_f1:.3f} | {display_note}",
        fontsize=16,
        fontweight="bold",
    )
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    task.train(was_training)
    print(f"Saved validation plot to {save_path}")
    return save_path


def save_prediction_cache(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    model_name: str,
    split: str = "val",
    n_samples: int = 20,
    setup_datamodule: bool = True,
) -> Path:
    """Run one model over a split and save lightweight prediction .npz files."""
    cache_dir = Path(output_dir) / "prediction_cache" / model_name / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else split)
    dataloader = _get_split_dataloader(datamodule, split)

    was_training = task.training
    task.eval()

    manifest = []
    saved = 0
    device = task.device

    with torch.no_grad():
        for batch in dataloader:
            image_paths, label_paths = _extract_paths(batch)
            batch = _move_batch_to_device(batch, device)
            x, y = _extract_image_and_mask(batch)
            output = _extract_logits(task(x))
            probs = _prediction_probabilities(output)
            preds = (probs > 0.5).long()

            images = x.detach().cpu()
            labels = y.detach().cpu()
            probs = probs.detach().cpu()
            preds = preds.detach().cpu()

            for i in range(images.shape[0]):
                if saved >= n_samples:
                    break
                image_path = image_paths[i] if i < len(image_paths) else None
                label_path = label_paths[i] if i < len(label_paths) else None
                sample_key = _sample_key(image_path, saved)
                filename = f"{saved:04d}_{sample_key}.npz"
                save_path = cache_dir / filename

                np.savez_compressed(
                    save_path,
                    image=images[i].numpy(),
                    label=labels[i].numpy(),
                    pred=preds[i].numpy(),
                    prob=probs[i].numpy(),
                    sample_key=sample_key,
                    model_name=model_name,
                    image_path=image_path or "",
                    label_path=label_path or "",
                )
                manifest.append(
                    {
                        "index": saved,
                        "sample_key": sample_key,
                        "file": filename,
                        "image_path": image_path,
                        "label_path": label_path,
                    }
                )
                saved += 1
            if saved >= n_samples:
                break

    task.train(was_training)
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {saved} prediction cache file(s) to {cache_dir}")
    return cache_dir


def _load_prediction_cache(cache_dir: str | Path) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        files = [cache_dir / item["file"] for item in manifest]
    else:
        files = sorted(cache_dir.glob("*.npz"))

    samples = {}
    for path in files:
        data = np.load(path, allow_pickle=False)
        sample_key = str(data["sample_key"])
        samples[sample_key] = {
            "file": path,
            "image": data["image"],
            "label": data["label"],
            "pred": data["pred"],
            "prob": data["prob"],
            "image_path": str(data["image_path"]),
            "label_path": str(data["label_path"]),
        }
    return samples


def _model_display_name(model_name: str) -> str:
    if model_name.lower() == "toy":
        return "Toy"
    if model_name.lower() == "graha":
        return "Graha"
    return model_name.replace("_", " ").title()


def _model_color(model_name: str) -> tuple[float, float, float]:
    if model_name.lower() == "graha":
        return (0.0, 0.85, 1.0)
    return (1.0, 1.0, 0.0)


def _binary_metrics(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool).reshape(-1)
    label_bool = label.astype(bool).reshape(-1)

    tp = float(np.sum(pred_bool & label_bool))
    fp = float(np.sum(pred_bool & ~label_bool))
    fn = float(np.sum(~pred_bool & label_bool))
    tn = float(np.sum(~pred_bool & ~label_bool))
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "pixel_accuracy": accuracy,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": f1,
        "iou": iou,
        "predicted_foreground_fraction": float(np.mean(pred_bool)),
        "ground_truth_foreground_fraction": float(np.mean(label_bool)),
    }


def evaluate_prediction_caches(
    cache_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    filename_prefix: str = "prediction_cache_metrics",
) -> tuple[list[dict], list[dict]]:
    """Compute comparable binary segmentation metrics from prediction caches."""
    loaded = {name: _load_prediction_cache(path) for name, path in cache_dirs.items()}
    if not loaded:
        raise ValueError("No prediction caches were provided.")

    first_model = next(iter(loaded))
    shared_keys = set(loaded[first_model])
    for samples in loaded.values():
        shared_keys &= set(samples)
    sample_keys = sorted(shared_keys)
    if not sample_keys:
        raise ValueError("Prediction caches do not contain matching sample keys.")

    rows = []
    for model_name, samples in loaded.items():
        display_name = _model_display_name(model_name)
        for sample_key in sample_keys:
            sample = samples[sample_key]
            metrics = _binary_metrics(sample["pred"], sample["label"])
            rows.append(
                {
                    "model": display_name,
                    "sample_key": _display_sample_key(sample_key),
                    **metrics,
                }
            )

    summary_rows = []
    metric_names = [
        "pixel_accuracy",
        "foreground_precision",
        "foreground_recall",
        "foreground_f1",
        "iou",
        "predicted_foreground_fraction",
        "ground_truth_foreground_fraction",
    ]
    for model_name in [_model_display_name(name) for name in loaded]:
        model_rows = [row for row in rows if row["model"] == model_name]
        summary = {"model": model_name, "n_samples": len(model_rows)}
        for metric in metric_names:
            summary[metric] = float(np.mean([row[metric] for row in model_rows]))
        summary_rows.append(summary)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"
    summary_json_path = output_dir / f"{filename_prefix}_summary.json"
    summary_csv_path = output_dir / f"{filename_prefix}_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = ["model", "sample_key", *metric_names]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_fieldnames = ["model", "n_samples", *metric_names]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-sample metrics to {csv_path}")
    print(f"Saved summary metrics to {summary_csv_path}")
    return rows, summary_rows


def plot_prediction_cache_comparison(
    cache_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    n_samples: int = 5,
    filename: str = "side_by_side_predictions.png",
    display_method: str = "minmax",
    dpi: int = 200,
) -> Path:
    """Create side-by-side plots from saved prediction caches."""
    loaded = {name: _load_prediction_cache(path) for name, path in cache_dirs.items()}
    if not loaded:
        raise ValueError("No prediction caches were provided.")

    first_model = next(iter(loaded))
    shared_keys = set(loaded[first_model])
    for samples in loaded.values():
        shared_keys &= set(samples)
    sample_keys = sorted(shared_keys)[:n_samples]
    if not sample_keys:
        raise ValueError("Prediction caches do not contain matching sample keys.")

    model_names = list(loaded)
    n_rows = 2 + (2 * len(model_names))
    n_cols = len(sample_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    cmap_label = ListedColormap(["black", "red"])
    for col, sample_key in enumerate(sample_keys):
        reference = loaded[first_model][sample_key]
        img = reference["image"].transpose(1, 2, 0)
        label = reference["label"]
        img_vis, display_note = prepare_image_for_display(img, method=display_method)
        cmap_image = "gray" if img_vis.ndim == 2 else None

        axes[0, col].imshow(img_vis, cmap=cmap_image)
        axes[0, col].set_title(f"{_display_sample_key(sample_key)}\n{display_note}", fontsize=10)
        axes[1, col].imshow(label, cmap=cmap_label, vmin=0, vmax=1)
        axes[1, col].set_title("Ground Truth", fontsize=10)

        row = 2
        for model_name in model_names:
            sample = loaded[model_name][sample_key]
            pred = sample["pred"]
            f1 = calculate_f1_score(pred, label)
            display_name = _model_display_name(model_name)
            model_color = _model_color(model_name)
            cmap_pred = ListedColormap(["black", model_color])

            axes[row, col].imshow(pred, cmap=cmap_pred, vmin=0, vmax=1)
            axes[row, col].set_title(f"{display_name} Pred\nF1: {f1:.3f}", fontsize=10)
            row += 1

            axes[row, col].imshow(
                create_colored_overlay_image(img_vis, pred, color=model_color)
            )
            axes[row, col].set_title(f"{display_name} Overlay", fontsize=10)
            row += 1

        for row_idx in range(n_rows):
            axes[row_idx, col].axis("off")

    fig.suptitle(
        "Side-by-Side Segmentation Predictions",
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
    print(f"Saved side-by-side comparison plot to {save_path}")
    return save_path


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
        title = _display_sample_key(_sample_key(filenames[i] if i < len(filenames) else None, i))

        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(f"{title}\n{display_note}", fontsize=10)

        axes[1, i].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1, i].set_title(f"Instance Mask\ninstances: {int(mask.max())}", fontsize=10)

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


def _draw_boxes(ax, boxes: torch.Tensor, *, color: str, scores: torch.Tensor | None = None) -> None:
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
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )


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
        title = _display_sample_key(_sample_key(filenames[i] if i < len(filenames) else None, i))

        gt_boxes = batch["boxes"][i].detach().cpu()
        gt_masks = batch["masks"][i].detach().cpu()
        pred = predictions[i]
        pred_scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32))
        keep = pred_scores >= score_threshold
        pred_boxes = pred.get("boxes", torch.zeros((0, 4), dtype=torch.float32))[keep]
        pred_masks = pred.get("masks", torch.zeros((0, *gt_masks.shape[-2:]), dtype=torch.uint8))[keep]
        pred_scores = pred_scores[keep]

        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(f"{title}\n{display_note}", fontsize=10)

        axes[1, i].imshow(img_vis, cmap=cmap_image)
        axes[1, i].imshow(_instance_union(gt_masks), cmap=ListedColormap(["none", "red"]), alpha=0.35)
        _draw_boxes(axes[1, i], gt_boxes, color="red")
        axes[1, i].set_title(f"GT Instances: {gt_boxes.shape[0]}", fontsize=10)

        axes[2, i].imshow(img_vis, cmap=cmap_image)
        axes[2, i].imshow(_instance_union(pred_masks), cmap=ListedColormap(["none", "cyan"]), alpha=0.35)
        _draw_boxes(axes[2, i], pred_boxes, color="cyan", scores=pred_scores)
        axes[2, i].set_title(f"Pred >= {score_threshold:.2f}: {pred_boxes.shape[0]}", fontsize=10)

        axes[3, i].imshow(img_vis, cmap=cmap_image)
        axes[3, i].imshow(_instance_union(gt_masks), cmap=ListedColormap(["none", "red"]), alpha=0.25)
        axes[3, i].imshow(_instance_union(pred_masks), cmap=ListedColormap(["none", "cyan"]), alpha=0.35)
        _draw_boxes(axes[3, i], gt_boxes, color="red")
        _draw_boxes(axes[3, i], pred_boxes, color="cyan", scores=pred_scores)
        axes[3, i].set_title("GT red / Pred cyan", fontsize=10)

        for row in range(4):
            axes[row, i].axis("off")

    fig.suptitle(f"Instance Predictions - {split}", fontsize=16, fontweight="bold", y=0.995)
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance prediction plot to {save_path}")
    return save_path


class ValidationPlotCallback(Callback):
    """Save a lightweight validation prediction plot at the end of each epoch."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        n_samples: int = 5,
        every_n_epochs: int = 1,
        plots_subdir: str | Path = "plots",
        display_method: str = "minmax",
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.plots_subdir = Path(plots_subdir)
        self.display_method = display_method
        self.dpi = dpi

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.datamodule is None:
            return

        plot_validation_predictions(
            task=pl_module,
            datamodule=trainer.datamodule,
            output_dir=self.output_dir,
            n_samples=self.n_samples,
            filename=f"validation_epoch_{epoch + 1:03d}.png",
            plots_subdir=self.plots_subdir,
            display_method=self.display_method,
            dpi=self.dpi,
            setup_datamodule=False,
        )
