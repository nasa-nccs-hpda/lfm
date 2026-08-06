"""Semantic segmentation test-suite output helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from lfm.all_models.all_tasks import (
    CheckpointRecord,
    load_lightning_checkpoint_state,
)

SEMANTIC_TEST_SUITE_METRICS = [
    "pixel_accuracy",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "iou",
    "predicted_foreground_fraction",
    "ground_truth_foreground_fraction",
]

SEMANTIC_CHECKPOINT_METRICS = [
    "pixel_accuracy",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "iou",
    "average_precision",
    "background_average_precision",
    "mean_average_precision",
    "predicted_foreground_fraction",
    "ground_truth_foreground_fraction",
]


def semantic_metric_array(metrics: dict[str, float]) -> np.ndarray:
    row = np.zeros((), dtype=[(name, "f8") for name in SEMANTIC_TEST_SUITE_METRICS])
    for name in SEMANTIC_TEST_SUITE_METRICS:
        row[name] = float(metrics[name])
    return row


def semantic_counts(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int | None = None,
) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    label_flat = label.reshape(-1)
    if ignore_index is not None:
        valid = label_flat != int(ignore_index)
        pred_flat = pred_flat[valid]
        label_flat = label_flat[valid]
    pred_bool = pred_flat.astype(bool)
    label_bool = label_flat.astype(bool)
    return {
        "tp": float(np.sum(pred_bool & label_bool)),
        "fp": float(np.sum(pred_bool & ~label_bool)),
        "fn": float(np.sum(~pred_bool & label_bool)),
        "tn": float(np.sum(~pred_bool & ~label_bool)),
        "n": float(pred_bool.size),
        "pred_fg": float(np.sum(pred_bool)),
        "label_fg": float(np.sum(label_bool)),
    }


def semantic_metrics(counts: dict[str, float]) -> dict[str, float]:
    eps = 1e-8
    tp, fp, fn, tn, n = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
        counts["n"],
    )
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "pixel_accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(2 * precision * recall / (precision + recall + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "predicted_foreground_fraction": float(counts["pred_fg"] / (n + eps)),
        "ground_truth_foreground_fraction": float(counts["label_fg"] / (n + eps)),
    }


def write_semantic_metrics(
    output_dir: Path, metrics: dict[str, float], *, header: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", semantic_metric_array(metrics))
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        for name in SEMANTIC_TEST_SUITE_METRICS:
            f.write(f"{name}: {metrics[name]:.8f}\n")


def image_for_plot(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] >= 4:
        rgb = np.stack([image[3], image[1], image[0]], axis=-1)
    elif image.ndim == 3:
        rgb = np.moveaxis(image[: min(3, image.shape[0])], 0, -1)
        if rgb.shape[-1] == 1:
            rgb = rgb[..., 0]
    else:
        rgb = image
    arr = rgb.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def plot_semantic_test_suite_samples(
    samples: list[dict[str, np.ndarray]], save_path: Path
) -> None:
    if not samples:
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(samples)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))
    if n_cols == 1:
        axes = axes.reshape(4, 1)
    cmap_pred = ListedColormap(["black", "yellow"])
    cmap_label = ListedColormap(["black", "red"])
    for col, sample in enumerate(samples):
        image = image_for_plot(sample["image"])
        pred = sample["pred"]
        label = sample["label"]
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = np.repeat(overlay[..., None], 3, axis=-1)
        overlay[pred > 0] = [1.0, 1.0, 0.0]
        axes[0, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0, col].set_title(sample["sample_key"], fontsize=10)
        axes[1, col].imshow(pred, cmap=cmap_pred, vmin=0, vmax=1)
        axes[1, col].set_title("Prediction", fontsize=10)
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Overlay", fontsize=10)
        axes[3, col].imshow(label, cmap=cmap_label, vmin=0, vmax=1)
        axes[3, col].set_title("Ground Truth", fontsize=10)
        for row in range(4):
            axes[row, col].axis("off")
    fig.suptitle(
        "Epoch Test Suite Semantic Predictions", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved semantic epoch test-suite plot to {save_path}", flush=True)


def sample_key_from_name(filename: str | None, index: int) -> str:
    if filename:
        return Path(str(filename)).name.split("_input", 1)[0].replace("_label", "")
    return f"sample_{index:04d}"


def extract_semantic_batch(batch: Any):
    if isinstance(batch, dict):
        return (
            batch["image"],
            batch["mask"],
            batch.get("filename", [None] * batch["image"].shape[0]),
        )
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        filenames = batch[2] if len(batch) > 2 else [None] * batch[0].shape[0]
        return batch[0], batch[1], filenames
    raise TypeError(f"Unsupported semantic batch type: {type(batch)}")


def sample_key_from_path(path: str | Path | None, fallback_index: int) -> str:
    if path is None:
        return f"sample_{fallback_index:04d}"
    stem = Path(str(path)).stem
    return stem.split("_input", 1)[0]


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        return tuple(
            value.to(device) if torch.is_tensor(value) else value for value in batch
        )
    return batch.to(device) if torch.is_tensor(batch) else batch


def logits_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "output"):
        return output.output
    raise TypeError(f"Unsupported model output type: {type(output)}")


def hard_predictions(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > 1:
        return logits.argmax(dim=1).long()
    return (torch.sigmoid(logits[:, 0]) > 0.5).long()


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > 1:
        return torch.softmax(logits, dim=1)
    foreground = torch.sigmoid(logits[:, 0])
    return torch.stack([1.0 - foreground, foreground], dim=1)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if labels.size == 0 or not np.any(labels):
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels, dtype=np.float64)
    fp = np.cumsum(~labels, dtype=np.float64)
    recall = tp / max(float(labels.sum()), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def semantic_valid_arrays(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred_flat = pred.reshape(-1)
    label_flat = label.reshape(-1)
    if ignore_index is not None:
        valid = label_flat != int(ignore_index)
        pred_flat = pred_flat[valid]
        label_flat = label_flat[valid]
    return pred_flat, label_flat


def semantic_ap_metrics_from_scores(
    foreground_scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    labels_bool = np.asarray(labels).astype(bool)
    foreground_ap = average_precision(foreground_scores, labels_bool)
    background_ap = average_precision(1.0 - foreground_scores, ~labels_bool)
    return {
        "average_precision": foreground_ap,
        "background_average_precision": background_ap,
        "mean_average_precision": float((foreground_ap + background_ap) / 2.0),
    }


def semantic_checkpoint_metrics(
    counts: dict[str, float],
    ap_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    metrics = {
        **semantic_metrics(counts),
        "average_precision": 0.0,
        "background_average_precision": 0.0,
        "mean_average_precision": 0.0,
    }
    if ap_metrics:
        metrics.update(ap_metrics)
    return metrics


def empty_semantic_counts() -> dict[str, float]:
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "n": 0.0,
        "pred_fg": 0.0,
        "label_fg": 0.0,
    }


def add_semantic_counts(total: dict[str, float], part: dict[str, float]) -> None:
    for key, value in part.items():
        total[key] += value


def semantic_checkpoint_metric_array(metrics: dict[str, float]) -> np.ndarray:
    dtype = [(name, "f8") for name in SEMANTIC_CHECKPOINT_METRICS]
    row = np.zeros((), dtype=dtype)
    for name in SEMANTIC_CHECKPOINT_METRICS:
        row[name] = float(metrics[name])
    return row


def write_semantic_checkpoint_metrics(
    output_dir: Path, metrics: dict[str, float], *, header: str | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", semantic_checkpoint_metric_array(metrics))
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for name in SEMANTIC_CHECKPOINT_METRICS:
            f.write(f"{name}: {metrics[name]:.8f}\n")


def run_semantic_checkpoint(
    *,
    task: torch.nn.Module,
    test_batches,
    checkpoint: CheckpointRecord,
    output_dir: Path,
    model_name: str,
    ignore_index: int | None = None,
) -> dict[str, float]:
    load_lightning_checkpoint_state(task, checkpoint.path)
    device = next(task.parameters()).device
    was_training = task.training
    task.eval()

    checkpoint_output_dir = output_dir / checkpoint.name
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    counts_total = empty_semantic_counts()
    all_foreground_scores = []
    all_labels = []
    sample_index = 0

    batch_bar = tqdm(
        test_batches,
        desc=f"{model_name} {checkpoint.name} batches",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    with torch.no_grad():
        for batch in batch_bar:
            batch = move_batch_to_device(batch, device)
            images, labels, image_paths = extract_semantic_batch(batch)
            logits = logits_from_output(task(images))
            probs = class_probabilities(logits)
            preds = hard_predictions(logits)

            images_np = images.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            foreground_scores_np = probs[:, 1].detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                sample_key = sample_key_from_path(
                    image_paths[i] if i < len(image_paths) else None,
                    sample_index,
                )
                sample_dir = checkpoint_output_dir / sample_key
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(sample_dir / f"{sample_key}_input.npy", images_np[i])
                np.save(sample_dir / f"{sample_key}_label.npy", labels_np[i])
                np.save(sample_dir / f"{sample_key}_pred.npy", preds_np[i])
                np.save(sample_dir / f"{sample_key}_class_pred.npy", preds_np[i])
                np.save(sample_dir / f"{sample_key}_logits.npy", logits_np[i])

                sample_counts = semantic_counts(
                    preds_np[i],
                    labels_np[i],
                    ignore_index=ignore_index,
                )
                add_semantic_counts(counts_total, sample_counts)
                valid_scores, valid_labels = semantic_valid_arrays(
                    foreground_scores_np[i],
                    labels_np[i],
                    ignore_index=ignore_index,
                )
                sample_ap_metrics = semantic_ap_metrics_from_scores(
                    valid_scores,
                    valid_labels,
                )
                all_foreground_scores.append(valid_scores.reshape(-1))
                all_labels.append(valid_labels.reshape(-1))
                sample_metrics = semantic_checkpoint_metrics(
                    sample_counts, sample_ap_metrics
                )
                write_semantic_checkpoint_metrics(
                    sample_dir,
                    sample_metrics,
                    header=(
                        f"model: {model_name}\n"
                        f"checkpoint: {checkpoint.path}\n"
                        f"sample_key: {sample_key}"
                    ),
                )
                sample_index += 1
            batch_bar.set_postfix(samples=sample_index)

    aggregate_ap_metrics = (
        semantic_ap_metrics_from_scores(
            np.concatenate(all_foreground_scores), np.concatenate(all_labels)
        )
        if all_foreground_scores
        else None
    )
    aggregate_metrics = semantic_checkpoint_metrics(counts_total, aggregate_ap_metrics)
    write_semantic_checkpoint_metrics(
        checkpoint_output_dir,
        aggregate_metrics,
        header=(
            f"model: {model_name}\n"
            f"checkpoint: {checkpoint.path}\n"
            f"epoch: {checkpoint.epoch}\n"
            f"samples: {sample_index}"
        ),
    )
    task.train(was_training)
    print(
        f"[{model_name}] {checkpoint.name}: "
        f"F1={aggregate_metrics['foreground_f1']:.4f}, "
        f"IoU={aggregate_metrics['iou']:.4f}, "
        f"AP={aggregate_metrics['average_precision']:.4f}, "
        f"mAP={aggregate_metrics['mean_average_precision']:.4f}, samples={sample_index}",
        flush=True,
    )
    return aggregate_metrics


def run_semantic_test_suite(
    *,
    datamodule,
    task,
    output_dir: Path,
    model_name: str,
    split: str,
    n_samples: int,
    suite_name: str,
    epoch: int | None = None,
    ignore_index: int | None = None,
) -> tuple[dict[str, float], int]:
    datamodule.setup("fit" if split in {"train", "val"} else "test")
    dataloader = getattr(datamodule, f"{split}_dataloader")()
    device = task.device
    was_training = task.training
    task.eval()
    suite_dir = output_dir / "test_suite" / model_name / suite_name
    total = {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "n": 0.0,
        "pred_fg": 0.0,
        "label_fg": 0.0,
    }
    saved = 0
    plot_samples = []
    with torch.no_grad():
        for batch in dataloader:
            images, labels, filenames = extract_semantic_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            output = task(images)
            logits = output.output if hasattr(output, "output") else output
            preds = (
                logits.argmax(dim=1).long()
                if logits.shape[1] > 1
                else (torch.sigmoid(logits[:, 0]) > 0.5).long()
            )
            images_np = images.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            for i in range(images_np.shape[0]):
                if saved >= n_samples:
                    break
                sample_key = sample_key_from_name(
                    filenames[i] if i < len(filenames) else None, saved
                )
                sample_dir = suite_dir / sample_key
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / f"{sample_key}_input.npy", images_np[i])
                np.save(sample_dir / f"{sample_key}_label.npy", labels_np[i])
                np.save(sample_dir / f"{sample_key}_pred.npy", preds_np[i])
                if len(plot_samples) < 5:
                    plot_samples.append(
                        {
                            "sample_key": sample_key,
                            "image": images_np[i],
                            "label": labels_np[i],
                            "pred": preds_np[i],
                        }
                    )
                counts = semantic_counts(
                    preds_np[i],
                    labels_np[i],
                    ignore_index=ignore_index,
                )
                for key, value in counts.items():
                    total[key] += value
                write_semantic_metrics(
                    sample_dir,
                    semantic_metrics(counts),
                    header=(
                        f"model: {model_name}\n"
                        f"{'epoch' if epoch is not None else 'suite'}: "
                        f"{epoch if epoch is not None else suite_name}\n"
                        f"sample_key: {sample_key}"
                    ),
                )
                saved += 1
            if saved >= n_samples:
                break
    aggregate = semantic_metrics(total)
    write_semantic_metrics(
        suite_dir,
        aggregate,
        header=(
            f"model: {model_name}\n"
            f"{'epoch' if epoch is not None else 'suite'}: "
            f"{epoch if epoch is not None else suite_name}\n"
            f"split: {split}\n"
            f"samples: {saved}"
        ),
    )
    plot_semantic_test_suite_samples(
        plot_samples,
        suite_dir / f"{split}_semantic_predictions.png",
    )
    task.train(was_training)
    return aggregate, saved
