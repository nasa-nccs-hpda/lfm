"""Semantic segmentation plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from lfm.full_model.all_tasks.utils.common import (
    _display_sample_key,
    _extract_image_and_mask,
    _extract_logits,
    _model_color,
    _model_display_name,
    _move_batch_to_device,
)
from lfm.full_model.all_tasks.utils.display import (
    create_colored_overlay_image,
    create_overlay_image,
    prepare_image_for_display,
)
from lfm.full_model.all_tasks.utils.metrics import calculate_f1_score
from lfm.full_model.all_tasks.utils.prediction_cache import _load_prediction_cache


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
        axes[0, col].set_title(
            f"{_display_sample_key(sample_key)}\n{display_note}", fontsize=10
        )
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
