"""Plotting helpers for Graha/Lunar-FM segmentation notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
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
    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis

    overlay = img_rgb.copy()
    mask_bool = pred_mask == 1
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] = 1.0
    overlay[mask_bool, 2] = 0.0
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


def plot_validation_predictions(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    n_samples: int = 5,
    filename: str = "validation_predictions.png",
    display_method: str = "minmax",
    dpi: int = 300,
    setup_datamodule: bool = True,
) -> Path:
    """Save a 4-row prediction figure for the first validation samples."""
    plots_dir = Path(output_dir) / "plots"
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


class ValidationPlotCallback(Callback):
    """Save a lightweight validation prediction plot at the end of each epoch."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        n_samples: int = 5,
        every_n_epochs: int = 1,
        display_method: str = "minmax",
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
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
            display_method=self.display_method,
            dpi=self.dpi,
            setup_datamodule=False,
        )
