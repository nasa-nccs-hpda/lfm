"""Shared low-level helpers for segmentation plotting and caches."""

from __future__ import annotations

from pathlib import Path

import torch


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
