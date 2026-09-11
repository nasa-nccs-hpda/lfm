"""Small inference adapters shared by Graha segmentation tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lfm.all_models.all_tasks.utils.common import _extract_logits


class GrahaLogitModel(nn.Module):
    """Expose a Graha semantic task as a plain image-to-logits model."""

    def __init__(self, task: nn.Module) -> None:
        super().__init__()
        self.task = task

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _extract_logits(self.task(image))


def _extract_instance_predictions(output: Any) -> list[dict[str, torch.Tensor]]:
    """Normalize TerraTorch instance output to one dictionary per image."""
    if hasattr(output, "output"):
        output = output.output
    if isinstance(output, dict):
        output = [output]
    if not isinstance(output, (list, tuple)):
        raise TypeError(
            "Unsupported Graha instance output type: "
            f"{type(output)}; expected a prediction list."
        )
    predictions = list(output)
    if not all(isinstance(prediction, dict) for prediction in predictions):
        raise TypeError("Every Graha instance prediction must be a dictionary.")
    return predictions


class GrahaInstanceModel(nn.Module):
    """Expose a Graha instance task as an image-to-prediction-list model."""

    def __init__(self, task: nn.Module) -> None:
        super().__init__()
        self.task = task

    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        output = self.task.predict_step({"image": image}, batch_idx=0)
        return _extract_instance_predictions(output)
