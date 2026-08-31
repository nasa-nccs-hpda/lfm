"""Small inference adapters for Graha semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn

from lfm.all_models.all_tasks.utils.common import _extract_logits


class GrahaLogitModel(nn.Module):
    """Expose a Graha task as a plain image-to-logits model for tiling."""

    def __init__(self, task: nn.Module) -> None:
        super().__init__()
        self.task = task

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _extract_logits(self.task(image))
