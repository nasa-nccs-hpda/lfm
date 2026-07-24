"""Shared nodata preprocessing policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoDataPolicy:
    ignore_in_loss: bool = False
    ignore_index: int = -1
    image_fill_value: float = 0.0

    def apply_to_image(self, image: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
        if not self.ignore_in_loss or not np.any(nodata_mask):
            return image
        image = image.copy()
        image[nodata_mask, :] = self.image_fill_value
        return image

    def apply_to_label(self, label: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
        if not self.ignore_in_loss or not np.any(nodata_mask):
            return label
        if nodata_mask.shape != label.shape:
            raise ValueError(
                f"Nodata mask and label shapes differ: "
                f"nodata={nodata_mask.shape}, label={label.shape}"
            )
        label = label.copy()
        label[nodata_mask] = int(self.ignore_index)
        return label
