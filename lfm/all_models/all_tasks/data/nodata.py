"""Shared nodata preprocessing policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NoDataPolicy:
    ignore_in_loss: bool = False
    ignore_index: int = -1
    image_fill_value: float = 0.0
    label_fill_value: int | None = None
    fill_image_nodata: bool = False
    excluded_values: tuple[float, ...] = ()
    image_nodata_policy: str = "union"

    def __post_init__(self) -> None:
        if self.image_nodata_policy not in {"union", "per_band", "preserve"}:
            raise ValueError(
                "image_nodata_policy must be one of "
                "{'union', 'per_band', 'preserve'}, got "
                f"{self.image_nodata_policy!r}."
            )

    @property
    def needs_per_band_image_mask(self) -> bool:
        return self.image_nodata_policy == "per_band"

    def apply_to_image(self, image: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
        if self.image_nodata_policy == "preserve":
            return image
        if not (self.fill_image_nodata or self.ignore_in_loss) or not np.any(
            nodata_mask
        ):
            return image
        image = image.copy()
        if nodata_mask.ndim == 2:
            image[nodata_mask, :] = self.image_fill_value
        elif nodata_mask.ndim == 3:
            if nodata_mask.shape != image.shape:
                raise ValueError(
                    f"Per-band nodata mask and image shapes differ: "
                    f"nodata={nodata_mask.shape}, image={image.shape}"
                )
            image[nodata_mask] = self.image_fill_value
        else:
            raise ValueError(
                f"Expected 2D or 3D nodata mask, got {nodata_mask.shape}"
            )
        return image

    def apply_to_label(self, label: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
        if self.label_fill_value is not None and np.issubdtype(
            label.dtype, np.floating
        ):
            label = np.nan_to_num(
                label,
                nan=float(self.label_fill_value),
            ).astype(label.dtype, copy=False)
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

    def apply_to_image_tensor(
        self,
        image: torch.Tensor,
        nodata_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.image_nodata_policy == "preserve":
            return image
        if not self.fill_image_nodata and not self.ignore_in_loss:
            return image
        image = torch.nan_to_num(image, nan=float(self.image_fill_value))
        if not torch.any(nodata_mask):
            return image
        image = image.clone()
        if nodata_mask.ndim == 2:
            image[:, nodata_mask] = float(self.image_fill_value)
        elif nodata_mask.ndim == 3:
            if tuple(nodata_mask.shape) != tuple(image.shape):
                raise ValueError(
                    f"Per-band nodata mask and image shapes differ: "
                    f"nodata={tuple(nodata_mask.shape)}, image={tuple(image.shape)}"
                )
            image[nodata_mask] = float(self.image_fill_value)
        else:
            raise ValueError(
                f"Expected 2D or 3D nodata mask, got {tuple(nodata_mask.shape)}"
            )
        return image

    def apply_to_mask_tensor(
        self,
        mask: torch.Tensor,
        nodata_mask: torch.Tensor,
        *,
        fill_label: bool = True,
        ignore_nodata: bool = True,
    ) -> torch.Tensor:
        if fill_label and self.label_fill_value is not None:
            mask = torch.nan_to_num(
                mask.float(),
                nan=float(self.label_fill_value),
            ).long()
        if not ignore_nodata or not self.ignore_in_loss or not torch.any(nodata_mask):
            return mask
        if tuple(nodata_mask.shape) != tuple(mask.shape):
            raise ValueError(
                f"Nodata mask and label shapes differ: "
                f"nodata={tuple(nodata_mask.shape)}, label={tuple(mask.shape)}"
            )
        mask = mask.clone()
        mask[nodata_mask] = int(self.ignore_index)
        return mask


def build_nodata_policy(
    *,
    no_data_replace: float | None = None,
    no_label_replace: int | None = None,
    ignore_nodata_in_loss: bool = False,
    nodata_ignore_index: int = -1,
    excluded_nodata_values: list[float] | tuple[float, ...] | None = None,
    image_nodata_policy: str = "union",
    nodata_policy: NoDataPolicy | None = None,
) -> NoDataPolicy:
    if nodata_policy is not None:
        return nodata_policy
    return NoDataPolicy(
        ignore_in_loss=ignore_nodata_in_loss,
        ignore_index=nodata_ignore_index,
        image_fill_value=(
            float(no_data_replace) if no_data_replace is not None else 0.0
        ),
        label_fill_value=no_label_replace,
        fill_image_nodata=no_data_replace is not None,
        excluded_values=tuple(float(value) for value in excluded_nodata_values or ()),
        image_nodata_policy=image_nodata_policy,
    )
