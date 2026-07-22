"""DINOv3 backbone wired to TorchVision Mask R-CNN for instance segmentation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def apply_flexible_patch_weights(
    encoder: nn.Module, weight_assignments: Sequence[str]
) -> None:
    """Expand DINO's RGB patch embedding to the requested WAC band layout."""
    patch_embed = encoder.patch_embed.proj
    with torch.no_grad():
        original_weights = patch_embed.weight.data.clone()
        red_weights = original_weights[:, 0, :, :]
        green_weights = original_weights[:, 1, :, :]
        blue_weights = original_weights[:, 2, :, :]
        new_weights = torch.zeros(
            original_weights.shape[0],
            len(weight_assignments),
            original_weights.shape[2],
            original_weights.shape[3],
            device=original_weights.device,
            dtype=original_weights.dtype,
        )
        for i, assignment in enumerate(weight_assignments):
            if assignment == "blue":
                new_weights[:, i, :, :] = blue_weights
            elif assignment == "green":
                new_weights[:, i, :, :] = green_weights
            elif assignment == "red":
                new_weights[:, i, :, :] = red_weights
            elif assignment == "0.95*red":
                new_weights[:, i, :, :] = red_weights
            elif assignment == "0.7*red+0.3*green":
                new_weights[:, i, :, :] = 0.7 * red_weights + 0.3 * green_weights
            else:
                print(
                    f"Warning: Unknown weight assignment '{assignment}' for band {i}; using red weights.",
                    flush=True,
                )
                new_weights[:, i, :, :] = red_weights
        patch_embed.weight.data = new_weights
    print(
        f"Applied flexible DINO patch embedding: {list(weight_assignments)}", flush=True
    )


class DinoMaskRCNNBackbone(nn.Module):
    """Expose DINO intermediate layers as a small feature pyramid for Mask R-CNN."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        out_channels: int = 256,
        layers_to_extract: Sequence[int] = (5, 11, 17, 23),
        output_strides: Sequence[int] = (8, 16, 32, 64),
        weight_assignments: Sequence[str] | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layers_to_extract = list(layers_to_extract)
        self.output_strides = list(output_strides)
        hidden_size = self._hidden_size()
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(hidden_size, out_channels, kernel_size=1)
                for _ in self.layers_to_extract
            ]
        )
        self.out_channels = out_channels

        if weight_assignments is not None and len(weight_assignments) > 3:
            apply_flexible_patch_weights(self.encoder, weight_assignments)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("DINO Mask R-CNN encoder frozen.", flush=True)
        else:
            print("DINO Mask R-CNN encoder unfrozen.", flush=True)

    def _hidden_size(self) -> int:
        if hasattr(self.encoder, "norm") and hasattr(
            self.encoder.norm, "normalized_shape"
        ):
            return int(self.encoder.norm.normalized_shape[0])
        if hasattr(self.encoder, "patch_embed"):
            return int(self.encoder.patch_embed.proj.out_channels)
        raise ValueError("Could not infer DINO hidden size from encoder.")

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        input_h, input_w = x.shape[-2:]
        features = self.encoder.get_intermediate_layers(
            x,
            n=self.layers_to_extract,
            return_class_token=False,
            reshape=True,
        )
        pyramid = OrderedDict()
        for i, (feature, projection, stride) in enumerate(
            zip(features, self.projections, self.output_strides)
        ):
            projected = projection(feature)
            target_h = max(1, input_h // int(stride))
            target_w = max(1, input_w // int(stride))
            if projected.shape[-2:] != (target_h, target_w):
                projected = F.interpolate(
                    projected,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            pyramid[str(i)] = projected
        return pyramid


def create_dino_mask_rcnn_model(
    encoder: nn.Module,
    *,
    num_classes: int = 2,
    num_bands: int = 7,
    target_size: int = 256,
    weight_assignments: Sequence[str] | None = None,
    freeze_backbone: bool = False,
    anchor_sizes: Sequence[Sequence[int]] = ((8,), (16,), (32,), (64,)),
    anchor_aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0),
) -> MaskRCNN:
    """Create a DINOv3 + Mask R-CNN model for crater instance segmentation."""
    backbone = DinoMaskRCNNBackbone(
        encoder,
        weight_assignments=weight_assignments,
        freeze_encoder=freeze_backbone,
    )
    aspect_ratios = tuple(
        tuple(float(r) for r in anchor_aspect_ratios) for _ in anchor_sizes
    )
    anchor_generator = AnchorGenerator(
        sizes=tuple(tuple(int(s) for s in level) for level in anchor_sizes),
        aspect_ratios=aspect_ratios,
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14,
        sampling_ratio=2,
    )
    return MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        min_size=target_size,
        max_size=target_size,
        image_mean=[0.0] * int(num_bands),
        image_std=[1.0] * int(num_bands),
    )
