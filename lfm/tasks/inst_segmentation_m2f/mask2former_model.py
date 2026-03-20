"""
DINOv3-Mask2Former Large Model Implementation

This module provides a complete DINOv3-Mask2Former model with custom backbone replacement.
The model uses DINOv3-Large as the backbone and Mask2Former as the segmentation head.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoModel, AutoModelForUniversalSegmentation
import logging

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    """
    Adapter module to convert DINOv3 features to expected channels for Mask2Former head.
    """

    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_ch, kernel_size=1)
                for out_ch in out_channels
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projections[i](feat) for i, feat in enumerate(features)]


class DinoV3WithAdapterBackbone(nn.Module):
    """
    Custom backbone that combines DINOv3-Large with adapter layers for Mask2Former compatibility.
    """

    def __init__(
        self, model_name: str, out_channels: List[int], use_flexible=False
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.adapter = Adapter(self.model.config.hidden_size, out_channels)

        # Define output features for Mask2Former compatibility
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = {
            name: ch for name, ch in zip(self.out_features, out_channels)
        }
        self._out_feature_strides = {
            "stage_0": 8,
            "stage_1": 16,
            "stage_2": 32,
            "stage_3": 32,
        }

        # Layers to extract from DINOv3-Large (different depths for multi-scale features)
        self.layers_to_extract = [
            5,
            11,
            17,
            23,
        ]  # Adjusted for Large model (24 layers total)

        if use_flexible:
            print(f"Flexible embeddings will be applied to DinoV3 backbone...")
            self._apply_flexible_weights()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get DINOv3 outputs with all hidden states
        outputs = self.model(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states

        # Calculate spatial dimensions after patch embedding
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size

        # Extract features from different layers
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]  # Skip CLS token
            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = (
                layer_output[:, 1:, :]
                .permute(0, 2, 1)
                .reshape(
                    batch_size,
                    self.model.config.hidden_size,
                    patch_height,
                    patch_width,
                )
            )
            extracted_features.append(feature_map)

        # Apply adapter to convert channels
        adapted_features = self.adapter(extracted_features)

        # Return features with proper naming for Mask2Former
        return {
            name: feat
            for name, feat in zip(self.out_features, adapted_features)
        }

    def _apply_flexible_weights(self):
        """Weight modification for 5-band input (Blue, Green, Orange, Red, NIR)

        Band mapping:
        - Channel 0 (Blue) <- Blue weights from DINOv3
        - Channel 1 (Green) <- Green weights from DINOv3
        - Channel 2 (Orange) <- Mean of Red and Green weights
        - Channel 3 (Red) <- Red weights from DINOv3
        - Channel 4 (NIR) <- Red weights (spectrally closest)
        """
        print("Modifying input weights for Blue-Green-Orange-Red-NIR bands...")

        # Access the patch embedding
        patch_embed = self.model.embeddings.patch_embeddings

        with torch.no_grad():
            original_weights = (
                patch_embed.weight.data.clone()
            )  # Shape: (out_channels, 3, 16, 16)
            # original_weights channels: [0]=Red, [1]=Green, [2]=Blue

            # Create new weights for 5-band input
            new_weights = torch.zeros(
                original_weights.shape[0],
                5,  # 5 input bands
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

            red_weights = original_weights[:, 0, :, :]
            green_weights = original_weights[:, 1, :, :]
            blue_weights = original_weights[:, 2, :, :]

            # Correct mapping based on RGB order in original_weights
            new_weights[:, 0, :, :] = blue_weights
            new_weights[:, 1, :, :] = green_weights
            new_weights[:, 2, :, :] = 0.7 * red_weights + 0.3 * green_weights
            new_weights[:, 3, :, :] = red_weights
            new_weights[:, 4, :, :] = 0.95 * red_weights

            # Replace patch embedding weights
            patch_embed.weight.data = new_weights

        print(
            "Applied flexible embedding approach to match input bands. "
            "Band mapping: [Blue, Green, Orange, Red, NIR] -> "
            "[Blue_weights, Green_weights, Mean(Red_weights, Green_weights), "
            "Red_weights, Red_weights]."
        )


def create_mask2former_dinov3_model(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    expected_channels: List[int] = [
        192,
        384,
        768,
        1536,
    ],  # Swin-Large head configuration
    freeze_backbone: bool = True,
    hub_token: str = None,
    use_flexible: bool = False,
) -> AutoModelForUniversalSegmentation:
    """
    Create a complete DINOv3-Large-Mask2Former model with custom backbone.
    """
    # Fixed Mask2Former base model - using Large version
    mask2former_model_name = "facebook/mask2former-swin-large-coco-instance"

    logger.info("Creating DINOv3-Large-Mask2Former model...")
    logger.info(f"  - Mask2Former base: {mask2former_model_name}")
    logger.info(f"  - DINOv3 backbone: {dinov3_model_name}")
    logger.info(f"  - Expected channels: {expected_channels}")
    logger.info(f"  - Freeze backbone: {freeze_backbone}")

    # 1. Load the base Mask2Former-Large model
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )

    # 2. Create custom DINOv3-Large backbone with adapter
    custom_backbone = DinoV3WithAdapterBackbone(
        dinov3_model_name, expected_channels, use_flexible
    )

    # 3. Replace the backbone - USE THE CORRECT PATH!
    # The encoder is in the pixel_level_module, not at model.backbone
    model.model.pixel_level_module.encoder = custom_backbone  # ✅ FIXED!

    logger.info(f"Replaced pixel_level_module.encoder with DINOv3 backbone")

    # 4. Freeze DINOv3 weights if requested
    if freeze_backbone:
        for param in custom_backbone.model.parameters():
            param.requires_grad = False
        logger.info("DINOv3-Large backbone weights frozen.")
    else:
        logger.info("DINOv3-Large backbone weights remain trainable.")

    logger.info("Successfully created DINOv3-Large-Mask2Former model.")

    return model


def get_model_info(model: AutoModelForUniversalSegmentation) -> Dict:
    """
    Get information about the DINOv3-Large-Mask2Former model.

    Args:
        model: The DINOv3-Large-Mask2Former model

    Returns:
        Dictionary with model information
    """
    backbone = model.model.backbone

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    backbone_params = sum(p.numel() for p in backbone.model.parameters())
    frozen_params = sum(
        p.numel() for p in backbone.model.parameters() if not p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "frozen_parameters": frozen_params,
        "backbone_model": (
            backbone.model.config.name_or_path
            if hasattr(backbone.model.config, "name_or_path")
            else "DINOv3-Large"
        ),
        "output_channels": list(backbone._out_feature_channels.values()),
        "output_strides": list(backbone._out_feature_strides.values()),
    }
