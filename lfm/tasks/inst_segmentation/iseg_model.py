"""
DINOv3-Mask2Former Large Model Implementation

This module provides a complete DINOv3-Mask2Former model with custom backbone replacement.
The model uses DINOv3-Large as the backbone and Mask2Former as the segmentation head.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from types import SimpleNamespace
from transformers import AutoModel, AutoModelForUniversalSegmentation
import logging
import os

logger = logging.getLogger(__name__)


def load_dinov3_encoder(
    weights_local_checkpoint, device="cuda", model="dinov3_vitl16"
):
    if os.path.exists(weights_local_checkpoint):
        print(f"Loading model from {weights_local_checkpoint}")
        encoder = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",  # GitHub repo
            model=model,
            source="github",
            weights=weights_local_checkpoint,
        ).to(device)
        print("Encoder loaded with pretrained weights.")
        return encoder
    else:
        raise Exception("DinoV3 local checkpoint not found. Exiting.")



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
    Custom backbone that combines torch.hub DINOv3-Large with adapter layers for Mask2Former compatibility.
    """

    def __init__(
        self,
        out_channels: List[int],
        num_bands: int = 3,
        encoder: nn.Module = None,
    ):
        """
        Args:
            out_channels: Output channels for each stage [192, 384, 768, 1536]
            num_bands: Number of input bands (3, 5, 7, or 8)
            encoder: Pre-loaded torch.hub DINOv3 encoder
        """
        super().__init__()

        if encoder is None:
            raise ValueError("encoder must be provided (load with torch.hub.load())")

        self.model = encoder

        # Get hidden size from torch.hub model
        hidden_size = self._get_hidden_size()

        self.adapter = Adapter(hidden_size, out_channels)

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

        # Layers to extract from DINOv3-Large (24 layers total, 0-indexed)
        self.layers_to_extract = [5, 11, 17, 23]

        if num_bands not in [3, 5, 7, 8, 12]:
            raise ValueError("Instance Segmentation expects 3, 5, 7, 8, or 12 bands.")
        self.num_bands = num_bands

        if num_bands > 3:
            self._apply_flexible_weights(self.num_bands)

    def _get_hidden_size(self) -> int:
        """Extract hidden size from torch.hub DINOv3 model."""
        if hasattr(self.model, 'norm') and hasattr(self.model.norm, 'normalized_shape'):
            # DinoVisionTransformer has LayerNorm with normalized_shape
            return self.model.norm.normalized_shape[0]
        elif hasattr(self.model, 'patch_embed'):
            # Alternative: get from patch embedding output channels
            return self.model.patch_embed.proj.out_channels
        else:
            raise ValueError(
                "Unable to determine hidden size from model. "
                "Expected torch.hub DINOv3 model with .norm or .patch_embed"
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through DINOv3 backbone with adapter.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            SimpleNamespace with feature_maps and hidden_states
        """
        # Use torch.hub DINOv3's built-in method to extract intermediate layers
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.layers_to_extract,  # Which layers to extract
            return_class_token=False,   # Don't need CLS token
            reshape=True                # Automatically reshape to (B, C, H, W)
        )

        # outputs is a list of tensors, already in (B, C, H, W) format
        # Apply adapter to convert channels to expected dimensions
        adapted_features = self.adapter(outputs)

        # Return in the format Mask2Former expects
        from types import SimpleNamespace
        return SimpleNamespace(
            feature_maps=adapted_features,
            hidden_states=tuple(adapted_features),
        )

    def _apply_flexible_weights(self, num_bands=5):
        """Weight modification for >3-band input

        Band mapping:
        - Channel 0 (Blue) <- Blue weights from DINOv3
        - Channel 1 (Green) <- Green weights from DINOv3
        - Channel 2 (Orange) <- Mean of Red and Green weights
        - Channel 3 (Red) <- Red weights from DINOv3
        - Channel 4 (NIR) <- Red weights (spectrally closest)
        - Channel 5 (UV 1) <- Blue weights (spectrally closest)
        - Channel 6 (UV 2) <- Blue weights (spectrally closest)
        - Channel 7 (STATIC 3) <- Red weights (spectrally closest)
        """

        print("Modifying input weights for > 3 bands...")

        # Access the patch embedding
        patch_embed = self.encoder.patch_embed.proj

        with torch.no_grad():
            original_weights = (
                patch_embed.weight.data.clone()
            )  # Shape: (out_channels, 3, 16, 16)
            # original_weights channels: [0]=Red, [1]=Green, [2]=Blue

            # Create new weights for >3-band input
            # new_weights shape: [:, n_bands, :, :]
            new_weights = torch.zeros(
                original_weights.shape[0],
                num_bands,  # 5/7/8 input bands
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

            red_weights = original_weights[:, 0, :, :]
            green_weights = original_weights[:, 1, :, :]
            blue_weights = original_weights[:, 2, :, :]

            # Correct mapping based on RGB order in original_weights
            new_weights[:, :5, :, :] = torch.stack([
                blue_weights,
                green_weights,
                0.7 * red_weights + 0.3 * green_weights,
                red_weights,
                0.95 * red_weights
            ], dim=1)
            if num_bands >= 7:  # add 2 additional weights for 7-band input
                new_weights[:, 5:7, :, :] = blue_weights.unsqueeze(1).expand(
                    -1, 2, -1, -1
                )
            if num_bands >= 8:
                # For STATIC data, just use red embeddings for all bands
                for idx in range(7, num_bands):
                    new_weights[:, 7:, :, :] = red_weights.unsqueeze(1).expand(
                        -1, num_bands - 7, -1, -1
                    )

            # Replace patch embedding weights
            patch_embed.weight.data = new_weights

        print(
            "Applied flexible embedding approach to match input bands. "
            f"Bands specified: {num_bands}"
        )

def create_mask2former_dinov3_model(
    encoder: nn.Module,
    expected_channels: List[int] = [192, 384, 768, 1536],
    freeze_backbone: bool = True,
    num_bands: int = 3,
    device: str = "cuda",
    hub_token: str = None,
) -> AutoModelForUniversalSegmentation:
    """
    Create a complete DINOv3-Large-Mask2Former model with torch.hub backbone.

    Args:
        encoder: Pre-loaded torch.hub DINOv3 encoder
        expected_channels: Output channels for each stage
        freeze_backbone: Whether to freeze DINOv3 weights
        num_bands: Number of input bands (3, 5, 7, or 8)
        device: Device to load model on
        hub_token: HuggingFace token (for Mask2Former)

    Returns:
        Complete Mask2Former model with DINOv3 backbone
    """
    # Fixed Mask2Former base model - using Large version
    mask2former_model_name = "facebook/mask2former-swin-large-coco-instance"

    logger.info("Creating DINOv3-Large-Mask2Former model...")
    logger.info(f"  - Mask2Former base: {mask2former_model_name}")
    logger.info(f"  - DINOv3 backbone: torch.hub (dinov3_vitl16)")
    logger.info(f"  - Expected channels: {expected_channels}")
    logger.info(f"  - Freeze backbone: {freeze_backbone}")
    logger.info(f"  - Number of bands: {num_bands}")

    # Label mapping for crater detection
    label2id = {"background": 0, "crater": 1}
    id2label = {v: k for k, v in label2id.items()}

    # 1. Load base Mask2Former-Large model
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )

    # 2. Create custom DINOv3-Large backbone with adapter
    custom_backbone = DinoV3WithAdapterBackbone(
        out_channels=expected_channels,
        num_bands=num_bands,
        encoder=encoder,
    )

    # 3. Replace the backbone with DinoV3
    model.model.pixel_level_module.encoder = custom_backbone

    # 4. Freeze DINOv3 weights if requested
    if freeze_backbone:
        for param in custom_backbone.model.parameters():
            param.requires_grad = False
        logger.info("✓ DINOv3-Large backbone weights frozen.")
    else:
        logger.info("✓ DINOv3-Large backbone weights remain trainable.")

    logger.info("✓ Successfully created DINOv3-Large-Mask2Former model.")

    return model.to(device)


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
