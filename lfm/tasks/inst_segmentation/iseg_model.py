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
import os

logger = logging.getLogger(__name__)


def load_dinov3_encoder(
    weights_local_checkpoint, device, model="dinov3_vitl16"
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
    Custom backbone that combines DINOv3-Large with adapter layers for Mask2Former compatibility.
    """

    def __init__(
        self,
        model_name: str,
        weights_local_checkpoint: str,
        out_channels: List[int],
        num_bands=3,
        device="cuda",
    ):
        super().__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        # if model_name == "facebook/dinov3-vitl16-pretrain-sat493m":
        #     self.model = load_dinov3_encoder(
        #         weights_local_checkpoint, device, model="dinov3_vitl16"
        #     )
        # else:
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except Exception:
            raise ValueError(
                f"Error loading model from model name. {model_name}"
                f"Ensure valid model name."
            )
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

        # Layers to extract from DINOv3-Large
        # Adjusted for Large model (24 layers total)
        self.layers_to_extract = [5, 11, 17, 23]

        if num_bands not in [3, 5, 7]:
            raise ValueError("Instance Segmentation expects 3, 5, or 7 bands.")
        self.num_bands = num_bands

        if num_bands > 3:
            self._apply_flexible_weights(self.num_bands)

    def forward(self, x: torch.Tensor):
        # Get DINOv3 outputs with all hidden states
        outputs = self.model(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states

        # Calculate spatial dimensions after patch embedding
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        num_patches = patch_height * patch_width

        # DEBUG: Print once
        if not hasattr(self, "_debug_printed"):
            layer_output = hidden_states[self.layers_to_extract[0] + 1]
            total_tokens = layer_output.shape[1]
            num_extra_tokens = total_tokens - num_patches
            print(f"\n🔍 DinoV3 Forward Pass Debug:")
            print(f"  Input shape: {x.shape}")
            print(f"  Patch size: {patch_size}")
            print(
                f"  Patch grid: {patch_height}×{patch_width} = {num_patches} patches"
            )
            print(f"  Total tokens: {total_tokens}")
            print(f"  Extra tokens (CLS+registers): {num_extra_tokens}")
            print(f"  Tokens to extract: {total_tokens - num_extra_tokens}")
            self._debug_printed = True

        # Extract features from different layers
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[
                layer_idx + 1
            ]  # Shape: [B, num_tokens, C]

            # DinoV3 adds: [CLS token, register tokens, patch tokens]
            # Calculate how many non-patch tokens there are
            total_tokens = layer_output.shape[1]
            num_extra_tokens = total_tokens - num_patches

            # Skip CLS + register tokens, keep only patch tokens
            patch_tokens = layer_output[
                :, num_extra_tokens:, :
            ]  # Shape: [B, num_patches, C]

            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(
                batch_size,
                self.model.config.hidden_size,
                patch_height,
                patch_width,
            )
            extracted_features.append(feature_map)

        # Apply adapter to convert channels
        adapted_features = self.adapter(extracted_features)

        # Return in the format Mask2Former expects: an object with .feature_maps attribute
        # Create a simple namespace object
        from types import SimpleNamespace

        return SimpleNamespace(
            feature_maps=adapted_features,  # List of tensors
            hidden_states=tuple(
                adapted_features
            ),  # Optional: same as feature_maps
        )

    def _apply_flexible_weights(self, num_bands=5):
        """Weight modification for 5-band input (Blue, Green, Orange, Red, NIR)

        Band mapping:
        - Channel 0 (Blue) <- Blue weights from DINOv3
        - Channel 1 (Green) <- Green weights from DINOv3
        - Channel 2 (Orange) <- Mean of Red and Green weights
        - Channel 3 (Red) <- Red weights from DINOv3
        - Channel 4 (NIR) <- Red weights (spectrally closest)
        - Channel 5 (UV 1) <- Blue weights (spectrally closest)
        - Channel 6 (UV 2) <- Blue weights (spectrally closest)
        """

        print("Modifying input weights for > 3 bands...")

        if num_bands not in [5, 7]:
            raise ValueError("Flexible embeddings expects 5 or 7 band input.")

        # Access the patch embedding - this IS the Conv2d layer
        patch_embed = self.model.embeddings.patch_embeddings

        with torch.no_grad():
            # Access weights directly from the Conv2d layer
            original_weights = (
                patch_embed.weight.data.clone()
            )  # Shape: (out_channels, 3, 16, 16)

            print(f"  Original weights shape: {original_weights.shape}")

            # Create new weights for 5-band input
            new_weights = torch.zeros(
                original_weights.shape[0],
                num_bands,  # 5/7 input bands
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

            # Extract RGB channel weights
            # DinoV3 typically uses RGB order: [R, G, B]
            red_weights = original_weights[:, 0, :, :]
            green_weights = original_weights[:, 1, :, :]
            blue_weights = original_weights[:, 2, :, :]

            # Correct mapping based on RGB order in original_weights
            # For 5 and 7-band input, these are constant
            new_weights[:, 0, :, :] = blue_weights
            new_weights[:, 1, :, :] = green_weights
            new_weights[:, 2, :, :] = 0.7 * red_weights + 0.3 * green_weights
            new_weights[:, 3, :, :] = red_weights
            new_weights[:, 4, :, :] = 0.95 * red_weights
            if num_bands == 7:  # add 2 additional weights for 7-band input
                new_weights[:, 5, :, :] = blue_weights
                new_weights[:, 6, :, :] = blue_weights

            # Replace patch embedding weights
            patch_embed.weight.data = new_weights

        print(
            "Applied flexible embedding approach to match input bands. "
            f"Bands specified: {num_bands}"
        )


def create_mask2former_dinov3_model(
    dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    dinov3_checkpoint: str = "",
    expected_channels=[192, 384, 768, 1536],
    freeze_backbone: bool = True,
    hub_token: str = None,
    num_bands: int = 3,
    device="cuda",
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

    # Required for mask2former model
    label2id = {
        "background": 0,
        "crater": 1,
    }
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
        dinov3_model_name,
        dinov3_checkpoint,
        expected_channels,
        num_bands,
        device,
    )

    # 3. Replace the backbone with DinoV3
    model.model.pixel_level_module.encoder = custom_backbone

    # 4. Freeze DINOv3 weights if requested
    if freeze_backbone:
        for param in custom_backbone.model.parameters():
            param.requires_grad = False
        logger.info("DINOv3-Large backbone weights frozen.")
    else:
        logger.info("DINOv3-Large backbone weights remain trainable.")

    logger.info("Successfully created DINOv3-Large-Mask2Former model.")

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
