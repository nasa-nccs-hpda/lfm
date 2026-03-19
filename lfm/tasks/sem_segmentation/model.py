"""
model.py
Simple DINO encoder with UNet decoder for segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    """UNet-style decoder for segmentation."""

    def __init__(self, in_channels, num_classes, dropout_rate=0.3):
        """
        Args:
            in_channels (int): Input embedding dimension from encoder
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability
        """
        super().__init__()

        # Progressive upsampling
        self.up1 = self._make_up_block(in_channels, 512, dropout_rate)
        self.up2 = self._make_up_block(512, 256, dropout_rate)
        self.up3 = self._make_up_block(256, 128, dropout_rate)
        self.up4 = self._make_up_block(128, 64, dropout_rate)

        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_up_block(self, in_ch, out_ch, dropout_rate):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_ch),  # More stable than BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Add regularization
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, embed_dim) from encoder
        """
        batch_size, num_patches, embed_dim = x.shape

        # Calculate spatial dimensions
        patch_h = patch_w = int(num_patches**0.5)

        # Reshape to spatial: (batch, embed_dim, patch_h, patch_w)
        x = x.transpose(1, 2).reshape(batch_size, embed_dim, patch_h, patch_w)

        # Progressive upsampling
        x = self.up1(x)  # 2x upsampling
        x = self.up2(x)  # 4x upsampling
        x = self.up3(x)  # 8x upsampling
        x = self.up4(x)  # 16x upsampling

        # Final classification
        x = self.final(x)

        return x


class DINOSegmentation(nn.Module):
    """DINO encoder with UNet decoder for segmentation."""

    def __init__(
        self, encoder, num_classes=2, img_size=(304, 304), use_flexible=False
    ):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size

        # Get embedding dimension from encoder
        self.embed_dim = encoder.embed_dim  # Should be 1024 for vitl16

        # UNet decoder
        self.decoder = UNetDecoder(self.embed_dim, num_classes)

        # Change weights if using flexible embeddings approach
        if use_flexible:
            self._apply_flexible_weights()

    def forward(self, x):
        # Get patch embeddings from encoder
        features_dict = self.encoder.forward_features(x)
        patch_embeddings = features_dict[
            "x_norm_patchtokens"
        ]  # (batch, num_patches, embed_dim)

        # Decode to segmentation map
        logits = self.decoder(patch_embeddings)  # (batch, num_classes, H, W)

        # Interpolate to exact target size if needed
        if logits.shape[2:] != self.img_size:
            logits = F.interpolate(
                logits,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def save_parameters(self, filename):
        """Save model state (encoder + decoder)."""
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename):
        """Load model state (encoder + decoder)."""
        self.load_state_dict(torch.load(filename))

    def _apply_flexible_weights(self):
        """Weight modification for 4-band input (Blue, Green, Red, NIR)

        Band mapping:
        - Channel 0 (Blue) <- Red weights (closest RGB match)
        - Channel 1 (Green) <- Green weights
        - Channel 2 (Red) <- Red weights
        - Channel 3 (NIR) <- Red weights (spectral closest)
        """
        print("Modifying input weights for Blue-Green-Red-NIR bands...")

        # Access the patch embedding
        patch_embed = self.encoder.embeddings.patch_embeddings

        with torch.no_grad():
            original_weights = (
                patch_embed.weight.data.clone()
            )  # Shape: (out_channels, 3, 16, 16)

            # Create new weights for 4-band input
            # We need to expand from 3 channels to 4 channels
            new_weights = torch.zeros(
                original_weights.shape[0],
                5,  # 5 input bands instead of 3
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

            # Copy weights: Blue <- Red, Green <- Green, Red <- Red, NIR <- Red
            new_weights[:, 0, :, :] = original_weights[
                :, 0, :, :
            ]  # Blue <- Blue
            new_weights[:, 1, :, :] = original_weights[
                :, 1, :, :
            ]  # Green <- Green
            # new_weights[:, 1, :, :] = original_weights[:, 1, :, :]  # Orange <- Mean(Green, Red)
            new_weights[:, 2, :, :] = original_weights[
                :, 2, :, :
            ]  # Red <- Red
            new_weights[:, 3, :, :] = original_weights[
                :, 2, :, :
            ]  # NIR <- Red

            # Replace patch embedding weights
            patch_embed.weight.data = new_weights

        print(
            "Band mapping: [Blue, Green, Orange, Red, NIR] -> "
            "[Blue_weights, Green_weights, Mean(Red_weights, Green_weights), "
            "Red_weights, Red_weights]"
        )


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
        try:
            encoder = torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",  # GitHub repo
                model=model,
                source="github",
                weights=weights_URL,
            ).to(device)
            print("Encoder loaded with pretrained weights.")
            return encoder
        except Exception as e:
            if isinstance(e, HTTPError) and e.code == 403:
                raise RuntimeError(
                    "DINOv3 checkpoint download failed (HTTP 403). "
                    "Meta-hosted signed URLs need you to sign an agreement. "
                    "Please go to the following link and follow the directions"
                    "to obtain a new download URL for your own use: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/."
                ) from e
            raise
