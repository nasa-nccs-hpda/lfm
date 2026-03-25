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
        self,
        encoder,
        num_classes=2,
        img_size=(304, 304),
        use_flexible=False,
        num_bands=3,
    ):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size

        # Get embedding dimension from encoder
        self.embed_dim = encoder.embed_dim  # Should be 1024 for vitl16

        # UNet decoder
        self.decoder = UNetDecoder(self.embed_dim, num_classes)

        if num_bands not in [3, 5, 7]:
            if num_bands > 3 and not use_flexible:
                raise ValueError(
                    "Flexible embeddings not specified for > 3 band input."
                )
            raise ValueError("Dino Segmentation expects 3, 5, or 7 bands.")

        self.num_bands = num_bands

        # Change weights if using flexible embeddings approach
        if use_flexible:
            self._apply_flexible_weights(self.num_bands)

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

        # Access the patch embedding
        patch_embed = self.encoder.patch_embed.proj

        with torch.no_grad():
            original_weights = (
                patch_embed.weight.data.clone()
            )  # Shape: (out_channels, 3, 16, 16)
            # original_weights channels: [0]=Red, [1]=Green, [2]=Blue

            # Create new weights for 5-band input
            new_weights = torch.zeros(
                original_weights.shape[0],
                num_bands,  # 5/7 input bands
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

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
