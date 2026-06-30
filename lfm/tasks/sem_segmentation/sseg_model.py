"""
model.py
Simple DINO encoder with UNet decoder for segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = '/explore/nobackup/projects/lfm/model_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

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
        freeze_encoder=False,
        weight_assignments=None
    ):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size

        # Get embedding dimension from encoder
        self.embed_dim = encoder.embed_dim  # Should be 1024 for vitl16

        # UNet decoder
        self.decoder = UNetDecoder(self.embed_dim, num_classes)

        if weight_assignments is None:
            raise ValueError("Model didn't receive weight assignments.")
        else:
            self.weight_assignments = weight_assignments
            self.num_bands = len(weight_assignments)

        if self.num_bands > 3:
            self._apply_flexible_weights()

        self.freeze_encoder = freeze_encoder
        if  self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (only decoder will be trained).")
        else:
            print("Encoder unfrozen! Full model will be trained.")

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
        """
        Apply flexible weight modifications for multi-band input.

        Uses self.weight_assignments to dynamically map input bands to
        DINOv3's RGB pretrained weights based on spectral similarity.
        """

        print(f"Modifying input weights for {self.num_bands} bands...")

        # Access the patch embedding
        patch_embed = self.encoder.patch_embed.proj

        with torch.no_grad():
            original_weights = patch_embed.weight.data.clone()  # Shape: (out_channels, 3, H, W)
            # original_weights channels: [0]=Red, [1]=Green, [2]=Blue

            # Create new weights for multi-band input
            new_weights = torch.zeros(
                original_weights.shape[0],
                self.num_bands,
                original_weights.shape[2],
                original_weights.shape[3],
            ).to(original_weights.device)

            red_weights = original_weights[:, 0, :, :]
            green_weights = original_weights[:, 1, :, :]
            blue_weights = original_weights[:, 2, :, :]

            # Dynamically assign weights based on weight_assignments
            for i, assignment in enumerate(self.weight_assignments):
                if assignment == "blue":
                    new_weights[:, i, :, :] = blue_weights
                elif assignment == "green":
                    new_weights[:, i, :, :] = green_weights
                elif assignment == "red":
                    new_weights[:, i, :, :] = red_weights
                elif assignment == "0.95*red":
                    new_weights[:, i, :, :] = 0.95 * red_weights
                elif assignment == "0.7*red+0.3*green":
                    new_weights[:, i, :, :] = 0.7 * red_weights + 0.3 * green_weights
                else:
                    # Default fallback to red weights
                    print(f"Warning: Unknown weight assignment '{assignment}' for band {i}, using red weights")
                    new_weights[:, i, :, :] = red_weights

            # Replace patch embedding weights
            patch_embed.weight.data = new_weights

        print(f"Applied flexible embedding approach: {self.weight_assignments}")


def load_dinov3_encoder(
    weights_local_checkpoint=CKPT, device="cuda", model="dinov3_vitl16"
):
    if os.path.exists(weights_local_checkpoint):
        print(f"Loading model from {weights_local_checkpoint}")
        encoder = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",  # GitHub repo
            model=model,
            source="github",
            weights=weights_local_checkpoint,
            force_reload=True,
        ).to(device)
        print("Encoder loaded with pretrained weights.")
        return encoder
    else:
        raise Exception("DinoV3 local checkpoint not found. Exiting.")
