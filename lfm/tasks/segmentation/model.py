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

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Progressive upsampling with skip connections
        # Starting from patch embeddings: (batch, in_channels, H, W)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, patches_h * patches_w, embedding_dim)
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

    def __init__(self, encoder, num_classes=2, img_size=(304, 304)):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size

        # Get embedding dimension from encoder
        self.embed_dim = encoder.embed_dim  # Should be 1024 for vitl16

        # UNet decoder
        self.decoder = UNetDecoder(self.embed_dim, num_classes)

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

def load_dinov3_encoder(weights_local_checkpoint, device, model='dinov3_vitl16'):
    if os.path.exists(weights_local_checkpoint):
        print(f'Loading model from {weights_local_checkpoint}')
        encoder = torch.hub.load(
            repo_or_dir='facebookresearch/dinov3',  # GitHub repo
            model=model,
            source='github',
            weights=weights_local_checkpoint
        ).to(device)
        print("Encoder loaded with pretrained weights.")
        return encoder
    else:
        try:
            encoder = torch.hub.load(
                repo_or_dir='facebookresearch/dinov3',  # GitHub repo
                model=model,
                source='github',
                weights=weights_URL
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
    
