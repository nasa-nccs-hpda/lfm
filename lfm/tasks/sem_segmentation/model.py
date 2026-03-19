"""
model.py
Simple DINO encoder with UNet decoder for segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    """UNet-style decoder with skip connections."""

    def __init__(self, encoder_dims, num_classes):
        super().__init__()
        # encoder_dims: list of feature dimensions from encoder layers
        # e.g., [768, 768, 768, 768] for DINO-Base
        
        self.up1 = self._make_up_block(encoder_dims[-1], 512)
        self.up2 = self._make_up_block(512 + encoder_dims[-2], 256)  # +skip
        self.up3 = self._make_up_block(256 + encoder_dims[-3], 128)  # +skip
        self.up4 = self._make_up_block(128 + encoder_dims[-4], 64)   # +skip
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_ch),  # More stable than BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_features):
        # x: final output, skip_features: list of intermediate features
        batch_size, num_patches, embed_dim = x.shape
        patch_h = patch_w = int(num_patches**0.5)
        
        x = x.transpose(1, 2).reshape(batch_size, embed_dim, patch_h, patch_w)
        
        x = self.up1(x)
        x = torch.cat([x, skip_features[-2]], dim=1)  # Skip connection
        
        x = self.up2(x)
        x = torch.cat([x, skip_features[-3]], dim=1)
        
        x = self.up3(x)
        x = torch.cat([x, skip_features[-4]], dim=1)
        
        x = self.up4(x)
        
        return self.final(x)


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
    
