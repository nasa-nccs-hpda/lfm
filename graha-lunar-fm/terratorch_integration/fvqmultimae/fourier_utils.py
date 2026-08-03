from math import pi

import torch

from einops import rearrange
from timm.layers import to_2tuple


class GaussianFourierFeatureTransform(torch.nn.Module):
    """An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://github.com/tancik/fourier-feature-networks

    Given an input of size [B, C, H, W],
    returns a tensor of size [B, ST, D] where ST = T*H*W and D = mapping_size*2.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        scale: float = 10,
        use_scale_scheduler: bool = False,
        max_scale: float = 20,
        min_scale: float = 0,
        total_steps: int = 1000,
    ):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self._num_input_channels = in_chans
        self._mapping_size = embed_dim
        self._initial_scale = scale

        # Ensure embed_dim is even
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        fourier_embed_dim = embed_dim // 2

        # Calculate patch dimension
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans

        self._B = torch.randn((self.patch_dim, fourier_embed_dim)) * scale

        # Scale scheduling parameters
        self.use_scale_scheduler = use_scale_scheduler
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.total_steps = total_steps
        self.current_step = 0

        # If using scheduler, initialize with max_scale
        if use_scale_scheduler:
            self._current_scale = max_scale
            # Save the original B matrix (normalized)
            self._B_normalized = torch.randn((self.patch_dim, fourier_embed_dim))
        else:
            self._current_scale = scale

    def patchify(self, imgs: torch.Tensor):
        """Convert images to patches."""
        p1, p2 = self.patch_size
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)
        return x

    def forward(self, x: torch.Tensor):
        """Forward pass for 4D input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, ST, D) where ST = num_patches and D = embed_dim*2
        """
        if x.dim() != 4:
            raise ValueError(f'Expected 4D input [B, C, H, W] (got {x.dim()}D input)')

        c = x.shape[1]

        assert c == self._num_input_channels, f"Expected input to have {self._num_input_channels} channels (got {c} channels)"

        # Apply current scale if using scheduler
        if self.use_scale_scheduler:
            B_matrix = self._B_normalized * self._current_scale
        else:
            B_matrix = self._B

        # Patchify: (B, C, H, W) -> (B, num_patches, patch_dim)
        # where patch_dim = patch_size[0] * patch_size[1] * C
        x_patches = self.patchify(x)

        # Apply Fourier feature transform:
        # (B, num_patches, patch_dim) @ (patch_dim, embed_dim) -> (B, num_patches, embed_dim)
        x_fourier = x_patches @ B_matrix.to(x_patches.device, dtype=x_patches.dtype)

        # Apply 2π scaling and compute sin/cos
        x_fourier = 2 * pi * x_fourier

        # Concatenate sin and cos features along the last dimension
        # Result shape: (B, num_patches, embed_dim*2)
        fourier_features = torch.cat([torch.sin(x_fourier), torch.cos(x_fourier)], dim=-1)

        # Add positional encoding
        # fourier_features = fourier_features + self.pos_embed.to(fourier_features.device)

        return fourier_features

    def update_scale(self, step=None):
        """Update the scale factor based on the current step."""
        if not self.use_scale_scheduler:
            return

        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Linear scheduler from max_scale to min_scale
        progress = min(1.0, self.current_step / self.total_steps)
        self._current_scale = self.max_scale - progress * (self.max_scale - self.min_scale)

        return self._current_scale

    def get_current_scale(self):
        """Get the current scale value."""
        if self.use_scale_scheduler:
            return self._current_scale
        else:
            return self._initial_scale


# if __name__ == "__main__":
#     gft = GaussianFourierFeatureTransform(
#         img_size=224,
#         patch_size=16,
#         in_chans=13,
#         embed_dim=768,
#         scale=10,
#         use_scale_scheduler=True,
#         max_scale=20,
#         min_scale=0,
#         total_steps=1000,
#     )
#     x = torch.randn((8, 13, 1, 224, 224))
#     print(gft(x).shape)
