import torch
import torch.nn.functional as F

from einops import rearrange
from timm.layers import to_2tuple, trunc_normal_

from .fourier_utils import GaussianFourierFeatureTransform
from .multimae_utils import build_2d_sincos_posemb


class LunarInputAdapter(torch.nn.Module):
    """Adapter that creates tokens from patches over the images.

    Args:
        num_channels: Number of input channels
        stride_level: Stride level compared to the full-sized image. E.g. 4 for 1/4th the size of the image.
        patch_size: Int or tuple of the patch size over the full image size.
            Patch size for smaller inputs will be computed accordingly.
        embed_dim: embedding dimension (number of out channels for Conv2D).
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings.
        learnable_pos_emb: Set to True to learn positional embeddings instead (ignored for sincos).
        image_size: Default image size. Used to initialize size of positional embeddings.
        drop_rate: dropout rate for positional embeddings.
        norm_layer: Whether to apply a LayerNorm at the end of forward function.
    """
    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size: int | tuple[int, int],
        embed_dim: int = 768,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: int | tuple[int, int] = 224,
        drop_rate: float = 0.0,
        norm_layer: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = to_2tuple(image_size)
        self.drop_rate = drop_rate
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.proj = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=(self.P_H, self.P_W),
            stride=(self.P_H, self.P_W),
        )

        self._generate_position_encoding()

        self.pos_drop = torch.nn.Identity() if drop_rate == 0 else torch.nn.Dropout(p=drop_rate)

        self.norm = torch.nn.LayerNorm(self.embed_dim, eps=1e-6) if norm_layer else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb"}

    def _generate_position_encoding(self):
        """Generates a positional encoding signal for the model."""

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.embed_dim)
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = torch.nn.Parameter(torch.zeros(1, self.embed_dim, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

    def _check_dims(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        assert (h % self.P_H == 0) and (w % self.P_W == 0), (
            f"Image sizes {h}x{w} must be divisible by patch sizes {self.P_H}x{self.P_W}"
        )

        return h, w

    def init(self, dim_tokens: int = 768):
        pass

    def forward(self, x: torch.Tensor):
        H, W = self._check_dims(x)
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Image to patches: [B, C, H, W] -> [B, (H*W), C]
        x_patch = self.proj(x)
        x_patch = rearrange(x_patch, "b d nh nw -> b (nh nw) d")

        # Interpolate positional embedding
        x_pos_emb = self.interpolate_pos_emb(N_H, N_W)

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        x = self.pos_drop(x)
        x = self.norm(x)
        return x

    def interpolate_pos_emb(self, size_h: int, size_w: int):
        x_pos_emb = F.interpolate(self.pos_emb, size=(size_h, size_w), mode="bicubic", align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, "b d nh nw -> b (nh nw) d")
        return x_pos_emb


class LunarFourierInputAdapter(LunarInputAdapter):
    """Adapter that creates tokens from patches over the images and applies GFFT.

    Args:
        num_channels: Number of input channels
        stride_level: Stride level compared to the full-sized image. E.g. 4 for 1/4th the size of the image.
        patch_size: Int or tuple of the patch size over the full image size.
            Patch size for smaller inputs will be computed accordingly.
        embed_dim: embedding dimension (number of out channels for Conv2D).
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings.
        learnable_pos_emb: Set to True to learn positional embeddings instead (ignored for sincos).
        image_size: Default image size. Used to initialize size of positional embeddings.
        drop_rate: dropout rate for positional embeddings.
        norm_layer: Whether to apply a LayerNorm at the end of forward function.
    """
    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size: int | tuple[int, int],
        embed_dim: int = 768,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: int | tuple[int, int] = 224,
        drop_rate: float = 0.0,
        norm_layer: bool = False,
    ) -> None:
        super().__init__(
            num_channels=num_channels,
            stride_level=stride_level,
            patch_size=patch_size,
            embed_dim=embed_dim,
            sincos_pos_emb=sincos_pos_emb,
            learnable_pos_emb=learnable_pos_emb,
            image_size=image_size,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
        )

        self.gft = GaussianFourierFeatureTransform(
            img_size=self.image_size,
            patch_size=(self.P_H, self.P_W),
            in_chans=self.num_channels,
            embed_dim=self.embed_dim,
        )

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5):
        """Creates tokens from the images.

        Args:
            x (torch.Tensor): input of shape (B, C, H, W)
            mask_ratio (float): ratio of tokens to mask

        Returns:
            Tensor of shape (B, total_num_patches, embed_dim)
        """
        H, W = self._check_dims(x)
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Image to patches: [B, C, H, W] -> [B, (H*W), C]
        x_patch = self.proj(x)
        x_vis = rearrange(x_patch, "b d nh nw -> b (nh nw) d")
        x_hid = self.gft(x)

        x, _ = self.mask_tokens(x_vis, x_hid, mask_ratio)

        # Interpolate positional embedding
        x_pos_emb = self.interpolate_pos_emb(N_H, N_W)

        # Add patches and positional embeddings
        x = x + x_pos_emb

        x = self.pos_drop(x)
        x = self.norm(x)

        return x

    @staticmethod
    def mask_tokens(x_vis: torch.Tensor, x_hid: torch.Tensor, mask_ratio: float):
        """Masks tokens from the images.

        Args:
            x_vis (torch.Tensor): visible tokens of shape (B, total_num_patches, embed_dim)
            x_hid (torch.Tensor): hidden tokens of shape (B, total_num_patches, embed_dim)
            mask_ratio (float): ratio of tokens to mask (use x_hid for masked tokens)

        Returns:
            torch.Tensor: combined tokens of shape (B, total_num_patches, embed_dim)
            torch.Tensor: mask indicating which tokens came from x_vis (1) or x_hid (0)
        """
        B, N, _ = x_vis.shape
        assert x_hid.shape == x_vis.shape, (
            f"x_vis and x_hid must have the same shape, got {x_vis.shape} and {x_hid.shape}"
        )

        # Calculate number of tokens to keep visible
        num_keep = int(N * (1 - mask_ratio))

        # Generate random indices for each sample in the batch
        noise = torch.rand(B, N, device=x_vis.device)  # (B, N)

        # Sort noise to get random permutation indices
        ids_shuffle = torch.argsort(noise, dim=1)  # (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N)

        # Keep the first num_keep tokens as visible, rest as masked
        ids_keep = ids_shuffle[:, :num_keep]  # (B, num_keep)
        ids_mask = ids_shuffle[:, num_keep:]  # (B, N - num_keep)

        # Create mask: 1 for visible tokens, 0 for masked tokens
        mask = torch.zeros(B, N, device=x_vis.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, True)  # Set visible tokens to True

        # Combine tokens: use x_vis for visible tokens, x_hid for masked tokens
        x_combined = torch.where(mask.unsqueeze(-1), x_vis, x_hid)

        return x_combined, mask.float()
