# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
# Diffusion-based UNet decoder: https://github.com/apple/ml-4m/blob/main/fourm/vq/models/unet/unet.py


import math
import torch
import torch.nn as nn


def pair(t) -> tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, groups=32):
    """Make a standard normalization layer.

    Args:
        channels: number of input channels.

    Return: an nn.Module for normalization.
    """
    return GroupNorm32(groups, channels)


class Upsample(nn.Module):
    """Nearest-neighbor upsample by x2 with optional conv to clean aliasing."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        use_conv: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                3,
                padding=1,
                groups=16,
                padding_mode=padding_mode,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


def icnr_init(
    weight: torch.Tensor, upsample_factor: int = 2, init=nn.init.kaiming_normal_
):
    """ICNR initialization for sub-pixel convolutions to reduce checkerboard artifacts.

    weight: [out_c, in_c, k, k] where out_c = out_ch * r^2
    """
    out_c, in_c, k1, k2 = weight.shape
    r = upsample_factor
    # out channels should be divisible by r^2
    assert out_c % (r * r) == 0, f"out_channels={out_c} not divisible by r^2={r*r}"
    new_out_c = out_c // (r * r)
    # initialize a smaller kernel then tile
    subkernel = torch.zeros(
        [new_out_c, in_c, k1, k2], device=weight.device, dtype=weight.dtype
    )
    init(subkernel)
    subkernel = subkernel.repeat_interleave(r * r, dim=0)
    with torch.no_grad():
        weight.copy_(subkernel)
    return weight


class PixelShuffleUpsample(nn.Module):
    """Conv (in_ch -> out_ch * r^2) -> PixelShuffle(r) -> optional 3x3 conv cleanup.
    Produces [B, out_ch, H*r, W*r].
    """

    def __init__(
        self, in_ch: int, out_ch: int, r: int = 2, padding_mode: str = "reflect"
    ):
        super().__init__()
        self.r = r
        self.expand = nn.Conv2d(
            in_ch, out_ch * (r * r), kernel_size=3, padding=1, padding_mode=padding_mode
        )
        # ICNR init to reduce checkerboard patterns
        icnr_init(self.expand.weight, upsample_factor=r)
        nn.init.zeros_(self.expand.bias)

        self.shuffle = nn.PixelShuffle(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.shuffle(x)
        return x


class ResBlock(nn.Module):
    """Simple residual block (no time/cond). Matches the structure used in diffusion UNet
    but without FiLM. Supports channel change via 1x1 (or 3x3) skip.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        use_conv_skip: bool = False,
        padding_mode: str = "zeros",
        norm_groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.channels = channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            normalization(channels, groups=norm_groups),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1, padding_mode=padding_mode),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels, groups=norm_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1, padding_mode=padding_mode
            ),
        )

        if out_channels == channels:
            self.skip = nn.Identity()
        else:
            self.skip = (
                nn.Conv2d(
                    channels, out_channels, 3, padding=1, padding_mode=padding_mode
                )
                if use_conv_skip
                else nn.Conv2d(channels, out_channels, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip(x) + h


class QKVAttention(nn.Module):
    """Same functional shape as in unet_diffusion: apply self-attention over flattened spatial tokens."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        # qkv: [B, (3 * H * C), T] where H = n_heads, C = head_dim, T = tokens
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)  # head_dim
        q, k, v = qkv.chunk(3, dim=1)  # each [B, H*C, T]
        scale = 1.0 / math.sqrt(math.sqrt(ch))
        q = (q * scale).view(bs * self.n_heads, ch, length)
        k = (k * scale).view(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)
        attn = torch.einsum("bct,bcs->bts", q, k)  # [B*H, T, T]
        attn = torch.softmax(attn.float(), dim=-1).type_as(attn)
        out = torch.einsum("bts,bcs->bct", attn, v)  # [B*H, C, T]
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Spatial self-attention (GroupNorm + 1x1 qkv + attention + 1x1 proj)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        norm_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels, groups=norm_groups)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attn = QKVAttention(num_heads)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.qkv(self.norm(x))
        h = self.attn(h)
        h = self.proj(h)
        return (x + h).reshape(b, c, *spatial)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: int = 16,
        model_channels: int = 256,
        channel_mult: tuple[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        attn_blocks: list | tuple[int] = (0,),
        num_heads: int = 1,
        padding_mode: str = "reflect",
        norm_groups: int = 32,
        **_kwargs,
    ):
        super().__init__()
        assert len(channel_mult) >= 1, "channel_mult must have at least one entry"
        if patch_size != 2 ** len(channel_mult):
            print(
                f"Output size is will not match image size. "
                f"2 ** channel_mult must match patch_size {patch_size} ({channel_mult})."
            )
        self.out_channels = out_channels
        self.channel_mult = tuple(int(m) for m in channel_mult)
        self.model_channels = int(model_channels)
        self.dropout = float(dropout)

        # Stem to receive post-quant proj features
        self.stem = nn.Conv2d(
            model_channels, model_channels, 3, padding=1, padding_mode=padding_mode
        )

        self.middle_block = nn.Sequential(
            ResBlock(
                model_channels,
                model_channels,
                dropout=self.dropout,
                norm_groups=norm_groups,
            ),
            AttentionBlock(
                model_channels, num_heads=num_heads, norm_groups=norm_groups
            ),
            ResBlock(
                model_channels,
                model_channels,
                dropout=self.dropout,
                norm_groups=norm_groups,
            ),
        )

        # Build upsampling blocks
        up_blocks = []
        in_ch = model_channels
        for i, mult in enumerate(channel_mult):
            # Each block: PixelShuffleUpscale -> ResBlock -> (optional Attn) -> ResBlock
            out_ch = model_channels // mult
            up_blocks.append(
                nn.Sequential(
                    PixelShuffleUpsample(in_ch, out_ch, padding_mode=padding_mode),
                    ResBlock(
                        out_ch,
                        out_ch,
                        dropout=self.dropout,
                        padding_mode=padding_mode,
                        norm_groups=norm_groups,
                    ),
                    (
                        AttentionBlock(
                            out_ch, num_heads=num_heads, norm_groups=norm_groups
                        )
                        if i in attn_blocks
                        else nn.Identity()
                    ),
                    ResBlock(
                        out_ch,
                        out_ch,
                        dropout=self.dropout,
                        padding_mode=padding_mode,
                        norm_groups=norm_groups,
                    ),
                )
            )
            in_ch = out_ch
        self.up_blocks = nn.Sequential(*up_blocks)

        self.out = nn.Sequential(
            normalization(out_ch, groups=norm_groups),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(out_ch, out_channels, 3, padding=1, padding_mode="reflect")
            ),
        )

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
          quant: [B, C_dec, H0, W0] with C_dec == model_channels

        Returns:
          dec: [B, out_channels, H0 * 2^L, W0 * 2^L]
        """
        x = self.stem(quant)
        x = self.middle_block(x)
        x = self.up_blocks(x)
        x = self.out(x)
        return x


def unet(**kwargs) -> UNetDecoder:
    return UNetDecoder(**kwargs)


def unet_small(**kwargs) -> UNetDecoder:
    return UNetDecoder(model_channels=192, attn_blocks=[], norm_groups=24, **kwargs)
