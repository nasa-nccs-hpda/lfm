import math

from collections import OrderedDict
from functools import partial
from typing import Literal

import torch

from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from torch import nn
from torch.utils.checkpoint import checkpoint

from .input_adapter import LunarFourierInputAdapter
from .output_adapter import SpatialOutputAdapter
from .transformer_ls import AttentionLS
from .vqvae import VQ2
from .multimae_utils import Attention, Mlp


class Block(nn.Module):
    """Block with otptions: regular Attention or Long-short Attention."""
    def __init__(
        self,
        dim,
        num_heads: int = 8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        w=2,
        dp_rank=2,
        rpe=False,
        attention_type: Literal["normal", "long_short"] = "long_short",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attention_type == "long_short":
            self.attn = AttentionLS(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                qkv_bias=qkv_bias,
                w=w,
                dp_rank=dp_rank,
                nglo=0,
                rpe=rpe,
            )
        else:  # normal attention
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """ViT module based on Block. Can use chekpoints for layers."""
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads: int = 8,
        qkv_bias=False,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        window_size=2,
        dp_rank=2,
        norm_layer=None,
        checkpoint_layers: list[int] | None = None,
        rpe=False,
        attention_type: Literal["normal", "long_short"] = "long_short",
    ):
        """ViT initialization.

        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            qkv_bias (bool): Whether to add qkv_bias to attention modules
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim
            drop_rate (float): dropout rate
            attn_drop_rate (float): dropout rate for attention
            drop_path_rate (float): drop path rate
            window_size: window size for long/short attention
            dp_rank: dp rank for long/short attention
            norm_layer: (nn.Module): normalization layer
            dtype: data type for attention blocks
            checkpoint_layers: indicate which layers to use for checkpointing
            rpe: Use relative position encoding in attention blocks
            attention_type: Type of attention to use ("normal" or "long_short")
        """
        super().__init__()
        self.embed_dim = embed_dim
        self._checkpoint_layers = checkpoint_layers or []

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                w=window_size,
                dp_rank=dp_rank,
                rpe=rpe,
                attention_type=attention_type,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            if i in self._checkpoint_layers:
                tokens = checkpoint(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FourierVQMultiMAE(nn.Module):
    """Fourier Feature Multi-task Multi-modal Masked Autoencoder."""

    def __init__(
        self,
        config,
        domain_conf: dict,
        dim_tokens: int = 768,
        dtype=torch.bfloat16,
        domains: list[str] | None = None,
    ):
        """Initialize FourierVQMultiMAE module.

        Args:
            config: Configuration object containing training parameters.
            domain_conf: dict defining input and output adapters to instantiate.
            dim_tokens: embedding dimension
            dtype: dtype of the model.
            domains: list of domains to consider.
        """
        super().__init__()
        self.domain_conf = domain_conf
        self.dim_tokens = dim_tokens
        self.dtype = dtype
        self.mask_ratio = config.model.mask_ratio
        self.vq_dim = self.dim_tokens  # * (2**num_patch_merges)  # 768 * (2^2) = 3072
        self.domains = domains or ["vis", "uv", "dtm", "slope", "aspect"]
        self.configure_model(config=config, domain_conf=domain_conf)

    def configure_model(self, config, domain_conf):
        # Initialize input and output adapters

        input_adapters = {
            domain: domain_conf[domain]["input_adapter"](
                embed_dim=self.dim_tokens,
                stride_level=domain_conf[domain]["stride_level"],
                patch_size=domain_conf[domain]["patch_size"],
                image_size=domain_conf[domain]["image_size"],
            )
            for domain in self.domains
        }
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=self.dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)

        output_adapters = {
            domain: domain_conf[domain]["output_adapter"](
                stride_level=domain_conf[domain]["stride_level"],
                patch_size_full=domain_conf[domain]["patch_size"],
                dim_tokens=config.model.decoder_dim,
                depth=config.model.decoder_depth,
                num_heads=config.model.decoder_num_heads,
                use_task_queries=True,
                task=domain,
                image_size=domain_conf[domain]["image_size"],
                context_tasks=self.domains,
            )
            for domain in self.domains
        }
        for adapter in output_adapters.values():
            adapter.init(dim_tokens_enc=self.dim_tokens)
        self.output_adapters = nn.ModuleDict(output_adapters)

        # Initialize ViTEncoder
        self.encoder = ViT(
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            qkv_bias=config.model.qkv_bias,
            mlp_ratio=config.model.mlp_ratio,
            drop_rate=config.model.drop_rate,
            attn_drop_rate=config.model.attn_drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            window_size=config.model.window_size,
            dp_rank=config.model.dp_rank,
            checkpoint_layers=config.model.checkpoint_layers,
            rpe=config.model.rpe,
            attention_type=config.model.attention_type
            if hasattr(config.model, "attention_type")
            else "normal",
        )

        # weight Initialize methods
        self.apply(self._init_weights)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d) and ".proj" in name:
                # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                w = m.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if config.model.use_vq:
            self.vq = VQ2(
                dim=self.vq_dim,
                quant_type=config.model.vq_type,
                codebook_size=config.model.codebook_size,
                dtype=self.dtype,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {"global_tokens"}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"input_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"output_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def generate_input_info(self, input_task_tokens) -> dict:
        """ Generate dict with input info.

        Args:
            input_task_tokens: Dictionary. Keys are domains. Values are tensors for that domain. Shape (B, N, C).

        Returns:
            Dictionary with structure as follows (key: value)
                tasks: Dictionary. Keys are domains. They are in the same order as in
                `input_task_tokens`.
                    num_tokens: Number of tokens in input_task_tokens for this domain.
                    has_2d_posemb:
                    start_idx: Index in input_task_tokens where domain starts.
                    end_idx: Index in input_task_tokens where domain ends.
                image_size:
                num_task_tokens: Total number of tokens in `input_task_tokens` across all domains.
        """
        input_info = OrderedDict()
        i = 0
        input_info["tasks"] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                "num_tokens": num_tokens,
                "has_2d_posemb": True,  # TODO: Modify when adding non-2D tasks
                "start_idx": i,
                "end_idx": i + num_tokens,
                "patch_size": self.domain_conf[domain]["patch_size"],
                "image_size": self.domain_conf[domain]["image_size"],
            }
            i += num_tokens
            input_info["tasks"][domain] = d

        input_info["num_task_tokens"] = i
        return input_info

    def forward(self, x: dict[str, torch.Tensor] | torch.Tensor):
        """Forward pass through input adapters, transformer encoder and output adapters.

        Args:
            x: Input tensor or dictionary of tensors.

        Returns:
            Tuple `(preds, task_masks, vq_loss)`.
        """

        # Processing input modalities
        # Verify only the modalities that exist
        for d in self.domains:
            assert x[d].shape[1] == self.domain_conf[d]["channels"]
            assert x[d].ndim == 4

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor, mask_ratio=self.mask_ratio)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens)

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        B, N, _ = input_tokens.shape
        ids_keep = torch.arange(N, device=input_tokens.device).unsqueeze(0).expand(B, -1)
        ids_restore = torch.arange(N, device=input_tokens.device).unsqueeze(0).expand(B, -1)
        task_masks = dict()

        # Transformer forward pass
        encoder_tokens = self.encoder(input_tokens)

        # Apply vector quantization
        if hasattr(self, "vq"):
            encoder_tokens, vq_loss = self.vq(encoder_tokens)
        else:
            vq_loss = torch.tensor(0.0, device=encoder_tokens.device)

        # Output decoders
        if self.output_adapters is None:
            return encoder_tokens, task_masks, vq_loss, input_info, ids_keep, ids_restore

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
        }

        return preds, task_masks, vq_loss
