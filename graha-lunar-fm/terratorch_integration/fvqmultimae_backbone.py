"""TerraTorch-compatible wrapper for the Fourier-VQ MultiMAE lunar model.

Vendored source lives under ``terratorch_integration/fvqmultimae/`` — this module
is the thin adapter that makes it interchangeable with ``LunarBackbone``.

Design notes
------------
* Encoder-only. The vendored ``FourierVQMultiMAE`` always instantiates output
  adapters (they're used only in its reconstruction path); we build them at
  ``__init__`` time to keep the class happy, then delete them so they don't
  waste memory or leak into optimiser param groups.
* The wrapper exposes ``self.blocks``, ``self.model.encoder.blocks``, and
  ``self._new_modalities`` so the shared ``_LunarLLRDMixin`` picks up the ViT
  layers and any new-modality embedders without changes.
* Multi-modal merge methods mirror ``LunarBackbone`` (None / mean / max /
  concat / dict). Token layout is deterministic: modalities in ``self.modalities``
  order, each contributes ``(image_size / patch_size) ** 2`` tokens.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from functools import partial

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY  # noqa: F401  (kept for future use)

from .fvqmultimae import FourierVQMultiMAE, LunarInputAdapter, SpatialOutputAdapter


# Pretraining defaults, taken from the config published alongside the
# Fourier-VQ MultiMAE checkpoint (`nasa_team_config.yaml` in the same bundle as
# `weights/fvqmultimae/nasa_team_ckpt.pth`).
DEFAULT_CHANNELS: dict[str, int] = {
    "vis": 5,
    "uv": 2,
    "dtm": 1,
    "slope": 1,
    "aspect": 2,
    "nac": 1,
}

# nasa_team_config.yaml model.* — used when the caller doesn't override.
_DEFAULT_MODEL_KWARGS: dict[str, Any] = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 8,
    "qkv_bias": False,
    "mlp_ratio": 4.0,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "decoder_depth": 6,
    "decoder_dim": 768,
    "decoder_num_heads": 8,
    "window_size": 2,
    "dp_rank": 4,
    "checkpoint_layers": [],
    "rpe": False,
    "attention_type": "normal",
    "use_vq": False,
    "vq_type": "fsq",
    "codebook_size": 4096,
    "mask_ratio": 0.0,
}


def _build_config(**overrides: Any) -> SimpleNamespace:
    """Wrap the model kwargs in the ``config.model.*`` attribute-access shape
    expected by ``FourierVQMultiMAE.configure_model``."""
    merged = {**_DEFAULT_MODEL_KWARGS, **overrides}
    return SimpleNamespace(model=SimpleNamespace(**merged))


class FVQMultiMAEBackbone(nn.Module):
    """Fourier-VQ MultiMAE wrapped as a TerraTorch backbone."""

    _VALID_MERGE_METHODS = (None, "mean", "max", "concat", "dict")

    def __init__(
        self,
        modalities: list[str],
        checkpoint_path: str | None = None,
        image_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.0,
        use_vq: bool = False,
        drop_path_rate: float = 0.0,
        new_modalities: dict[str, dict[str, Any]] | None = None,
        merge_method: str | None = None,
        pretrained_image_size: int = 256,
        seed_new_modality_from: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        # ``image_size`` is the RUNTIME image size (what the datamodule
        # delivers); ``pretrained_image_size`` is the image size the
        # checkpoint was trained at, used to build the input adapter's
        # ``pos_emb`` buffer at a matching shape. At forward time the
        # adapter bicubic-interpolates the buffer to whatever spatial
        # resolution the input has (see ``LunarInputAdapter.forward``),
        # so runtime image_size can differ from pretrained_image_size
        # without a state-dict shape mismatch on ``pos_emb``.
        super().__init__()

        if merge_method not in self._VALID_MERGE_METHODS:
            raise ValueError(
                f"Invalid merge_method: {merge_method!r}. "
                f"Must be one of: {self._VALID_MERGE_METHODS}"
            )

        # `backbone_new_modalities` from TerraTorch configs strips its prefix
        # and shows up as `new_modalities`; also accept the underscore alias
        # the factory can produce.
        if new_modalities is None and "backbone_new_modalities" in kwargs:
            new_modalities = kwargs.pop("backbone_new_modalities")
        self._new_modalities: dict[str, dict[str, Any]] = new_modalities or {}
        # Terratorch prefix stripping produces both `seed_new_modality_from` and
        # `backbone_seed_new_modality_from` depending on the factory; accept both.
        if seed_new_modality_from is None and "backbone_seed_new_modality_from" in kwargs:
            seed_new_modality_from = kwargs.pop("backbone_seed_new_modality_from")
        self._seed_new_modality_from: dict[str, str] = seed_new_modality_from or {}

        self.modalities = list(modalities)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.pretrained_image_size = int(pretrained_image_size)
        self.embed_dim = int(embed_dim)
        self.merge_method = merge_method

        # Resolve per-modality channel/patch/size, filling from defaults +
        # optional new_modalities overrides. `num_channels` is the only field
        # that MUST be resolvable — the rest fall back to the top-level values.
        # ``image_size`` on the per-mod spec is treated as a RUNTIME hint
        # (used only for the pre-forward token-count table below); the
        # adapter's ``pos_emb`` buffer is always built at
        # ``pretrained_image_size`` so state-dict loading finds matching
        # shapes. Runtime spatial size is inferred from the actual input in
        # ``forward()`` via ``LunarInputAdapter.interpolate_pos_emb``.
        self._domain_channels: dict[str, int] = {}
        self._domain_image_size: dict[str, int] = {}
        self._domain_patch_size: dict[str, int] = {}
        for mod in self.modalities:
            spec = self._new_modalities.get(mod, {})
            if "num_channels" in spec:
                n_ch = int(spec["num_channels"])
            elif mod in DEFAULT_CHANNELS:
                n_ch = DEFAULT_CHANNELS[mod]
            else:
                raise ValueError(
                    f"Cannot resolve num_channels for modality {mod!r}. "
                    f"Pass it via backbone_new_modalities[{mod!r}]['num_channels'] "
                    f"or add {mod!r} to DEFAULT_CHANNELS."
                )
            self._domain_channels[mod] = n_ch
            self._domain_image_size[mod] = int(spec.get("image_size", self.image_size))
            self._domain_patch_size[mod] = int(spec.get("patch_size", self.patch_size))

        # Fallback per-modality token count from the runtime image size —
        # used by merge methods when the wrapper is called with tensors that
        # match this hint. When the actual runtime shape differs, forward()
        # overwrites this with per-forward counts derived from the adapter's
        # own output.
        self._num_tokens_per_mod = {
            mod: (self._domain_image_size[mod] // self._domain_patch_size[mod]) ** 2
            for mod in self.modalities
        }

        # domain_conf that FourierVQMultiMAE.configure_model wants. Note:
        # ``image_size`` here goes to LunarInputAdapter and defines the
        # pos_emb buffer shape — pin it to pretrained_image_size so the
        # buffer matches the checkpoint exactly. Runtime spatial resolution
        # is picked up per-forward by ``interpolate_pos_emb``.
        domain_conf: dict[str, dict[str, Any]] = {}
        for mod in self.modalities:
            n_ch = self._domain_channels[mod]
            ps = self._domain_patch_size[mod]
            im_pre = int(
                self._new_modalities.get(mod, {}).get(
                    "pretrained_image_size", self.pretrained_image_size
                )
            )
            domain_conf[mod] = {
                "channels": n_ch,
                "stride_level": 1,
                "input_adapter": partial(LunarInputAdapter, num_channels=n_ch),
                "output_adapter": partial(
                    SpatialOutputAdapter, num_channels=n_ch, use_xattn=True,
                ),
                "loss": None,
                "patch_size": ps,
                "image_size": im_pre,
            }

        cfg = _build_config(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            use_vq=use_vq,
            drop_path_rate=float(drop_path_rate),
        )

        self.model = FourierVQMultiMAE(
            config=cfg,
            domain_conf=domain_conf,
            dim_tokens=embed_dim,
            domains=self.modalities,
        )

        # Encoder-only usage: drop the output adapters so they don't inflate
        # parameter counts or land in the LLRD groups. The reconstruction path
        # (which is the only consumer of self.output_adapters) is unreachable
        # once we call the encoder directly in forward().
        self.model.output_adapters = None

        # Expose depth-1 shortcuts for the LLRD mixin. ``blocks`` matches the
        # ``_encoder_block_index`` heuristic and ``_num_encoder_blocks``' first
        # branch; keeping the FM's own ``self.model.encoder.blocks`` intact
        # for anyone that wants direct access.
        self.blocks = self.model.encoder.blocks

        self.out_channels = [embed_dim] * depth

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        # AFTER the pretrained weights land, copy source-modality adapter weights
        # into each mapped new-modality adapter. See `_seed_new_modality_adapters`
        # for the tensors copied and the channel-inflation behaviour when the
        # source and target have a different `num_channels`.
        if self._seed_new_modality_from:
            self._seed_new_modality_adapters()

    # ------------------------------------------------------------------ forward

    def _unpack_modalities(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a channel-concatenated tensor back into a per-modality dict.

        Layout matches the ``output_mode="packed"`` convention used by
        ``LunarNACDTMDataset`` and ``LunarBackbone``: modalities in
        ``self.modalities`` order, each contributing its
        ``num_channels`` in that order.
        """
        expected = sum(self._domain_channels[m] for m in self.modalities)
        if packed.shape[1] != expected:
            raise ValueError(
                f"Packed tensor has {packed.shape[1]} channels but modalities "
                f"{self.modalities} expect {expected}."
            )
        out: dict[str, torch.Tensor] = {}
        cursor = 0
        for mod in self.modalities:
            n_ch = self._domain_channels[mod]
            out[mod] = packed[:, cursor : cursor + n_ch].contiguous()
            cursor += n_ch
        return out

    def _encode_per_modality(
        self, x: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run each modality's input adapter. Returns per-modality
        ``(B, N_mod, D)`` tensors in ``self.modalities`` order."""
        tokens: list[torch.Tensor] = []
        for mod in self.modalities:
            if mod not in x:
                raise ValueError(
                    f"Missing modality {mod!r} in forward input. "
                    f"Expected one of: {self.modalities}."
                )
            adapter = self.model.input_adapters[mod]
            # LunarInputAdapter.__call__(x) — the pretraining Fourier variant
            # takes an extra mask_ratio, but the plain LunarInputAdapter used
            # here (via domain_conf) does not. Keep it simple.
            tokens.append(adapter(x[mod]))
        return tokens

    def forward(
        self, x: dict[str, torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Encoder-only forward. Returns all encoder block outputs as a list.

        Args:
            x: Either a ``{modality: (B, C, H, W)}`` dict, a packed
               ``(B, C_total, H, W)`` tensor (channels concatenated in
               ``self.modalities`` order), or a dict with a single ``"image"``
               key holding the packed tensor.

        Returns:
            List of length ``depth``. Each element is a ``(B, N_total, D)``
            tensor, or a per-modality dict when ``merge_method='dict'``.
        """
        if isinstance(x, torch.Tensor) or (isinstance(x, dict) and set(x.keys()) == {"image"}):
            packed = x if isinstance(x, torch.Tensor) else x["image"]
            x = self._unpack_modalities(packed)

        # Validate modality keys.
        for mod in x.keys():
            if mod not in self.modalities:
                raise ValueError(
                    f"Unexpected modality: {mod!r}. Expected one of: {self.modalities}"
                )

        per_mod_tokens = self._encode_per_modality(x)
        # Refresh per-mod token count from the actual adapter outputs so
        # merge methods split correctly when the runtime image size differs
        # from the pretrained-size buffer (pos_emb is interpolated inside
        # LunarInputAdapter.forward).
        self._num_tokens_per_mod = {
            mod: int(tok.shape[1])
            for mod, tok in zip(self.modalities, per_mod_tokens)
        }
        tokens = torch.cat(per_mod_tokens, dim=1)  # (B, N_total, D)

        # Iterate encoder blocks manually so we can capture every layer's
        # output (mirrors LunarBackbone.forward).
        outs: list[torch.Tensor] = []
        for i, blk in enumerate(self.model.encoder.blocks):
            tokens = blk(tokens)
            outs.append(tokens.clone())

        # Apply final norm to the last output (matches ViT.forward semantics).
        outs[-1] = self.model.encoder.norm(outs[-1])

        if self.merge_method is not None:
            outs = self._apply_merge_method(outs)
        return outs

    # ---------------------------------------------------------- merge methods

    def _split_by_modality(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Split ``(B, N_total, D)`` back into a list of per-modality tensors."""
        sizes = [self._num_tokens_per_mod[m] for m in self.modalities]
        return list(torch.split(x, sizes, dim=1))

    def _stack_image_modalities(self, x: torch.Tensor) -> torch.Tensor:
        """Stack per-modality tensors along a new axis: ``(B, M, N_common, D)``.
        All modalities must have identical token counts.
        """
        counts = [self._num_tokens_per_mod[m] for m in self.modalities]
        if len(set(counts)) > 1:
            raise ValueError(
                f"Cannot aggregate modalities with different token counts: {counts}. "
                f"All modalities must share (image_size / patch_size). "
                f"Use merge_method=None or 'dict' for mixed-resolution inputs."
            )
        parts = self._split_by_modality(x)
        return torch.stack(parts, dim=1)

    def _apply_merge_method(
        self, out: list[torch.Tensor],
    ) -> list[torch.Tensor] | list[dict[str, torch.Tensor]]:
        if self.merge_method == "dict":
            result_dicts: list[dict[str, torch.Tensor]] = []
            for x in out:
                parts = self._split_by_modality(x)
                result_dicts.append(dict(zip(self.modalities, parts)))
            return result_dicts

        if self.merge_method == "mean":
            return [self._stack_image_modalities(x).mean(dim=1) for x in out]
        if self.merge_method == "max":
            return [self._stack_image_modalities(x).max(dim=1)[0] for x in out]
        if self.merge_method == "concat":
            return [
                torch.cat(list(self._stack_image_modalities(x).unbind(dim=1)), dim=-1)
                for x in out
            ]
        return out  # unreachable

    # ------------------------------------------------------------- checkpoint

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretraining checkpoint into the wrapped model.

        - Strips ``module.`` prefixes (DDP).
        - Filters out ``output_adapters.*`` from *unexpected* keys (we drop
          the reconstruction head, so these are expected).
        - Filters out ``input_adapters.<new_mod>.*`` from *missing* keys.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            if "state_dict" in blob:
                sd = blob["state_dict"]
            elif "model" in blob:
                sd = blob["model"]
            else:
                sd = blob
        else:
            sd = blob
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        # `output_adapters` was set to None post-init, so PyTorch will treat
        # every `output_adapters.*` key as unexpected. Drop them from the
        # incoming state dict so load_state_dict's unexpected list only
        # contains genuinely surprising keys.
        sd = {k: v for k, v in sd.items() if not k.startswith("output_adapters.")}

        result = self.model.load_state_dict(sd, strict=False)

        new_mod_names = list(self._new_modalities.keys())
        expected_missing = [
            k for k in result.missing_keys
            if any(k.startswith(f"input_adapters.{m}.") for m in new_mod_names)
        ]
        other_missing = [k for k in result.missing_keys if k not in expected_missing]

        # "Unexpected" keys fall into two buckets:
        #   1. input_adapters.<mod>.* where <mod> is a pretraining modality
        #      the user chose NOT to include in self.modalities. These are
        #      intentional drops — silence them.
        #   2. Anything else — surprising, log loudly.
        active_mods = set(self.modalities)
        expected_unexpected = [
            k for k in result.unexpected_keys
            if k.startswith("input_adapters.")
            and k.split(".", 2)[1] not in active_mods
        ]
        other_unexpected = [k for k in result.unexpected_keys if k not in expected_unexpected]

        if expected_missing:
            print(
                f"[FVQMultiMAEBackbone] new-modality embedders (fresh init): "
                f"{len(expected_missing)} keys under "
                f"{sorted({k.split('.')[1] for k in expected_missing})}"
            )
        if expected_unexpected:
            dropped_mods = sorted({k.split(".", 2)[1] for k in expected_unexpected})
            print(
                f"[FVQMultiMAEBackbone] pretraining modalities not used by this "
                f"task ({len(expected_unexpected)} ckpt keys dropped): {dropped_mods}"
            )
        if other_missing:
            print(
                f"[FVQMultiMAEBackbone] WARNING missing keys ({len(other_missing)}): "
                f"{other_missing[:8]}"
            )
        if other_unexpected:
            print(
                f"[FVQMultiMAEBackbone] WARNING unexpected keys "
                f"({len(other_unexpected)}): {other_unexpected[:8]}"
            )
        print(f"[FVQMultiMAEBackbone] loaded checkpoint from {ckpt_path}")

    # ---------------------------------------------------- adapter seeding

    @staticmethod
    def _inflate_conv_weight(src_w: torch.Tensor, target_c: int) -> torch.Tensor:
        """timm-style channel inflation for a (out, C_src, P, P) conv weight.

        - target_c == C_src  → exact copy.
        - target_c == 1      → sum along the channel axis (energy-preserving).
        - target_c  > C_src  → tile with (target_c / C_src) rescaling.
        - target_c  < C_src  → truncate then rescale to keep output magnitude.
        """
        w = src_w.detach().clone().float()
        c_src = w.shape[1]
        if target_c == c_src:
            return w
        if target_c == 1:
            return w.sum(dim=1, keepdim=True)
        if target_c > c_src:
            repeat = (target_c + c_src - 1) // c_src
            return w.repeat(1, repeat, 1, 1)[:, :target_c, :, :] * (c_src / target_c)
        # target_c < c_src (and > 1)
        return w[:, :target_c, :, :] * (c_src / target_c)

    def _seed_new_modality_adapters(self) -> None:
        """Copy adapter weights from pretrained source modalities into new ones.

        For each ``target: source`` pair in ``self._seed_new_modality_from``:
          * ``proj.weight`` — Conv2d weight, channel-inflated to the target's
            ``num_channels`` via :meth:`_inflate_conv_weight`.
          * ``proj.bias`` — copied verbatim (independent of input channel count).
          * ``pos_emb`` — buffer copied verbatim IFF the shapes match. If they
            differ (different patch_size / image_size configured per modality),
            the target keeps its freshly-built sincos buffer, which is already
            correct for its own grid.

        Silent if a target is not in ``self._new_modalities`` (nothing to seed).
        """
        adapters = self.model.input_adapters
        for target, source in self._seed_new_modality_from.items():
            if target not in self._new_modalities:
                print(
                    f"[FVQMultiMAEBackbone] seed_new_modality_from: skip {target!r} "
                    f"— not in new_modalities."
                )
                continue
            if source not in adapters:
                raise KeyError(
                    f"seed_new_modality_from: source modality {source!r} not "
                    f"in input_adapters ({list(adapters.keys())})."
                )
            src, tgt = adapters[source], adapters[target]
            with torch.no_grad():
                tgt.proj.weight.copy_(
                    self._inflate_conv_weight(src.proj.weight, tgt.num_channels)
                )
                if getattr(src.proj, "bias", None) is not None and getattr(tgt.proj, "bias", None) is not None:
                    tgt.proj.bias.copy_(src.proj.bias.detach())
                if hasattr(src, "pos_emb") and hasattr(tgt, "pos_emb"):
                    if src.pos_emb.shape == tgt.pos_emb.shape:
                        tgt.pos_emb.copy_(src.pos_emb.detach())
                    else:
                        print(
                            f"[FVQMultiMAEBackbone] pos_emb shape mismatch for "
                            f"{target!r}<-{source!r} ({tuple(tgt.pos_emb.shape)} vs "
                            f"{tuple(src.pos_emb.shape)}) — keeping the target's own buffer."
                        )
            print(
                f"[FVQMultiMAEBackbone] seeded new-modality adapter {target!r} "
                f"from pretrained {source!r} (C:{src.num_channels}->{tgt.num_channels})"
            )

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
