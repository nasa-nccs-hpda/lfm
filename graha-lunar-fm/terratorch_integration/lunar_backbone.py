"""TerraTorch-compatible wrapper for lunar TerraMind models.

This module provides the LunarBackbone class that wraps lunar-fm's TerraMind
architecture to conform to TerraTorch's backbone interface. It supports:
- Schema-based embedding initialization (post-refactoring)
- Three-tier configuration system
- Both training (cfg provided) and generation (cfg=None) modes
- Codebook fusion for multi-codebook tokenizers
"""

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from omegaconf import OmegaConf

from terramind.data.modality_info import (
    MODALITY_INFO,
    ModalityInfoImg,
    ModalityInfoImgTokenized,
    ModalityInfoSeq,
    compute_modality_id,
)
from terramind.models.terramind import MODEL_CONFIGS, TerraMind
from terramind.models.tm_utils import init_enc_dec_embeddings


class LunarBackbone(nn.Module):
    """TerraTorch-compatible wrapper for lunar TerraMind models.

    This wrapper enables lunar models to work within TerraTorch's framework by:
    1. Using schema-based embedding initialization via init_enc_dec_embeddings()
    2. Supporting both training mode (with cfg) and generation mode (cfg=None)
    3. Returning all encoder block outputs as a list (multi-scale features)
    4. Supporting multi-modal feature aggregation with strict validation
    5. Handling checkpoint loading with state dict key mapping

    The wrapper does NOT modify lunar-fm source code - it provides a thin
    adapter layer that bridges the interfaces.

    **Multi-Modal Aggregation:**

    When multiple modalities are present, the `merge_method` parameter controls
    how their encoded features are combined AFTER the encoder forward pass:

    - `None` (default): Return concatenated tokens from all modalities (B, N_total, D)
    - `"mean"`: Average features across modalities (B, N_common, D)
    - `"max"`: Max-pool features across modalities (B, N_common, D)
    - `"concat"`: Concatenate features along dimension (B, N_common, D*M)
    - `"dict"`: Return per-modality dict {modality: (B, N, D)}

    **Token Count Validation:**

    For aggregation methods ("mean", "max", "concat"), ALL image modalities must
    have identical token counts (same spatial resolution and patch size). If token
    counts differ, a ValueError is raised with guidance to use merge_method=None
    or "dict" for mixed-resolution inputs.

    **Future Enhancement Strategies (Not Yet Implemented):**

    The following strategies are documented for future work to handle modalities
    with different token counts:

    1. **Spatial Interpolation**: Interpolate all modalities to a common resolution
       (e.g., largest or specified target size) using bilinear interpolation

    2. **Learned Projection**: Use learnable linear layers to project modalities
       with different token counts to a common count

    3. **Attention-Based Fusion**: Use cross-attention to fuse modalities with
       different counts, where target tokens attend to all modality tokens

    4. **Padding/Masking**: Pad shorter sequences to max length with masking
       to ignore padded positions during aggregation

    See WP7 documentation for detailed implementation guidance on these strategies.

    Args:
        variant: Model variant name ("tiny", "small", "base", "large")
        modalities: List of modality names (e.g., ["vis", "tok_vis@100m_phaedra"])
        encoder_depth: Number of encoder layers (overrides variant default)
        decoder_depth: Number of decoder layers (overrides variant default)
        dim: Model embedding dimension (overrides variant default)
        num_heads: Number of attention heads (overrides variant default)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        merge_method: Multi-modal aggregation method (None, "mean", "max", "concat", "dict")
        checkpoint_path: Optional path to pretrained checkpoint
        cfg: Optional Hydra config for training mode (None for generation)
        remove_register_tokens: Whether to remove register tokens before neck (default: False)
        new_modalities: Optional dict defining new modalities not in MODALITY_INFO. Format:
            {
                "modality_name": {
                    "type": "image" | "tokenized" | "metadata",
                    "num_channels": 1,  # For image type
                    "data_range": [0.0, 1.0],  # Optional
                    # ... other modality-specific parameters
                }
            }
        **kwargs: Additional model arguments passed to TerraMind

    Raises:
        ValueError: If merge_method is invalid or modalities have mismatched token counts

    Examples:
        >>> # Single modality, no aggregation
        >>> backbone = LunarBackbone(
        ...     variant="tiny",
        ...     modalities=["vis"],
        ...     encoder_depth=12,
        ...     decoder_depth=4,
        ...     dim=192,
        ...     num_heads=3,
        ...     cfg=None
        ... )
        >>> x = {"vis": torch.randn(2, 5, 224, 224)}
        >>> outputs = backbone(x)  # List of 12 tensors, each (2, 196, 192)

        >>> # Multi-modality with mean aggregation (same resolution)
        >>> backbone_mean = LunarBackbone(
        ...     variant="tiny",
        ...     modalities=["vis", "dtm"],
        ...     encoder_depth=12,
        ...     decoder_depth=4,
        ...     dim=192,
        ...     num_heads=3,
        ...     merge_method="mean",
        ...     cfg=None
        ... )
        >>> x_multi = {
        ...     "vis": torch.randn(2, 5, 224, 224),  # 196 tokens
        ...     "dtm": torch.randn(2, 1, 224, 224)   # 196 tokens
        ... }
        >>> outputs = backbone_mean(x_multi)  # List of 12 tensors, each (2, 196, 192)

        >>> # Mixed resolutions - use dict merge method
        >>> backbone_dict = LunarBackbone(
        ...     variant="tiny",
        ...     modalities=["vis", "dtm"],
        ...     encoder_depth=12,
        ...     decoder_depth=4,
        ...     dim=192,
        ...     num_heads=3,
        ...     merge_method="dict",
        ...     cfg=None
        ... )
        >>> x_mixed = {
        ...     "vis": torch.randn(2, 5, 224, 224),  # 196 tokens
        ...     "dtm": torch.randn(2, 1, 112, 112)   # 49 tokens
        ... }
        >>> outputs = backbone_dict(x_mixed)  # List of 12 dicts
        >>> # outputs[0] = {"vis": (2, 196, 192), "dtm": (2, 49, 192)}

    Example (Training mode with config):
        >>> # Training mode with config
        >>> backbone = LunarBackbone(
        ...     variant="tiny",
        ...     modalities=["vis", "tok_vis@100m_phaedra"],
        ...     encoder_depth=12,
        ...     decoder_depth=4,
        ...     dim=192,
        ...     num_heads=3,
        ...     cfg=hydra_config
        ... )
        >>>
        >>> # Generation mode without config
        >>> backbone = LunarBackbone(
        ...     variant="tiny",
        ...     modalities=["vis"],
        ...     encoder_depth=12,
        ...     decoder_depth=4,
        ...     dim=192,
        ...     num_heads=3,
        ...     cfg=None
        ... )
    """

    def __init__(
        self,
        variant: str,
        modalities: list[str],
        mlp_ratio: float = 4.0,
        merge_method: str | None = None,
        checkpoint_path: str | None = None,
        cfg: object | None = None,
        remove_register_tokens: bool = False,
        new_modalities: dict | None = None,
        num_register_tokens: int | None = None,
        patch_size: int | None = None,
        modality_info: dict | None = None,  # ignored — built internally from `modalities`
        modality_info_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__()

        # Store metadata
        self.variant = variant
        self.modalities = modalities
        self._mlp_ratio = mlp_ratio
        self.merge_method = merge_method
        self.remove_register_tokens = remove_register_tokens
        self._new_modalities = new_modalities  # Store for checkpoint loading

        # Optional pretraining modality_info.yaml. When provided, entries in this
        # file are the source of truth for each requested modality (num_channels,
        # patch_size, codebook_size, id, stats, ...) and replace the heuristic
        # block below. Modalities not present in the yaml still fall back to
        # heuristics. cfg / new_modalities / patch_size overrides still win.
        pretrained_modality_info: dict = {}
        if modality_info_path is not None:
            mi_path = Path(modality_info_path)
            if not mi_path.exists():
                raise FileNotFoundError(
                    f"modality_info_path does not exist: {mi_path}"
                )
            loaded = OmegaConf.to_container(OmegaConf.load(str(mi_path)), resolve=True)
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"modality_info_path {mi_path} did not parse to a dict"
                )
            # save_modality_info() serialises encoder_embedding / decoder_embedding
            # as dotted-path strings (e.g. "terramind.models.encoder_embeddings.
            # ImageEncoderEmbedding"). Resolve them back to the actual classes so
            # _instantiate_embedding() can call them.
            for mod, info in loaded.items():
                if not isinstance(info, dict):
                    continue
                for key in ("encoder_embedding", "decoder_embedding"):
                    dotted = info.get(key)
                    if isinstance(dotted, str):
                        module_name, _, cls_name = dotted.rpartition(".")
                        if not module_name:
                            raise ValueError(
                                f"{mi_path}: {mod}.{key}={dotted!r} is not a "
                                f"dotted path"
                            )
                        info[key] = getattr(
                            importlib.import_module(module_name), cls_name,
                        )
            pretrained_modality_info = loaded
        self._pretrained_modality_info = pretrained_modality_info

        # Auto-detect num_register_tokens from checkpoint if not specified
        if num_register_tokens is None and checkpoint_path is not None:
            num_register_tokens = self._detect_num_register_tokens(checkpoint_path)
            print(f"Auto-detected num_register_tokens={num_register_tokens} from checkpoint")
        elif num_register_tokens is None:
            num_register_tokens = 0  # Default to no register tokens

        # Store for model creation
        self._num_register_tokens = num_register_tokens
        self._patch_size_override = patch_size

        # Accept cfg as either an OmegaConf object or a path to a YAML config file.
        # TerraTorch/jsonargparse may pass `backbone_cfg` through as a string path.
        if isinstance(cfg, (str, Path)):
            cfg = OmegaConf.load(str(cfg))

        # Validate merge_method
        valid_merge_methods = [None, "mean", "max", "concat", "dict"]
        if merge_method not in valid_merge_methods:
            raise ValueError(f"Invalid merge_method: {merge_method}. Must be one of: {valid_merge_methods}")

        # Per-forward tracking of the modalities actually seen in the input dict
        # and their runtime token counts, both in canonical `self.modalities`
        # order. image-vs-seq is looked up on demand from modality_info rather
        # than mirrored into a parallel flag list.
        self._present_modalities: list[str] = []
        self._num_tokens_per_mod: list[int] = []

        # Get base model config for this variant
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Available variants: {list(MODEL_CONFIGS.keys())}")

        model_config = MODEL_CONFIGS[variant].copy()

        # Merge additional kwargs
        model_config.update(kwargs)

        # Set num_register_tokens (auto-detected or provided)
        model_config["num_register_tokens"] = self._num_register_tokens

        # Register new modalities if provided
        if new_modalities:
            for mod_name, mod_config in new_modalities.items():
                if mod_name in MODALITY_INFO:
                    print(f"Warning: Modality '{mod_name}' already exists in MODALITY_INFO, skipping")
                    continue

                # Get modality type
                mod_type = mod_config.get("type")
                if mod_type not in ["image", "tokenized", "metadata"]:
                    raise ValueError(
                        f"Invalid modality type '{mod_type}' for '{mod_name}'. "
                        f"Must be one of: ['image', 'tokenized', 'metadata']",
                    )

                # Create base template based on type
                if mod_type == "image":
                    base_template = ModalityInfoImg()
                elif mod_type == "tokenized":
                    base_template = ModalityInfoImgTokenized()
                else:  # metadata
                    base_template = ModalityInfoSeq()

                # Convert to dict and merge with user parameters
                info_dict = asdict(base_template)
                # Remove "type" from mod_config to avoid overwriting
                user_params = {k: v for k, v in mod_config.items() if k != "type"}
                info_dict.update(user_params)

                # Register in MODALITY_INFO
                MODALITY_INFO[mod_name] = info_dict
                print(f"Registered new {mod_type} modality: {mod_name}")

        # Build per-run modality_info dict for each requested modality.
        # Priority order for each modality:
        #   1. Pretrained modality_info.yaml (if modality_info_path was set) —
        #      authoritative source of num_channels, patch_size, codebook_size,
        #      stats, id, ... as used at pretraining time.
        #   2. In-code MODALITY_INFO registry + heuristic runtime fallbacks
        #      (used when no yaml is provided or the modality is not in it).
        # cfg / patch_size overrides still win on top of both.
        modality_info: dict = {}
        for mod in modalities:
            if mod in self._pretrained_modality_info:
                info_dict = dict(self._pretrained_modality_info[mod])
                # Allow per-run patch_size override to still take effect.
                if self._patch_size_override is not None and info_dict.get("type") == "img":
                    info_dict["patch_size"] = self._patch_size_override
                modality_info[mod] = info_dict
                continue

            if mod not in MODALITY_INFO:
                raise ValueError(
                    f"Unknown modality: {mod}. "
                    f"Available modalities: {list(MODALITY_INFO.keys())}\n"
                    f"To add a new modality, use the 'new_modalities' parameter, "
                    f"or pass modality_info_path pointing at a pretrained "
                    f"modality_info.yaml that contains it.",
                )
            mod_info = MODALITY_INFO[mod]
            to_dict_fn = getattr(mod_info, "to_dict", None)
            info_dict: dict = cast(
                dict, to_dict_fn() if callable(to_dict_fn) else dict(mod_info),
            )

            # Runtime fallbacks - only used when no pretrained yaml entry exists.
            if info_dict.get("type") == "img":
                if not info_dict.get("pretokenized"):
                    if self._patch_size_override is not None:
                        info_dict["patch_size"] = self._patch_size_override
                    else:
                        info_dict.setdefault("patch_size", 16)
                    info_dict.setdefault("input_size", 256)
                    if "num_channels" not in info_dict:
                        if mod == "vis":
                            info_dict["num_channels"] = 5
                        elif mod == "uv":
                            info_dict["num_channels"] = 2
                        elif (mod == "aspect") | (mod == "aspect_3m"):
                            info_dict["num_channels"] = 2
                        else:
                            info_dict["num_channels"] = 1
                else:
                    info_dict.setdefault("codebook_size", 8192)
                    info_dict.setdefault("num_codebooks", 1)
                    if self._patch_size_override is not None:
                        info_dict["patch_size"] = self._patch_size_override
                    else:
                        info_dict.setdefault("patch_size", 16)
                    info_dict.setdefault("input_size", 256)
            elif info_dict.get("type") in ("seq", "seq_emb", "seq_token"):
                # Sequence modalities need a max_tokens hint so the "packed"
                # input layout (see forward()) can slice them back out.
                info_dict.setdefault("max_tokens", 10)

            # Add modality ID (required by TerraMind)
            info_dict["id"] = compute_modality_id(mod, info_dict)

            modality_info[mod] = info_dict

        # Initialize embeddings using schema-based approach
        # This handles the three-tier configuration hierarchy:
        # 1. Schema defaults (from embedding_schemas.py)
        # 2. MODALITY_INFO overrides
        # 3. Hydra config overrides (if cfg provided)
        encoder_embeddings, decoder_embeddings = init_enc_dec_embeddings(
            cfg=cfg,  # None for generation mode, full config for training
            modality_info=modality_info,
            in_domains=modalities,
            out_domains=modalities,  # Use same modalities for encoder and decoder
        )

        # Create TerraMind model
        self.model = TerraMind(
            encoder_embeddings=encoder_embeddings,
            decoder_embeddings=decoder_embeddings,
            modality_info=modality_info,
            **model_config,
        )

        # Set out_channels attribute (required by TerraTorch)
        # For multi-scale features, this should be a list of channel counts per scale
        # For lunar models, all encoder blocks output the same dimension
        self.out_channels = [model_config["dim"]] * model_config["encoder_depth"]

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x: dict[str, torch.Tensor] | torch.Tensor) -> list[torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Forward pass through encoder only.

        This method runs the encoder portion of the TerraMind model and returns
        all encoder block outputs as a list, compatible with TerraTorch's multi-scale
        feature extraction requirements.

        Args:
            x: Either a dict mapping modality names to tensors, or a single tensor
               - Dict format: {modality: (B, C, H, W)}
               - Tensor format: (B, C, H, W) - will be wrapped as {modality[0]: tensor}

        Returns:
            List of encoder block outputs, one per encoder layer:
                - If merge_method is None: Each element is (B, N, D) with concatenated tokens
                - If merge_method is "mean"/"max": Each element is (B, N_common, D) aggregated
                - If merge_method is "concat": Each element is (B, N_common, D*M) concatenated features
                - If merge_method is "dict": Each element is dict mapping modality to (B, N, D)

        Raises:
            ValueError: If input contains unexpected modalities or token counts don't match

        Example:
            >>> x = {"vis": torch.randn(2, 5, 224, 224)}
            >>> outputs = backbone(x)
            >>> print(len(outputs))  # 12 (for encoder_depth=12)
            >>> print(outputs[-1].shape)  # (2, N, 192)
        """
        # Handle packed input: a single (B, C, H, W) tensor produced by the
        # dataset's "packed" output mode. Image modalities occupy the first
        # sum(num_channels) channels concatenated in self.modalities order;
        # each sequence modality occupies one extra channel whose flattened
        # spatial dim holds its token IDs in positions [0:max_tokens]
        # (positions >= max_tokens are padded with the sentinel -1 and
        # ignored by the embedding because we cast back to long).
        if isinstance(x, torch.Tensor) or (isinstance(x, dict) and set(x.keys()) == {"image"}):
            packed = x if isinstance(x, torch.Tensor) else x["image"]
            x = self._unpack_modalities(packed)

        # Validate input modalities
        for mod in x.keys():
            if mod not in self.modalities:
                raise ValueError(f"Unexpected modality: {mod}. Expected one of: {self.modalities}")

        # Prepare modality dict in canonical `self.modalities` order so that
        # downstream splits (which zip against _present_modalities) do not
        # depend on the caller's dict-insertion order.
        mod_dict: dict[str, dict[str, torch.Tensor]] = {}
        self._present_modalities = [mod for mod in self.modalities if mod in x]

        for mod in self._present_modalities:
            emb_dict = self.model.encoder_embeddings[mod](x[mod])
            B, N = emb_dict["x"].shape[0], emb_dict["x"].shape[1]
            emb_dict["input_mask"] = torch.zeros(B, N, dtype=torch.bool, device=emb_dict["x"].device)
            mod_dict[mod] = emb_dict

        # `mod_mask` is the (B, N_total) int16 tensor pretraining uses to mark
        # each token's modality id. We derive per-mod token counts from it
        # rather than reading emb_dict shapes ourselves, so our bookkeeping is
        # sourced from the same tensor the FM code path uses.
        encoder_tokens, emb_all, encoder_mask, mod_mask = self.model.cat_encoder_tensors(mod_dict)
        self._num_tokens_per_mod = self._counts_from_mod_mask(mod_mask)

        # Add register tokens if present
        if self.model.num_register_tokens > 0 and hasattr(self.model, "register_tokens"):
            B = encoder_tokens.shape[0]
            register_tokens = self.model.register_tokens.expand(B, -1, -1)
            encoder_tokens = torch.cat([register_tokens, encoder_tokens], dim=1)
            # Embeddings for register tokens are zero so the sum below is a no-op for them
            emb_all = torch.cat([torch.zeros_like(register_tokens), emb_all], dim=1)
            # Register tokens are never masked
            register_mask = torch.zeros(
                B, self.model.num_register_tokens,
                dtype=torch.bool, device=encoder_tokens.device,
            )
            encoder_mask = torch.cat([register_mask, encoder_mask], dim=1)

        # Add modality/positional embeddings before the encoder blocks (matches terramind.forward)
        encoder_tokens = encoder_tokens + emb_all

        # Collect encoder block outputs (following terramind pattern)
        out = []
        for block in self.model.encoder:
            encoder_tokens = block(encoder_tokens, mask=None)
            out.append(encoder_tokens.clone())  # Clone to avoid in-place modifications

        # Apply final norm to last output only
        out[-1] = self.model.encoder_norm(out[-1])

        # Remove register tokens if requested (for TerraTorch necks that expect only patch tokens)
        if self.remove_register_tokens and self.model.num_register_tokens > 0:
            out = [x[:, self.model.num_register_tokens:].contiguous() for x in out]

        # Apply merge method if specified
        if self.merge_method is not None:
            out = self._apply_merge_method(out)

        # Return list of encoder block outputs (TerraTorch compatible)
        return out

    def _unpack_modalities(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a packed ``(B, C, H, W)`` tensor back into per-modality inputs.

        Layout (must match ``LunarNACDTMDataset(output_mode="packed")``):

        * Channels ``[0 : sum(num_channels_img_i)]`` hold the image modalities
          concatenated in ``self.modalities`` order, each contributing
          ``modality_info[mod]["num_channels"]`` channels.
        * Each sequence modality contributes exactly one additional channel.
          That channel, flattened to ``(B, H*W)``, holds the token IDs in the
          first ``max_tokens`` positions; remaining positions are padded with
          ``-1`` and discarded here.

        Args:
            packed: Stacked tensor with shape ``(B, C_total, H, W)``.

        Returns:
            Per-modality dict consumable by the existing forward logic.
        """
        mod_dict: dict[str, torch.Tensor] = {}
        cursor = 0
        expected = 0
        for mod in self.modalities:
            info = self.model.modality_info[mod]
            if info.get("type") == "img":
                expected += info["num_channels"]
            elif info.get("type") in ("seq", "seq_emb", "seq_token"):
                expected += 1
            else:
                raise ValueError(
                    f"Cannot unpack modality '{mod}' of unknown type "
                    f"'{info.get('type')}'"
                )
        if expected != packed.shape[1]:
            raise ValueError(
                f"Packed tensor has {packed.shape[1]} channels but modalities "
                f"{self.modalities} expect {expected} (one channel per image-modality "
                f"channel, plus one extra per sequence modality). Either align "
                f"backbone_modalities with the dataset's output_mode='packed' "
                f"layout, or enable the dataset's metadata padding so it emits "
                f"the missing sequence channels."
            )

        for mod in self.modalities:
            info = self.model.modality_info[mod]
            if info.get("type") == "img":
                n_ch = info["num_channels"]
                mod_dict[mod] = packed[:, cursor : cursor + n_ch, :, :].contiguous()
                cursor += n_ch
            else:  # sequence modality (validated above)
                max_tokens = info["max_tokens"]
                seq_channel = packed[:, cursor, :, :].reshape(packed.shape[0], -1)
                tokens = seq_channel[:, :max_tokens].to(torch.long)
                mod_dict[mod] = tokens
                cursor += 1
        return mod_dict

    def _counts_from_mod_mask(self, mod_mask: torch.Tensor) -> list[int]:
        """Derive per-modality token counts from cat_encoder_tensors' mod_mask.

        `mod_mask` is (B, N_total) int16 where each position holds the modality
        id (matches ``modality_info[mod]["id"]``). Tokens are contiguous per
        modality in the concat, so one ``unique_consecutive`` gives us all
        counts in encounter order — the same order as ``_present_modalities``.

        As a sanity check we compare the encountered ids against the ids we
        expect from ``_present_modalities``; a mismatch means our bookkeeping
        has drifted from the FM's mask (should never happen, but the check
        is cheap and localises the failure).
        """
        row = mod_mask[0]  # layout is identical across the batch
        uniq_ids, counts = torch.unique_consecutive(row, return_counts=True)
        mi = self.model.modality_info
        expected = [int(mi[mod]["id"]) for mod in self._present_modalities]
        actual = uniq_ids.tolist()
        if actual != expected:
            raise RuntimeError(
                f"mod_mask ids {actual} do not match expected order "
                f"{expected} for modalities {self._present_modalities}. "
                f"cat_encoder_tensors and _present_modalities have diverged.",
            )
        return counts.tolist()

    def _unstack_modalities(self, x: torch.Tensor) -> torch.Tensor:
        """Split concatenated tokens back into per-modality tensors.

        Args:
            x: Concatenated encoder tokens (B, N_total, D)

        Returns:
            Stacked modality tokens (B, M, N_common, D) where M is number of image modalities

        Raises:
            ValueError: If image modalities have different token counts
        """
        # image-vs-seq comes from modality_info, not a mirrored flag list.
        mi = self.model.modality_info
        is_img = [mi[mod].get("type") == "img" for mod in self._present_modalities]

        # Validate all image modalities have same token count BEFORE splitting
        image_token_counts = [n for n, img in zip(self._num_tokens_per_mod, is_img) if img]

        if len(image_token_counts) == 0:
            raise ValueError("No image modalities found for aggregation")

        if len(set(image_token_counts)) > 1:
            raise ValueError(
                f"Cannot aggregate modalities with different token counts: {image_token_counts}. "
                f"All image modalities must have the same spatial resolution and patch size. "
                f"Use merge_method=None or 'dict' for mixed-resolution inputs.",
            )

        # Remove register tokens if present
        if self.model.num_register_tokens > 0:
            x = x[:, self.model.num_register_tokens:]

        # Split by modality token counts, then keep image modalities only.
        x_split = torch.split(x, self._num_tokens_per_mod, dim=1)
        x_image = [m for m, img in zip(x_split, is_img) if img]

        # Stack: (B, M, N, D)
        return torch.stack(x_image, dim=1)

    def _apply_merge_method(self, out: list[torch.Tensor]) -> list[torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Apply merge method to aggregate multi-modal features.

        Args:
            out: List of encoder block outputs, each (B, N_total, D)

        Returns:
            Aggregated outputs based on merge_method:
                - "mean": List of (B, N_common, D) averaged across modalities
                - "max": List of (B, N_common, D) max-pooled across modalities
                - "concat": List of (B, N_common, D*M) concatenated features
                - "dict": List of dicts mapping modality to (B, N, D)
        """
        if self.merge_method == "mean":
            # Unstack and average across modalities
            out = [self._unstack_modalities(x) for x in out]
            out = [x.mean(dim=1) for x in out]

        elif self.merge_method == "max":
            # Unstack and max-pool across modalities
            out = [self._unstack_modalities(x) for x in out]
            out = [x.max(dim=1)[0] for x in out]

        elif self.merge_method == "concat":
            # Unstack and concatenate features
            out = [self._unstack_modalities(x) for x in out]
            out = [torch.cat(x.unbind(dim=1), dim=-1) for x in out]

        elif self.merge_method == "dict":
            # Return per-modality dict keyed by the modalities actually seen
            # this forward pass, in the canonical order they were consumed.
            out_dicts = []
            for x in out:
                if self.model.num_register_tokens > 0:
                    x = x[:, self.model.num_register_tokens:]
                x_split = torch.split(x, self._num_tokens_per_mod, dim=1)
                out_dicts.append(dict(zip(self._present_modalities, x_split)))

            return out_dicts

        return out

    def _detect_num_register_tokens(self, checkpoint_path: str) -> int:
        """Detect number of register tokens from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Number of register tokens (0 if not present in checkpoint)
        """
        try:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                return 0

            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Extract state dict
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Check if register_tokens exists in state dict
            # Remove "module." prefix if present
            state_dict_keys = [k.replace("module.", "") for k in state_dict.keys()]

            if "register_tokens" in state_dict_keys:
                # Get the shape to determine number of tokens
                register_tokens_key = [k for k in state_dict.keys() if k.replace("module.", "") == "register_tokens"][0]
                register_tokens_shape = state_dict[register_tokens_key].shape
                # Shape is (1, num_register_tokens, dim)
                return register_tokens_shape[1]
            else:
                # No register tokens in checkpoint
                return 0

        except Exception as e:
            print(f"Warning: Could not detect num_register_tokens from checkpoint: {e}")
            return 0

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained checkpoint.

        Supports loading lunar-fm checkpoint formats with automatic handling of:
        - Different checkpoint structures ("model", "state_dict", or raw dict)
        - DDP "module." prefix removal
        - Partial loading (strict=False) for missing/unexpected keys

        Args:
            checkpoint_path: Path to checkpoint file (.pth or .pt)

        Raises:
            FileNotFoundError: If checkpoint file doesn"t exist

        Example:
            >>> backbone.load_checkpoint("checkpoints/lunar_tiny.pth")
            Loaded checkpoint from checkpoints/lunar_tiny.pth
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load checkpoint
        # Note: weights_only=False is required for lunar-fm checkpoints that contain
        # OmegaConf DictConfig objects. Only use checkpoints from trusted sources.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove "module." prefix if present (from DDP training)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load into model (strict=False to allow partial loading)
        result = self.model.load_state_dict(state_dict, strict=False)

        # Report missing/unexpected keys (filter out expected new modality keys)
        if result.missing_keys:
            # Filter out new modality embeddings (expected to be missing)
            new_mod_keys = []
            if self._new_modalities:
                for mod_name in self._new_modalities.keys():
                    new_mod_keys.extend([k for k in result.missing_keys if mod_name in k])

            other_missing = [k for k in result.missing_keys if k not in new_mod_keys]

            if new_mod_keys:
                print(f"New modality embeddings (randomly initialized): {len(new_mod_keys)} keys")
            if other_missing:
                print(f"Warning: Missing keys in checkpoint: {other_missing[:5]}")
                for o in other_missing:
                    print(o)
                # if len(other_missing) > 5:
                #     print(f"  ... and {len(other_missing) - 5} more")

        if result.unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint N = {len(result.unexpected_keys)}")
            for r in result.unexpected_keys:
                print(r)
            # if len(result.unexpected_keys) > 5:
            #     print(f"  ... and {len(result.unexpected_keys) - 5} more")

        print(f"Loaded checkpoint from {ckpt_path}")

    def get_num_params(self) -> int:
        """Get total number of parameters.

        Returns:
            Total number of trainable parameters

        Example:
            >>> num_params = backbone.get_num_params()
            >>> print(f"Model has {num_params:,} parameters")
        """
        return sum(p.numel() for p in self.parameters())
