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

    Requires the pretraining config + modality info bundle that ships with
    each checkpoint (``weights/backbone/full_config.yaml`` and
    ``weights/modality_info.yaml`` by convention). Both must be passed to
    ``__init__`` — the wrapper raises ``ValueError`` otherwise.

    What the wrapper does:
    1. Schema-based embedding init via ``init_enc_dec_embeddings()``.
    2. Encoder-only forward pass, returning all encoder block outputs as a
       list of ``(B, N, D)`` tensors (multi-scale features for TerraTorch necks).
    3. Multi-modal feature aggregation controlled by ``merge_method``.
    4. Checkpoint loading with state-dict key mapping.

    **Multi-Modal Aggregation.** When multiple modalities are present,
    ``merge_method`` controls how their encoded features are combined AFTER
    the encoder forward pass:

    - ``None`` (default): concatenated tokens from all modalities ``(B, N_total, D)``.
    - ``"mean"``: average features across modalities ``(B, N_common, D)``.
    - ``"max"``: max-pool features across modalities ``(B, N_common, D)``.
    - ``"concat"``: concatenate features along feature dim ``(B, N_common, D*M)``.
    - ``"dict"``: per-modality dict ``{modality: (B, N, D)}``.

    For ``"mean" | "max" | "concat"``, all image modalities must have identical
    token counts (same spatial resolution and patch size); otherwise a
    ``ValueError`` is raised with guidance to use ``None`` or ``"dict"``.

    Args:
        variant: Model variant name (``"tiny"``, ``"base"``, ``"large"``).
        modalities: List of modality names (e.g. ``["vis", "dtm", "tok_vis"]``).
        cfg: Path to the pretraining ``full_config.yaml`` (or a loaded
            ``DictConfig``). **Required.**
        modality_info_path: Path to the pretraining ``modality_info.yaml``.
            **Required.**
        mlp_ratio: MLP expansion ratio (default 4.0).
        merge_method: Multi-modal aggregation method — see above.
        checkpoint_path: Optional path to a pretrained checkpoint.
        remove_register_tokens: Strip register tokens before the neck
            (default False).
        new_modalities: Optional dict defining new modalities not in
            ``MODALITY_INFO``. Format::

                {
                    "modality_name": {
                        "type": "image" | "tokenized" | "metadata",
                        "num_channels": 1,        # for image type
                        "data_range": [0.0, 1.0], # optional
                    }
                }

        num_register_tokens: Override register-token count (auto-detected
            from checkpoint when a ``checkpoint_path`` is given).
        patch_size: Override patch size for all image modalities.
        seed_new_modality_from: Optional ``{target: source}`` mapping to
            copy encoder-embedding weights from a pretrained modality into
            a freshly-added one (channel-inflated as needed).
        **kwargs: Additional args passed to ``TerraMind``.

    Raises:
        ValueError: If ``cfg`` or ``modality_info_path`` is missing, if
            ``merge_method`` is invalid, or if modalities have mismatched
            token counts under an aggregating ``merge_method``.
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
        modality_info_path: str | None = None,
        seed_new_modality_from: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__()
        # ``backbone_modality_info`` in TerraTorch configs becomes ``modality_info``
        # after the factory strips the ``backbone_`` prefix. Accept it as an alias
        # so it doesn't leak through **kwargs into TerraMind() later.
        if modality_info_path is None and "modality_info" in kwargs:
            modality_info_path = kwargs.pop("modality_info")
        if seed_new_modality_from is None and "backbone_seed_new_modality_from" in kwargs:
            seed_new_modality_from = kwargs.pop("backbone_seed_new_modality_from")

        # Require both cfg and modality_info_path. LunarBackbone is only sensible
        # when initialised with the pretraining config + modality info bundle
        # published alongside each checkpoint (weights/backbone/full_config.yaml
        # and weights/modality_info.yaml by convention). Running without either
        # relies on undocumented heuristic fallbacks that silently disagree with
        # what the checkpoint was trained on.
        if cfg is None:
            raise ValueError(
                "LunarBackbone requires `cfg` (path to the pretraining "
                "full_config.yaml, e.g. weights/backbone/full_config.yaml). "
                "In TerraTorch YAML configs pass it as `backbone_cfg:`."
            )
        if modality_info_path is None:
            raise ValueError(
                "LunarBackbone requires `modality_info_path` (path to the "
                "pretraining modality_info.yaml, e.g. weights/modality_info.yaml). "
                "In TerraTorch YAML configs pass it as `backbone_modality_info_path:`."
            )

        # Store metadata
        self.variant = variant
        self.modalities = modalities
        self._mlp_ratio = mlp_ratio
        self.merge_method = merge_method
        self.remove_register_tokens = remove_register_tokens
        self._new_modalities = new_modalities  # Store for checkpoint loading
        self._seed_new_modality_from: dict[str, str] = seed_new_modality_from or {}

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
            print(
                f"LunarBackbone: using modality_info.yaml from '{mi_path}' "
                f"({len(pretrained_modality_info)} modalities: "
                f"{list(pretrained_modality_info.keys())})"
            )
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
        valid_merge_methods = [None, "mean", "max", "concat", "dict", "masked_group_mean"]
        if merge_method not in valid_merge_methods:
            raise ValueError(f"Invalid merge_method: {merge_method}. Must be one of: {valid_merge_methods}")

        # Per-forward tracking of the modalities actually seen in the input dict
        # and their runtime token counts, both in canonical `self.modalities`
        # order. image-vs-seq is looked up on demand from modality_info rather
        # than mirrored into a parallel flag list.
        self._present_modalities: list[str] = []
        self._num_tokens_per_mod: list[int] = []
        # Per-forward per-sample availability of each image modality. Shape
        # (B, M_img) with M_img the count of image modalities in
        # ``_present_modalities`` in the same order as ``_unstack_modalities``
        # returns them. Populated whenever we can compute it (image modalities
        # only), and consumed by merge_method="masked_group_mean". A modality
        # is considered available for a sample if the raw input has any
        # non-zero value in its (C, H, W) tensor.
        self._present_mod_availability: torch.Tensor | None = None

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
        # AFTER pretrained weights land, copy source-modality embedding weights
        # into each mapped new-modality embedding. See `_seed_new_modality_embeddings`.
        if self._seed_new_modality_from:
            self._seed_new_modality_embeddings()

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

        # Compute per-sample availability of each image modality from the raw
        # inputs (any non-zero pixel => available). Only masked_group_mean
        # consumes this; skip the B×C×H×W reduce for every other merge_method.
        # Shape: (B, M_img) in `_present_modalities` order, image-only.
        self._present_mod_availability = None
        if self.merge_method == "masked_group_mean":
            mi_pre = self.model.modality_info if hasattr(self.model, "modality_info") else None
            avail_cols: list[torch.Tensor] = []
            for mod in self._present_modalities:
                tensor = x[mod]
                info = mi_pre[mod] if mi_pre is not None else {}
                if info.get("type") != "img":
                    continue
                reduce_dims = tuple(range(1, tensor.dim()))
                avail_cols.append((tensor.abs().amax(dim=reduce_dims) > 0).to(tensor.dtype))
            if avail_cols:
                self._present_mod_availability = torch.stack(avail_cols, dim=1)

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

        elif self.merge_method == "masked_group_mean":
            # Per-modality spatial mean over tokens, then per-sample average
            # across modalities that are actually present in that sample.
            # Prevents all-zero "missing" inputs from dragging the pooled
            # feature toward the modality embedder's bias.
            #
            # Output at each block: (B, 1, D) — a single token per sample so
            # downstream AggregateTokens(pooling='mean') is a no-op and the
            # ScalarHead reshape (B, D, N=1) collapses cleanly. Keeping N=1
            # rather than returning (B, D) avoids surprising a neck that
            # expects a token dim.
            if self._present_mod_availability is None:
                raise RuntimeError(
                    "merge_method='masked_group_mean' requires at least one "
                    "image modality in the input; none were present."
                )
            avail = self._present_mod_availability  # (B, M_img)
            out_masked: list[torch.Tensor] = []
            for x in out:
                stacked = self._unstack_modalities(x)  # (B, M_img, N, D)
                per_mod = stacked.mean(dim=2)  # (B, M_img, D) — spatial mean
                w = avail.to(per_mod.dtype).unsqueeze(-1)  # (B, M_img, 1)
                denom = w.sum(dim=1).clamp_min(1.0)  # (B, 1)
                pooled = (per_mod * w).sum(dim=1) / denom  # (B, D)
                out_masked.append(pooled.unsqueeze(1))  # (B, 1, D)
            out = out_masked

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

    # ---------------------------------------------------- embedding seeding

    @staticmethod
    def _inflate_linear_patch_weight(
        src_w: torch.Tensor, c_src: int, c_tgt: int, patch_size: int, dim_tokens: int,
    ) -> torch.Tensor:
        """timm-style channel inflation for ImageEncoderEmbedding's Linear proj.

        The linear projection flattens each patch as ``(patch_size**2 * C)`` in
        pixel-major, channel-inner order (see the ``rearrange`` in
        ``ImageEncoderEmbedding.forward``). Reshape to ``(dim_tokens, P*P, C)``,
        inflate the channel axis exactly like a Conv2d inflation, then flatten
        back to ``(dim_tokens, C_tgt * P * P)`` for ``nn.Linear.weight``.
        """
        w = src_w.detach().clone().float().reshape(dim_tokens, patch_size * patch_size, c_src)
        if c_tgt == c_src:
            out = w
        elif c_tgt == 1:
            out = w.sum(dim=2, keepdim=True)
        elif c_tgt > c_src:
            repeat = (c_tgt + c_src - 1) // c_src
            out = w.repeat(1, 1, repeat)[:, :, :c_tgt] * (c_src / c_tgt)
        else:  # 1 < c_tgt < c_src
            out = w[:, :, :c_tgt] * (c_src / c_tgt)
        return out.reshape(dim_tokens, c_tgt * patch_size * patch_size)

    def _seed_new_modality_embeddings(self) -> None:
        """Copy encoder-embedding weights from pretrained source modalities into new ones.

        Copies ``proj.weight`` (channel-inflated for the target's ``num_channels``)
        and ``pos_emb`` (only when shapes match). Skips ``mod_emb`` — the modality
        identifier must stay unique per modality to preserve the multi-modal
        attention pattern.
        """
        enc = self.model.encoder_embeddings
        for target, source in self._seed_new_modality_from.items():
            if target not in (self._new_modalities or {}):
                print(
                    f"LunarBackbone: seed_new_modality_from: skip {target!r} "
                    f"— not in new_modalities."
                )
                continue
            if source not in enc:
                raise KeyError(
                    f"seed_new_modality_from: source modality {source!r} not in "
                    f"encoder_embeddings ({list(enc.keys())})."
                )
            src, tgt = enc[source], enc[target]
            c_src = int(src.num_channels)
            c_tgt = int(tgt.num_channels)
            patch_h = int(src.patch_size[0])
            if int(tgt.patch_size[0]) != patch_h or int(tgt.patch_size[1]) != int(src.patch_size[1]):
                print(
                    f"LunarBackbone: seed_new_modality_from: patch_size mismatch for "
                    f"{target!r}<-{source!r} (src {tuple(src.patch_size)} vs tgt "
                    f"{tuple(tgt.patch_size)}) — skipping."
                )
                continue
            dim = int(src.dim_tokens)
            with torch.no_grad():
                tgt.proj.weight.copy_(
                    self._inflate_linear_patch_weight(
                        src.proj.weight, c_src=c_src, c_tgt=c_tgt,
                        patch_size=patch_h, dim_tokens=dim,
                    )
                )
                if hasattr(src, "pos_emb") and hasattr(tgt, "pos_emb"):
                    if src.pos_emb.shape == tgt.pos_emb.shape:
                        tgt.pos_emb.copy_(src.pos_emb.detach())
                    else:
                        print(
                            f"LunarBackbone: pos_emb shape mismatch for "
                            f"{target!r}<-{source!r} ({tuple(tgt.pos_emb.shape)} vs "
                            f"{tuple(src.pos_emb.shape)}) — keeping target's own buffer."
                        )
            print(
                f"LunarBackbone: seeded new-modality embedding {target!r} "
                f"from pretrained {source!r} (C:{c_src}->{c_tgt}, patch={patch_h})"
            )

    def get_num_params(self) -> int:
        """Get total number of parameters.

        Returns:
            Total number of trainable parameters

        Example:
            >>> num_params = backbone.get_num_params()
            >>> print(f"Model has {num_params:,} parameters")
        """
        return sum(p.numel() for p in self.parameters())
