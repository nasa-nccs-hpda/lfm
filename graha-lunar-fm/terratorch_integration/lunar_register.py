"""Model registration with TerraTorch registry.

This module registers lunar model variants with TerraTorch's backbone registry,
making them discoverable by name in configuration files and CLI commands.

The registration follows TerraTorch's custom module pattern, allowing lunar-fm
models to be used within TerraTorch without porting the entire codebase.
"""

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from .lunar_backbone import LunarBackbone
from .fvqmultimae_backbone import FVQMultiMAEBackbone

@TERRATORCH_BACKBONE_REGISTRY.register
def lunarmind_v1_tiny(**kwargs):
    """Lunar TerraMind v1 Tiny model.

    Architecture (from MODEL_CONFIGS["tiny"]):
        - Encoder depth: 12 layers
        - Decoder depth: 4 layers
        - Model dimension: 192
        - Attention heads: 3
        - MLP ratio: 4.0
        - Parameters: ~8M

    Required kwargs:
        - cfg / backbone_cfg: path to the pretraining ``full_config.yaml``
          (ships as ``weights/backbone/full_config.yaml``).
        - modality_info_path / backbone_modality_info_path: path to the
          pretraining ``modality_info.yaml`` (ships as
          ``weights/modality_info.yaml``).

    See ``LunarBackbone`` for all other kwargs.
    """
    return LunarBackbone(variant="tiny", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def lunarmind_v1_base(**kwargs):
    """Lunar TerraMind v1 Base model.

    Architecture (from MODEL_CONFIGS["base"]):
        - Encoder depth: 12 layers
        - Decoder depth: 12 layers
        - Model dimension: 768
        - Attention heads: 12
        - MLP ratio: 4.0
        - Parameters: ~86M

    Same required kwargs as ``lunarmind_v1_tiny`` (``cfg`` +
    ``modality_info_path``). See ``LunarBackbone`` for full kwargs.
    """
    return LunarBackbone(variant="base", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def lunarmind_v1_large(**kwargs):
    """Lunar TerraMind v1 Large model.

    Architecture (from MODEL_CONFIGS["large"]):
        - Encoder depth: 24 layers
        - Decoder depth: 24 layers
        - Model dimension: 1024
        - Attention heads: 16
        - MLP ratio: 4.0
        - Parameters: ~307M

    Same required kwargs as ``lunarmind_v1_tiny`` (``cfg`` +
    ``modality_info_path``). See ``LunarBackbone`` for full kwargs.
    """
    return LunarBackbone(variant="large", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def lunar_fvqmultimae(**kwargs):
    """Fourier-VQ MultiMAE lunar model (from origin/old-main) as a TerraTorch backbone.

    Architecture (pretraining defaults matching nasa_team_config.yaml):
        - Encoder depth: 12 layers
        - Model dimension: 768
        - Attention heads: 8
        - Patch size: 8
        - Input image size: 256

    Supported modalities (with default channel counts):
        vis (5), uv (2), dtm (1), slope (1), aspect (2), nac (1).
        Any other modality must be declared via ``backbone_new_modalities``.

    Args:
        **kwargs: Forwarded to :class:`FVQMultiMAEBackbone`. Notable knobs:
            - ``modalities``            list of modality names in canonical order.
            - ``checkpoint_path``       path to a pretraining .pth file.
            - ``image_size``/``patch_size``  override the pretraining defaults.
            - ``new_modalities``        dict of new-modality specs, same shape as
                                        ``LunarBackbone.new_modalities``.
            - ``merge_method``          None / "mean" / "max" / "concat" / "dict".

    Example (YAML):

        model:
          model_args:
            backbone: lunar_fvqmultimae
            backbone_checkpoint_path: weights/fvqmultimae/nasa_team_ckpt.pth
            backbone_modalities: [vis, nac]
            backbone_new_modalities:
              nac: {num_channels: 1}
            backbone_merge_method: concat
    """
    return FVQMultiMAEBackbone(**kwargs)