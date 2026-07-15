"""Model registration with TerraTorch registry.

This module registers lunar model variants with TerraTorch's backbone registry,
making them discoverable by name in configuration files and CLI commands.

The registration follows TerraTorch's custom module pattern, allowing lunar-fm
models to be used within TerraTorch without porting the entire codebase.
"""

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from .lunar_backbone import LunarBackbone


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

    Supported modalities:
        - Raw images: vis, dtm, slope, aspect, uv, nac
        - Tokenized images: tok_vis, tok_dtm, tok_slope, tok_aspect, tok_uv, tok_nac
        - Sequences: metadata, crater_bboxes, static_maps

    Multi-modal aggregation:
        - Use merge_method parameter to control feature aggregation
        - Options: None (default), "mean", "max", "concat", "dict"

    Args:
        **kwargs: Arguments passed to LunarBackbone, including:
            - modalities (list): List of modality names to use
            - new_modalities (dict): Define new modalities not in MODALITY_INFO
            - merge_method (str): Multi-modal aggregation method
            - checkpoint_path (str): Path to pretrained checkpoint
            - cfg (DictConfig): Hydra config for training mode
            - encoder_embedding_params (dict): Per-modality embedding configuration

    Returns:
        LunarBackbone: Initialized tiny model instance

    Examples:
        >>> # Single modality
        >>> model = TERRATORCH_BACKBONE_REGISTRY.build(
        ...     'lunarmind_v1_tiny',
        ...     modalities=['vis'],
        ...     checkpoint_path='checkpoints/lunar_tiny.pth'
        ... )

        >>> # Add new modality
        >>> model = TERRATORCH_BACKBONE_REGISTRY.build(
        ...     'lunarmind_v1_tiny',
        ...     modalities=['vis', 'thermal'],
        ...     new_modalities={'thermal': {'type': 'image', 'num_channels': 1}},
        ...     checkpoint_path='checkpoints/lunar_tiny.pth'
        ... )
    """
    # All architecture parameters come from MODEL_CONFIGS["tiny"]
    # No need to specify them here - LunarBackbone loads them automatically
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

    Supported modalities:
        - Raw images: vis, dtm, slope, aspect, uv, nac
        - Tokenized images: tok_vis, tok_dtm, tok_slope, tok_aspect, tok_uv, tok_nac
        - Sequences: metadata, crater_bboxes, static_maps

    Multi-modal aggregation:
        - Use merge_method parameter to control feature aggregation
        - Options: None (default), "mean", "max", "concat", "dict"

    Args:
        **kwargs: Arguments passed to LunarBackbone (see lunarmind_v1_tiny for details)

    Returns:
        LunarBackbone: Initialized base model instance

    Examples:
        >>> # Multi-modality with dict merge (mixed resolutions)
        >>> model = TERRATORCH_BACKBONE_REGISTRY.build(
        ...     'lunarmind_v1_base',
        ...     modalities=['vis', 'dtm', 'tok_vis'],
        ...     merge_method='dict',
        ...     checkpoint_path='checkpoints/lunar_base.pth'
        ... )
    """
    # All architecture parameters come from MODEL_CONFIGS["base"]
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

    Supported modalities:
        - Raw images: vis, dtm, slope, aspect, uv, nac
        - Tokenized images: tok_vis, tok_dtm, tok_slope, tok_aspect, tok_uv, tok_nac
        - Sequences: metadata, crater_bboxes, static_maps

    Multi-modal aggregation:
        - Use merge_method parameter to control feature aggregation
        - Options: None (default), "mean", "max", "concat", "dict"

    Args:
        **kwargs: Arguments passed to LunarBackbone (see lunarmind_v1_tiny for details)

    Returns:
        LunarBackbone: Initialized large model instance

    Examples:
        >>> # Multi-modality with concat aggregation
        >>> model = TERRATORCH_BACKBONE_REGISTRY.build(
        ...     'lunarmind_v1_large',
        ...     modalities=['vis', 'dtm', 'slope', 'aspect'],
        ...     merge_method='concat',
        ...     checkpoint_path='checkpoints/lunar_large.pth'
        ... )
    """
    # All architecture parameters come from MODEL_CONFIGS["large"]
    return LunarBackbone(variant="large", **kwargs)
