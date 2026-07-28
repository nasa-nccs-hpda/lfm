"""Local TerraTorch registration for Toy DINO instance segmentation backbones."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

try:
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
except ImportError as exc:  # pragma: no cover - depends on HPC environment.
    TERRATORCH_BACKBONE_REGISTRY = None
    _TERRATORCH_IMPORT_ERROR = exc
else:
    _TERRATORCH_IMPORT_ERROR = None


def _pop_first(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _normalize_weight_assignments(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


class ToyDinoTerraTorchBackbone(nn.Module):
    """TerraTorch-facing wrapper around the existing Toy DINO feature pyramid.

    TorchVision's Mask R-CNN expects ``backbone.out_channels`` to be one integer.
    TerraTorch's object detection factory expects a list-like channel contract
    while assembling necks. This wrapper keeps the same forward behavior as the
    Toy DINO Mask R-CNN backbone but reports per-level channels for TerraTorch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_strides: list[int],
        *,
        return_format: str = "ordered_dict",
        feature_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_strides = list(output_strides)
        self.out_channels = [int(backbone.out_channels)] * len(self.output_strides)
        self.return_format = return_format
        self.feature_names = feature_names

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if self.return_format == "ordered_dict":
            if self.feature_names is not None:
                return OrderedDict(
                    (name, value)
                    for name, value in zip(self.feature_names, features.values())
                )
            return features
        if self.return_format == "list":
            return list(features.values())
        raise ValueError(f"Unsupported return_format: {self.return_format}")


def _build_toy_dino_mask_rcnn_backbone(**kwargs: Any) -> ToyDinoTerraTorchBackbone:
    from lfm.toy_model.inst_seg.dino_mask_rcnn_model import DinoMaskRCNNBackbone
    from lfm.toy_model.inst_seg.iseg_model import load_dinov3_encoder

    checkpoint_path = _pop_first(
        kwargs,
        "checkpoint_path",
        "weights_local_checkpoint",
        "dino_checkpoint",
        default=None,
    )
    device = torch.device(_pop_first(kwargs, "device", default="cpu"))
    weight_assignments = _normalize_weight_assignments(
        _pop_first(kwargs, "weight_assignments", default=None)
    )
    num_bands = int(_pop_first(kwargs, "num_bands", "in_channels", default=3))
    out_channels = int(_pop_first(kwargs, "out_channels", default=256))
    layers_to_extract = _pop_first(
        kwargs,
        "layers_to_extract",
        default=(5, 11, 17, 23),
    )
    output_strides = _pop_first(kwargs, "output_strides", default=(8, 16, 32, 64))
    freeze_encoder = bool(_pop_first(kwargs, "freeze_encoder", default=False))
    return_format = str(_pop_first(kwargs, "return_format", default="ordered_dict"))
    feature_names = _pop_first(kwargs, "feature_names", default=None)
    if feature_names is not None:
        feature_names = [str(name) for name in feature_names]

    if weight_assignments is None and num_bands != 3:
        raise ValueError(
            "Toy DINO TerraTorch registration needs weight_assignments when "
            f"num_bands={num_bands}. Pass backbone_weight_assignments through "
            "the TerraTorch model args."
        )

    encoder_kwargs: dict[str, Any] = {"device": device}
    if checkpoint_path is not None:
        encoder_kwargs["weights_local_checkpoint"] = str(checkpoint_path)
    encoder = load_dinov3_encoder(**encoder_kwargs)

    if kwargs:
        print(
            "[toy_dino_v3_mask_rcnn_backbone] ignored unsupported registry "
            f"kwargs: {sorted(kwargs)}",
            flush=True,
        )

    backbone = DinoMaskRCNNBackbone(
        encoder,
        out_channels=out_channels,
        layers_to_extract=layers_to_extract,
        output_strides=output_strides,
        weight_assignments=weight_assignments,
        freeze_encoder=freeze_encoder,
    )
    return ToyDinoTerraTorchBackbone(
        backbone,
        output_strides=list(output_strides),
        return_format=return_format,
        feature_names=feature_names,
    )


if TERRATORCH_BACKBONE_REGISTRY is not None:

    @TERRATORCH_BACKBONE_REGISTRY.register
    def toy_dino_v3_mask_rcnn_backbone(**kwargs: Any):
        """Build a Toy DINO backbone registered in TerraTorch's local registry."""
        return _build_toy_dino_mask_rcnn_backbone(**kwargs)


def require_terratorch_registry():
    """Return TerraTorch's backbone registry or raise the original import error."""
    if TERRATORCH_BACKBONE_REGISTRY is None:
        raise ImportError(
            "TerraTorch is required to use toy_dino_v3_mask_rcnn_backbone."
        ) from _TERRATORCH_IMPORT_ERROR
    return TERRATORCH_BACKBONE_REGISTRY
