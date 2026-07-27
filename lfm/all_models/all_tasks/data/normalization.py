"""Composable normalization strategies for lunar image tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch

from lfm.full_model.all_tasks.utils import load_terramind_pretraining_stats


class NormalizationStrategy(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, *, band_filter: list[int]) -> np.ndarray:
        """Return a normalized HWC float image."""

    @abstractmethod
    def apply_tensor(
        self,
        image: torch.Tensor,
        *,
        band_filter: list[int] | None = None,
    ) -> torch.Tensor:
        """Return a normalized CHW image tensor."""


@dataclass(frozen=True)
class NoNormalization(NormalizationStrategy):
    def apply(self, image: np.ndarray, *, band_filter: list[int]) -> np.ndarray:
        return image

    def apply_tensor(
        self,
        image: torch.Tensor,
        *,
        band_filter: list[int] | None = None,
    ) -> torch.Tensor:
        return image


@dataclass(frozen=True)
class ZScoreNormalization(NormalizationStrategy):
    means: np.ndarray
    stds: np.ndarray

    def apply(self, image: np.ndarray, *, band_filter: list[int]) -> np.ndarray:
        means = np.asarray(self.means, dtype=np.float32)
        stds = np.asarray(self.stds, dtype=np.float32)
        if len(means) == image.shape[2]:
            mean_filtered = means
            std_filtered = stds
        else:
            mean_filtered = means[band_filter]
            std_filtered = stds[band_filter]
        if np.any(std_filtered <= 0):
            raise ValueError("All normalization std values must be positive.")
        return (image - mean_filtered.reshape(1, 1, -1)) / std_filtered.reshape(
            1, 1, -1
        )

    def apply_tensor(
        self,
        image: torch.Tensor,
        *,
        band_filter: list[int] | None = None,
    ) -> torch.Tensor:
        means = np.asarray(self.means, dtype=np.float32)
        stds = np.asarray(self.stds, dtype=np.float32)
        if len(means) == image.shape[0]:
            mean_filtered = means
            std_filtered = stds
        elif band_filter is not None:
            mean_filtered = means[band_filter]
            std_filtered = stds[band_filter]
        else:
            raise ValueError(
                f"Normalization stats have {len(means)} channel(s), "
                f"but image has {image.shape[0]} channel(s)."
            )

        mean = torch.tensor(mean_filtered, dtype=image.dtype, device=image.device).view(
            -1, 1, 1
        )
        std = torch.tensor(std_filtered, dtype=image.dtype, device=image.device).view(
            -1, 1, 1
        )
        if torch.any(std <= 0):
            raise ValueError("All normalization stds must be positive.")
        if mean.shape[0] == 1 and image.shape[0] != 1:
            warnings.warn(
                "Expanding single-channel norm stats to "
                f"{image.shape[0]} image channels. "
                "This is only appropriate when all channels share the same "
                "physical modality/range.",
                UserWarning,
                stacklevel=2,
            )
            mean = mean.expand(image.shape[0], -1, -1)
            std = std.expand(image.shape[0], -1, -1)
        elif mean.shape[0] != image.shape[0]:
            raise ValueError(
                f"Normalization stats have {mean.shape[0]} channel(s), "
                f"but image has {image.shape[0]} channel(s)."
            )
        return (image - mean) / std


@dataclass(frozen=True)
class FinetuneStatsNormalization(ZScoreNormalization):
    """Z-score normalization from train-split finetuning statistics."""


@dataclass(frozen=True)
class PretrainYamlNormalization(ZScoreNormalization):
    """Z-score normalization loaded from TerraMind pretraining modality YAML."""

    @classmethod
    def from_modality_info(
        cls,
        modality_info: str | Path,
        *,
        normalization_modality: str,
        band_filter: list[int] | None,
    ) -> "PretrainYamlNormalization":
        means, stds = load_terramind_pretraining_stats(
            Path(modality_info),
            normalization_modality=normalization_modality,
            band_filter=band_filter,
        )
        return cls(
            np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)
        )


def build_normalization_strategy(
    *,
    normalize_inputs: bool,
    means: list[float] | np.ndarray | None = None,
    stds: list[float] | np.ndarray | None = None,
    source: str = "finetune",
    modality_info: str | Path | None = None,
    normalization_modality: str = "vis_uv",
    band_filter: list[int] | None = None,
) -> NormalizationStrategy:
    if not normalize_inputs:
        return NoNormalization()
    if means is not None and stds is not None:
        strategy_cls: type[ZScoreNormalization] = (
            PretrainYamlNormalization
            if source == "pretrain"
            else FinetuneStatsNormalization
        )
        return strategy_cls(
            np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32),
        )
    if source == "pretrain":
        if modality_info is None:
            raise ValueError("modality_info is required for pretrain normalization.")
        return PretrainYamlNormalization.from_modality_info(
            modality_info,
            normalization_modality=normalization_modality,
            band_filter=band_filter,
        )
    raise ValueError(
        "normalize_inputs=True requires means/stds or pretrain modality_info."
    )
