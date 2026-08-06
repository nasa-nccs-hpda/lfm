"""Composable normalization strategies for lunar image tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch


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


def _load_stats_with_yaml(
    modality_info_path: Path,
) -> tuple[list[float], list[float]]:
    import yaml

    with modality_info_path.open("r", encoding="utf-8") as f:
        modality_info = yaml.safe_load(f)

    vis_stats = modality_info["vis"]["stats"]["vis"]
    uv_stats = modality_info["uv"]["stats"]["uv"]
    means = list(vis_stats["mean"]) + list(uv_stats["mean"])
    stds = list(vis_stats["std"]) + list(uv_stats["std"])
    return means, stds


def _load_stats_without_yaml(
    modality_info_path: Path,
) -> tuple[list[float], list[float]]:
    lines = modality_info_path.read_text(encoding="utf-8").splitlines()

    def top_level_block(name: str) -> list[str]:
        start = None
        for index, line in enumerate(lines):
            if line == f"{name}:":
                start = index + 1
                break
        if start is None:
            raise KeyError(name)

        end = len(lines)
        for index in range(start, len(lines)):
            line = lines[index]
            if line and not line.startswith(" ") and line.endswith(":"):
                end = index
                break
        return lines[start:end]

    def list_after(block: list[str], key: str) -> list[float]:
        for index, line in enumerate(block):
            if line.strip() == f"{key}:":
                values = []
                for value_line in block[index + 1 :]:
                    stripped = value_line.strip()
                    if not stripped.startswith("- "):
                        break
                    values.append(float(stripped[2:].strip().strip("'\"")))
                return values
        raise KeyError(key)

    vis_block = top_level_block("vis")
    uv_block = top_level_block("uv")
    means = list_after(vis_block, "mean") + list_after(uv_block, "mean")
    stds = list_after(vis_block, "std") + list_after(uv_block, "std")
    return means, stds


def load_terramind_wac_pretraining_stats(
    modality_info_path: str | Path,
    *,
    band_filter: list[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Load WAC normalization stats from TerraMind modality metadata.

    Lunar WAC chips are ordered as 5 VIS bands followed by 2 UV bands.
    TerraMind stores those as separate ``vis`` and ``uv`` modalities, so this
    returns the concatenated 7-band stats in chip order.
    """
    modality_info_path = Path(modality_info_path)
    try:
        means, stds = _load_stats_with_yaml(modality_info_path)
    except ModuleNotFoundError:
        means, stds = _load_stats_without_yaml(modality_info_path)
    except KeyError as exc:
        raise KeyError(
            f"Couldn't load vis/uv pretraining stats from {modality_info_path}"
        ) from exc

    if len(means) != 7 or len(stds) != 7:
        raise ValueError(
            "Expected TerraMind WAC stats to contain 5 VIS + 2 UV channels; "
            f"got {len(means)} means and {len(stds)} stds."
        )

    if band_filter is not None:
        means = [means[index] for index in band_filter]
        stds = [stds[index] for index in band_filter]

    print("TerraMind WAC pretraining mean:", means)
    print("TerraMind WAC pretraining std:", stds)
    return [float(value) for value in means], [float(value) for value in stds]


def _load_modality_stat(
    modality_info: dict,
    modality_name: str,
) -> tuple[float, float]:
    stats = modality_info[modality_name]["stats"][modality_name]
    mean = stats["mean"]
    std = stats["std"]
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(
            f"Expected one mean/std value for modality '{modality_name}', "
            f"got {len(mean)} means and {len(std)} stds."
        )
    return float(mean[0]), float(std[0])


def load_terramind_nac_pretraining_stats(
    modality_info_path: str | Path,
    *,
    band_filter: list[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Load NAC/DTM normalization stats from TerraMind modality metadata.

    The supported chip layouts are:

    * ``[0]``: PHO/NAC only.
    * ``[0, 1]``: PHO/NAC plus DTM.
    """
    import yaml

    modality_info_path = Path(modality_info_path)
    with modality_info_path.open("r", encoding="utf-8") as f:
        modality_info = yaml.safe_load(f)

    try:
        nac_mean, nac_std = _load_modality_stat(modality_info, "nac")
        dtm_mean, dtm_std = _load_modality_stat(modality_info, "dtm")
    except KeyError as exc:
        raise KeyError(
            f"Couldn't load nac/dtm pretraining stats from {modality_info_path}"
        ) from exc

    means = [nac_mean, dtm_mean]
    stds = [nac_std, dtm_std]

    if band_filter is None:
        band_filter = [0]

    if any(index not in {0, 1} for index in band_filter):
        raise ValueError(
            "normalization_modality='nac' only supports band_filter values "
            f"0 (PHO/NAC) and 1 (DTM), got {band_filter}."
        )

    means = [means[index] for index in band_filter]
    stds = [stds[index] for index in band_filter]

    print("TerraMind NAC pretraining modality info:", modality_info_path)
    print("TerraMind NAC pretraining mean:", means)
    print("TerraMind NAC pretraining std:", stds)
    return [float(value) for value in means], [float(value) for value in stds]


def load_terramind_pretraining_stats(
    modality_info_path: str | Path,
    *,
    normalization_modality: str = "vis_uv",
    band_filter: list[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Load pretraining normalization stats for a requested input modality."""
    if normalization_modality == "vis_uv":
        return load_terramind_wac_pretraining_stats(
            modality_info_path,
            band_filter=band_filter,
        )
    if normalization_modality == "nac":
        return load_terramind_nac_pretraining_stats(
            modality_info_path,
            band_filter=band_filter,
        )
    raise ValueError(
        "normalization_modality must be one of {'vis_uv', 'nac'}, got "
        f"{normalization_modality!r}."
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
