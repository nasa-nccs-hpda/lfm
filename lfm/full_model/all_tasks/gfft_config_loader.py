"""Load TerraTorch-style GFFT YAML config metadata.

The science-team GFFT configs are TerraTorch Lightning CLI files. This module
extracts the model and normalization pieces needed by the LFM OOP workflows
without making the workflows depend directly on Lightning CLI parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

import yaml


@dataclass(frozen=True)
class GfftNormalizationStats:
    """Per-modality normalization stats extracted from a TerraTorch YAML."""

    modality: str
    means: list[float]
    stds: list[float]
    mean_key: str
    std_key: str


@dataclass(frozen=True)
class GfftConfig:
    """Typed view of the GFFT-relevant portions of a TerraTorch YAML."""

    path: Path
    raw: dict[str, Any]
    trainer: dict[str, Any]
    data_init_args: dict[str, Any]
    model_init_args: dict[str, Any]
    model_args: dict[str, Any]
    backbone_checkpoint_path: PurePath | None
    backbone_modalities: list[str]
    backbone_new_modalities: dict[str, Any] | None
    backbone_patch_size: int | None
    backbone_image_size: int | None
    backbone_drop_path_rate: float | None
    necks: list[dict[str, Any]]
    optimizer_args: dict[str, Any]
    normalization_stats: dict[str, GfftNormalizationStats]

    def stats_for_modality(self, modality: str) -> GfftNormalizationStats:
        """Return stats for a named modality, falling back only when unambiguous."""
        if modality in self.normalization_stats:
            return self.normalization_stats[modality]
        if len(self.normalization_stats) == 1:
            return next(iter(self.normalization_stats.values()))
        available = ", ".join(sorted(self.normalization_stats)) or "<none>"
        raise KeyError(
            f"No GFFT normalization stats for modality {modality!r}. "
            f"Available modalities: {available}."
        )


def resolve_gfft_normalization_stats(
    config: GfftConfig,
    *,
    normalization_modality: str,
    band_filter: list[int] | None,
) -> tuple[list[float], list[float]]:
    """Return datamodule means/stds from a loaded GFFT YAML config.

    The LFM CLIs still use Graha-era normalization modality names
    (``vis_uv``/``nac``), while the GFFT YAMLs often store stats under task-data
    names such as ``wac`` or ``nac``. This resolver keeps that translation in
    one place and applies the same band-filtering expectation used by the
    datamodule image reader.
    """
    stats = _resolve_stats_record(config, normalization_modality)
    return _apply_band_filter(
        stats.means,
        stats.stds,
        band_filter=band_filter,
        stats_label=f"{config.path}::{stats.modality}",
    )


def _resolve_stats_record(
    config: GfftConfig,
    normalization_modality: str,
) -> GfftNormalizationStats:
    modality = normalization_modality.replace("-", "_").lower()
    if modality == "vis_uv":
        if "vis" in config.normalization_stats and "uv" in config.normalization_stats:
            vis = config.normalization_stats["vis"]
            uv = config.normalization_stats["uv"]
            return GfftNormalizationStats(
                modality="vis_uv",
                means=[*vis.means, *uv.means],
                stds=[*vis.stds, *uv.stds],
                mean_key=f"{vis.mean_key}+{uv.mean_key}",
                std_key=f"{vis.std_key}+{uv.std_key}",
            )
        for candidate in ("vis_uv", "wac", "vis"):
            if candidate in config.normalization_stats:
                return config.normalization_stats[candidate]
    elif modality == "wac":
        for candidate in ("wac", "vis_uv", "vis"):
            if candidate in config.normalization_stats:
                return config.normalization_stats[candidate]
    elif modality in config.normalization_stats:
        return config.normalization_stats[modality]

    return config.stats_for_modality(modality)


def _apply_band_filter(
    means: list[float],
    stds: list[float],
    *,
    band_filter: list[int] | None,
    stats_label: str,
) -> tuple[list[float], list[float]]:
    if band_filter is None:
        return list(means), list(stds)

    if len(means) != len(stds):
        raise ValueError(
            f"GFFT normalization stats for {stats_label} have mismatched lengths: "
            f"{len(means)} mean(s), {len(stds)} std(s)."
        )
    if not band_filter:
        return list(means), list(stds)

    max_band = max(band_filter)
    min_band = min(band_filter)
    if min_band < 0 or max_band >= len(means):
        raise ValueError(
            f"GFFT normalization stats for {stats_label} have {len(means)} "
            f"channel(s), but band_filter={band_filter} requires indices "
            f"{min_band}..{max_band}."
        )
    return [means[index] for index in band_filter], [
        stds[index] for index in band_filter
    ]


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _infer_generic_modality(
    data_init_args: dict[str, Any],
    model_args: dict[str, Any],
) -> str:
    output_mode = data_init_args.get("output_mode")
    if isinstance(output_mode, str) and output_mode not in {"packed", "image"}:
        return output_mode

    modalities = model_args.get("backbone_modalities") or []
    if isinstance(modalities, str):
        return modalities
    if isinstance(modalities, list) and len(modalities) == 1:
        return str(modalities[0])
    return "default"


def _add_stats_if_complete(
    stats: dict[str, GfftNormalizationStats],
    *,
    modality: str,
    mean_value: Any,
    std_value: Any,
    mean_key: str,
    std_key: str,
) -> None:
    if mean_value is None and std_value is None:
        return
    if mean_value is None or std_value is None:
        raise ValueError(
            f"GFFT config has incomplete normalization stats for {modality!r}: "
            f"{mean_key}={mean_value!r}, {std_key}={std_value!r}."
        )
    stats[modality] = GfftNormalizationStats(
        modality=modality,
        means=_as_float_list(mean_value),
        stds=_as_float_list(std_value),
        mean_key=mean_key,
        std_key=std_key,
    )


def extract_normalization_stats(
    data_init_args: dict[str, Any],
    model_args: dict[str, Any] | None = None,
) -> dict[str, GfftNormalizationStats]:
    """Extract normalization stats from ``data.init_args``.

    Supported TerraTorch config shapes include:
    - ``norm_mean`` / ``norm_std``
    - ``norm_means`` / ``norm_stds``
    - ``means`` / ``stds``
    - ``<modality>_norm_mean`` / ``<modality>_norm_std``
    - ``<modality>_norm_means`` / ``<modality>_norm_stds``
    """
    model_args = model_args or {}
    stats: dict[str, GfftNormalizationStats] = {}
    generic_modality = _infer_generic_modality(data_init_args, model_args)

    _add_stats_if_complete(
        stats,
        modality=generic_modality,
        mean_value=data_init_args.get("norm_mean"),
        std_value=data_init_args.get("norm_std"),
        mean_key="norm_mean",
        std_key="norm_std",
    )
    _add_stats_if_complete(
        stats,
        modality=generic_modality,
        mean_value=data_init_args.get("norm_means"),
        std_value=data_init_args.get("norm_stds"),
        mean_key="norm_means",
        std_key="norm_stds",
    )
    _add_stats_if_complete(
        stats,
        modality=generic_modality,
        mean_value=data_init_args.get("means"),
        std_value=data_init_args.get("stds"),
        mean_key="means",
        std_key="stds",
    )

    modality_prefixes = {
        key[: -len("_norm_mean")]
        for key in data_init_args
        if key.endswith("_norm_mean")
    }
    modality_prefixes |= {
        key[: -len("_norm_means")]
        for key in data_init_args
        if key.endswith("_norm_means")
    }

    for modality in sorted(modality_prefixes):
        if f"{modality}_norm_means" in data_init_args:
            mean_key = f"{modality}_norm_means"
            std_key = f"{modality}_norm_stds"
        else:
            mean_key = f"{modality}_norm_mean"
            std_key = f"{modality}_norm_std"
        _add_stats_if_complete(
            stats,
            modality=modality,
            mean_value=data_init_args.get(mean_key),
            std_value=data_init_args.get(std_key),
            mean_key=mean_key,
            std_key=std_key,
        )

    return stats


def _optional_path(value: Any, *, base_dir: Path) -> PurePath | None:
    if value is None:
        return None
    value_str = str(value)
    if value_str.startswith("/") and not value_str.startswith("//"):
        return PurePosixPath(value_str)
    path = Path(value_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _gfft_project_root(config_path: Path) -> Path:
    """Return the Graha/Lunar-FM root for TerraTorch integration config paths."""
    for parent in config_path.parents:
        if parent.name == "graha-lunar-fm":
            return parent
    return config_path.parent


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"GFFT config section {section!r} must be a mapping.")
    return value


def load_gfft_config(path: str | Path) -> GfftConfig:
    """Load a TerraTorch-style GFFT YAML and return extracted metadata."""
    config_path = Path(path).expanduser().resolve()
    project_root = _gfft_project_root(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _require_mapping(raw, "root")

    trainer = _require_mapping(raw.get("trainer", {}), "trainer")
    data_section = _require_mapping(raw.get("data", {}), "data")
    data_init_args = _require_mapping(
        data_section.get("init_args", {}), "data.init_args"
    )
    model_section = _require_mapping(raw.get("model", {}), "model")
    model_init_args = _require_mapping(
        model_section.get("init_args", {}),
        "model.init_args",
    )
    model_args = _require_mapping(
        model_init_args.get("model_args", {}),
        "model.init_args.model_args",
    )

    backbone_modalities = model_args.get("backbone_modalities") or []
    if isinstance(backbone_modalities, str):
        backbone_modalities = [backbone_modalities]
    else:
        backbone_modalities = [str(item) for item in backbone_modalities]

    optimizer_keys = {
        "backbone_lr",
        "head_lr",
        "layer_decay",
        "weight_decay",
        "head_weight_decay",
        "betas",
        "warmup_steps",
        "eta_min",
    }
    optimizer_args = {
        key: model_init_args[key] for key in optimizer_keys if key in model_init_args
    }

    return GfftConfig(
        path=config_path,
        raw=raw,
        trainer=trainer,
        data_init_args=data_init_args,
        model_init_args=model_init_args,
        model_args=model_args,
        backbone_checkpoint_path=_optional_path(
            model_args.get("backbone_checkpoint_path"),
            base_dir=project_root,
        ),
        backbone_modalities=backbone_modalities,
        backbone_new_modalities=model_args.get("backbone_new_modalities"),
        backbone_patch_size=model_args.get("backbone_patch_size"),
        backbone_image_size=model_args.get("backbone_image_size"),
        backbone_drop_path_rate=model_args.get("backbone_drop_path_rate"),
        necks=list(model_args.get("necks") or []),
        optimizer_args=optimizer_args,
        normalization_stats=extract_normalization_stats(data_init_args, model_args),
    )
