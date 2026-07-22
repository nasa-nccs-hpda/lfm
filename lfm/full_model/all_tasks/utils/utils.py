"""General notebook utilities for Graha/Lunar-FM fine-tuning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import yaml


def create_timestamped_output_dir(base_dir: str | Path) -> Path:
    """Create a timestamped subdirectory under ``base_dir``."""
    timestamp = datetime.now().strftime("date_%Y_%m_%d-time_%H_%M_%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output subdir: {output_dir}")
    return output_dir


def ensure_data_symlink(
    simlink_dest: str | Path | None,
    data_symlink: str | Path = "data",
) -> Path:
    """Create or verify a data symlink.

    If ``simlink_dest`` is ``None``, this leaves ``data_symlink`` unchanged. If
    ``data_symlink`` already exists as a symlink, it must point to the same
    resolved destination. Existing non-symlink paths are rejected.
    """
    data_symlink = Path(data_symlink)
    if simlink_dest is None:
        print("SIMLINK_DEST is None; leaving data symlink unchanged.")
        return data_symlink

    source = Path(simlink_dest).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"SIMLINK_DEST does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"SIMLINK_DEST must be a directory: {source}")

    if data_symlink.is_symlink():
        current_target = data_symlink.resolve()
        if current_target != source:
            raise FileExistsError(
                f"{data_symlink} already points to {current_target}, not "
                f"SIMLINK_DEST {source}. Remove or update the symlink "
                "explicitly before continuing."
            )
        print(f"Symlink created successfully: {data_symlink} -> {source}")
    elif data_symlink.exists():
        raise FileExistsError(
            f"{data_symlink} already exists and is not a symlink. "
            "Move it before creating the data symlink."
        )
    else:
        data_symlink.symlink_to(source, target_is_directory=True)
        print(f"Symlink created successfully: {data_symlink} -> {source}")

    return data_symlink


def _load_stats_with_yaml(
        modality_info_path: Path
) -> tuple[list[float], list[float]]:

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
                for value_line in block[index+1:]:
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

    The Lunar WAC chips are ordered as 5 VIS bands followed by 2 UV bands.
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
