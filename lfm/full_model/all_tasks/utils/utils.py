"""General notebook utilities for Graha/Lunar-FM fine-tuning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lfm.all_models.all_tasks.data.normalization import (
    load_terramind_nac_pretraining_stats as load_terramind_nac_pretraining_stats,
    load_terramind_pretraining_stats as load_terramind_pretraining_stats,
    load_terramind_wac_pretraining_stats as load_terramind_wac_pretraining_stats,
)


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
