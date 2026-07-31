"""General notebook utilities for Graha/Lunar-FM fine-tuning."""

from __future__ import annotations

import os
import sys
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


def _proj_dir_candidates(prefix: Path) -> list[Path]:
    candidates = []
    prefix_text = str(prefix)
    if prefix_text.startswith("/explore/nobackup/"):
        candidates.append(
            Path(
                prefix_text.replace("/explore/nobackup/", "/panfs/ccds02/nobackup/", 1)
            )
            / "share"
            / "proj"
        )
    candidates.extend(
        [
            prefix / "share" / "proj",
            prefix / "Library" / "share" / "proj",
            Path("/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/proj"),
        ]
    )
    return candidates


def setup_proj(
    proj_dir: str | Path | None = None,
    *,
    gdal_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Configure PROJ/GDAL for notebook and script geospatial imports.

    Call this before importing packages that eagerly construct pyproj CRS
    objects, such as TorchGeo through TerraTorch.
    """
    if proj_dir is None:
        for candidate in _proj_dir_candidates(Path(sys.prefix)):
            if (candidate / "proj.db").exists():
                proj_dir = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not find proj.db. Pass setup_proj(proj_dir=...) explicitly."
            )

    resolved_proj_dir = Path(proj_dir)
    if not (resolved_proj_dir / "proj.db").exists():
        raise FileNotFoundError(f"proj.db not found under {resolved_proj_dir}")

    if gdal_dir is None:
        candidate_gdal_dir = Path(sys.prefix) / "share" / "gdal"
        if str(candidate_gdal_dir).startswith("/explore/nobackup/"):
            panfs_gdal_dir = Path(
                str(candidate_gdal_dir).replace(
                    "/explore/nobackup/",
                    "/panfs/ccds02/nobackup/",
                    1,
                )
            )
            if panfs_gdal_dir.exists():
                candidate_gdal_dir = panfs_gdal_dir
        gdal_dir = candidate_gdal_dir

    os.environ["PROJ_LIB"] = str(resolved_proj_dir)
    os.environ["PROJ_DATA"] = str(resolved_proj_dir)
    if gdal_dir is not None and Path(gdal_dir).exists():
        os.environ["GDAL_DATA"] = str(Path(gdal_dir))

    try:
        import pyproj

        pyproj.datadir.set_data_dir(str(resolved_proj_dir))
        if verbose:
            print("pyproj data dir:", pyproj.datadir.get_data_dir())
    except ImportError:
        if verbose:
            print("pyproj is not installed; set PROJ environment variables only.")

    if verbose:
        print("PROJ_DATA:", os.environ["PROJ_DATA"])
        print("GDAL_DATA:", os.environ.get("GDAL_DATA"))
    return resolved_proj_dir


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
