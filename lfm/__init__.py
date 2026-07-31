"""Toy model training, inference, and shared utilities."""

from __future__ import annotations

import os
import site
import sys
import sysconfig
from pathlib import Path


def _active_env_site_paths() -> list[str]:
    """Return site-package paths for the active Python environment."""
    candidates = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            candidates.append(path)

    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass

    seen = set()
    paths = []
    for path in candidates:
        resolved = str(Path(path))
        key = resolved.lower()
        if key not in seen and Path(resolved).exists():
            seen.add(key)
            paths.append(resolved)
    return paths


def _is_user_site_path(path: str, user_paths: set[str]) -> bool:
    normalized = path.lower()
    return ".local" in normalized or any(
        normalized.startswith(user_path) for user_path in user_paths
    )


def _prioritize_active_env_packages() -> None:
    """Keep notebook kernels from preferring ~/.local over the active env."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    site.ENABLE_USER_SITE = False

    user_paths = {
        str(Path(site.getuserbase())).lower(),
        str(Path(site.getusersitepackages())).lower(),
    }
    active_env_paths = _active_env_site_paths()
    filtered_sys_path = [
        path
        for path in sys.path
        if path
        and not _is_user_site_path(path, user_paths)
        and path not in active_env_paths
    ]
    sys.path = [*active_env_paths, *filtered_sys_path]

    existing_pythonpath = [
        path
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path and not _is_user_site_path(path, user_paths)
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*active_env_paths, *existing_pythonpath]
    )


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
        ]
    )
    return candidates


def _configure_geospatial_data_dir() -> None:
    """Configure PROJ before TerraTorch/TorchGeo import pyproj CRS objects."""
    prefix = Path(sys.prefix)
    for proj_dir in _proj_dir_candidates(prefix):
        if (proj_dir / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(proj_dir)
            os.environ["PROJ_DATA"] = str(proj_dir)
            gdal_dir = prefix / "share" / "gdal"
            if gdal_dir.exists():
                os.environ["GDAL_DATA"] = str(gdal_dir)
            try:
                import pyproj

                pyproj.datadir.set_data_dir(str(proj_dir))
            except Exception:
                pass
            return


_prioritize_active_env_packages()
_configure_geospatial_data_dir()
