"""Toy model training, inference, and shared utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _proj_dir_candidates(prefix: Path) -> list[Path]:
    candidates = [
        prefix / "share" / "proj",
        prefix / "Library" / "share" / "proj",
    ]
    prefix_text = str(prefix)
    if prefix_text.startswith("/explore/nobackup/"):
        candidates.append(
            Path(
                prefix_text.replace("/explore/nobackup/", "/panfs/ccds02/nobackup/", 1)
            )
            / "share"
            / "proj"
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


_configure_geospatial_data_dir()
