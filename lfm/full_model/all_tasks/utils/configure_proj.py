import os
import sys
from pathlib import Path

def configure_proj_environment() -> None:
    """Configure PROJ/GDAL data paths for Conda or container environments."""

    prefix = Path(sys.executable).parents[1]

    proj_candidates = [
        # Conda / venv-local layouts
        prefix / "share" / "proj",
        prefix / "Library" / "share" / "proj",

        # System/container layout
        Path("/usr/share/proj"),
        Path("/usr/local/share/proj"),
    ]

    for candidate in proj_candidates:
        if (candidate / "proj.db").exists():
            proj_dir = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not find proj.db in any expected location: "
            + ", ".join(str(p) for p in proj_candidates)
        )

    gdal_candidates = [
        prefix / "share" / "gdal",
        prefix / "Library" / "share" / "gdal",
        Path("/usr/share/gdal"),
        Path("/usr/local/share/gdal"),
    ]

    for candidate in gdal_candidates:
        if candidate.is_dir():
            gdal_dir = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not find GDAL data directory in any expected location: "
            + ", ".join(str(p) for p in gdal_candidates)
        )

    os.environ["PROJ_LIB"] = str(proj_dir)
    os.environ["PROJ_DATA"] = str(proj_dir)
    os.environ["GDAL_DATA"] = str(gdal_dir)

    print("PROJ_DATA =", os.environ["PROJ_DATA"])
    print("GDAL_DATA =", os.environ["GDAL_DATA"])
