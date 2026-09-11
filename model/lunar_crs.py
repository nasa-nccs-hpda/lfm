"""Repository-owned lunar coordinate reference system definitions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LUNAR_GEOGRAPHIC_WKT_PATH = REPO_ROOT / "TMS" / "IAU_30100_2015.wkt"


def load_lunar_geographic_wkt() -> str:
    """Load the IAU:30100 lunar geographic CRS bundled with the repository."""
    wkt = LUNAR_GEOGRAPHIC_WKT_PATH.read_text(encoding="utf-8").strip()
    if not wkt:
        raise ValueError(
            f"Lunar geographic CRS file is empty: {LUNAR_GEOGRAPHIC_WKT_PATH}"
        )
    return wkt


__all__ = ["LUNAR_GEOGRAPHIC_WKT_PATH", "load_lunar_geographic_wkt"]
