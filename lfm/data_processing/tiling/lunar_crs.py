"""Repository-owned lunar coordinate reference system definitions."""

from __future__ import annotations

import math
from typing import Any

from .._paths import REPO_ROOT


LUNAR_GEOGRAPHIC_WKT_PATH = REPO_ROOT / "TMS" / "IAU_30100_2015.wkt"


def load_lunar_geographic_wkt() -> str:
    """Load the IAU:30100 lunar geographic CRS bundled with the repository."""
    wkt = LUNAR_GEOGRAPHIC_WKT_PATH.read_text(encoding="utf-8").strip()
    if not wkt:
        raise ValueError(
            f"Lunar geographic CRS file is empty: {LUNAR_GEOGRAPHIC_WKT_PATH}"
        )
    return wkt


def _proj4_raster_signature(spatial_reference: Any):
    """Return the coordinate-operation part of a projected CRS.

    GeoTIFF does not persist every WKT axis and authority detail for a custom
    projected CRS. PROJ.4 is intentionally used only as a fallback signature
    after OSR's full semantic comparison fails. Runtime/data-axis declarations
    are excluded because LFM raster coordinates consistently use traditional
    easting/northing order.
    """
    try:
        text = spatial_reference.ExportToProj4()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not text:
        return None
    signature: dict[str, str | float | bool] = {}
    for token in text.split():
        if token == "+type=crs" or token.startswith("+axis="):
            continue
        key, separator, raw_value = token.partition("=")
        if not key.startswith("+") or key in signature:
            return None
        if not separator:
            signature[key] = True
            continue
        try:
            signature[key] = float(raw_value)
        except ValueError:
            signature[key] = raw_value.casefold()
    return signature


def _signature_values_equal(first: object, second: object) -> bool:
    if isinstance(first, float) and isinstance(second, float):
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def raster_crs_equivalent(first: Any, second: Any) -> bool:
    """Compare raster CRSs while tolerating GeoTIFF axis metadata loss.

    Callers must put both inputs in their intended data-axis mapping strategy
    before calling this function. OSR remains authoritative whenever it finds
    the CRSs equivalent. The narrow fallback applies only to two projected
    CRSs and compares their complete PROJ.4 coordinate-operation signatures,
    excluding axis metadata that does not change LFM's easting/northing raster
    data order.
    """
    if first.IsSame(second):
        return True
    if not first.IsProjected() or not second.IsProjected():
        return False
    first_signature = _proj4_raster_signature(first)
    second_signature = _proj4_raster_signature(second)
    if (
        first_signature is None
        or second_signature is None
        or first_signature.keys() != second_signature.keys()
    ):
        return False
    return all(
        _signature_values_equal(first_signature[key], second_signature[key])
        for key in first_signature
    )


__all__ = [
    "LUNAR_GEOGRAPHIC_WKT_PATH",
    "load_lunar_geographic_wkt",
    "raster_crs_equivalent",
]
