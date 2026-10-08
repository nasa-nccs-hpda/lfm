"""Product-ID resolver contracts for product-scoped lunar raster sources."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias


ProductIdResolver: TypeAlias = Callable[[Path], str]


class ProductIdResolutionError(ValueError):
    """A configured resolver could not produce a usable product ID."""


def lunar_product_id_from_raster_path(path: str | Path) -> str:
    """Return the filename prefix before the first period as a lunar PID."""
    raster_path = Path(path)
    product_id = raster_path.name.split(".", 1)[0].strip()
    if not product_id:
        raise ValueError(
            f"Raster filename has no product prefix before its first period: "
            f"{raster_path}"
        )
    return product_id


__all__ = [
    "ProductIdResolver",
    "ProductIdResolutionError",
    "lunar_product_id_from_raster_path",
]
