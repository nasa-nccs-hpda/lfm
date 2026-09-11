"""Structured result and error contracts for LTM datacube creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


def safe_filename_component(value: str) -> str:
    """Return a deterministic filesystem-safe tiling identifier."""
    result = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip()).strip("-")
    if not result:
        raise ValueError(f"Cannot create a safe filename component from {value!r}")
    return result


def tile_cube_filename(
    *,
    source_name: str,
    zone: str,
    zoom_level: int,
    tile_x: int,
    tile_y: int,
    product_id: str | None = None,
) -> str:
    """Build the generic filename for one configured source cube."""
    product_suffix = (
        f"_Product-{safe_filename_component(product_id)}"
        if product_id is not None
        else ""
    )
    return (
        f"Cube-{safe_filename_component(source_name)}-LTM"
        f"{safe_filename_component(zone)}_Zoom-{int(zoom_level)}"
        f"_Tile-{int(tile_x)}-{int(tile_y)}{product_suffix}.tif"
    )


@dataclass(frozen=True)
class TileCubeRecord:
    """Metadata for one source cube written on an LTM tile grid."""

    source_name: str
    zone: str
    zoom_level: int
    tile_x: int
    tile_y: int
    product_id: str | None
    path: Path
    band_names: tuple[str, ...]
    crs_wkt: str
    nodata_values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_name", str(self.source_name).strip())
        object.__setattr__(self, "zone", str(self.zone).strip())
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "band_names", tuple(self.band_names))
        object.__setattr__(self, "nodata_values", tuple(self.nodata_values))
        if not self.source_name:
            raise ValueError("source_name must not be empty.")
        if not self.zone:
            raise ValueError("zone must not be empty.")
        if self.zoom_level < 1:
            raise ValueError("zoom_level must be positive.")
        if not self.band_names:
            raise ValueError("TileCubeRecord requires at least one band.")
        if len(self.band_names) != len(self.nodata_values):
            raise ValueError("band_names and nodata_values must have equal lengths.")
        if not self.crs_wkt.strip():
            raise ValueError("crs_wkt must not be empty.")


class TileSourceError(RuntimeError):
    """A configured source failed while creating one LTM tile."""

    def __init__(
        self,
        message: str,
        *,
        source_name: str,
        zone: str,
        tile_x: int,
        tile_y: int,
        completed_records: tuple[TileCubeRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.source_name = source_name
        self.zone = zone
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.completed_records = completed_records


class MissingRequiredSourceError(TileSourceError):
    """A required source had no usable raster data for an LTM tile."""


__all__ = [
    "MissingRequiredSourceError",
    "TileCubeRecord",
    "TileSourceError",
    "safe_filename_component",
    "tile_cube_filename",
]
