"""Explicit creation of raster vector indexes used by the tiling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH, load_lunar_geographic_wkt


@dataclass(frozen=True)
class VectorIndexBuildConfig:
    """Describe an explicit Shapefile or GeoPackage raster-index build."""

    data_dir: Path
    index_path: Path
    image_glob: str = "*.tif"
    layer_name: str | None = None
    location_field: str = "location"
    output_srs_path: Path = LUNAR_GEOGRAPHIC_WKT_PATH

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "index_path", Path(self.index_path))
        object.__setattr__(self, "output_srs_path", Path(self.output_srs_path))
        if self.index_path.suffix.lower() not in {".shp", ".gpkg"}:
            raise ValueError("index_path must end with .shp or .gpkg.")
        if not self.image_glob.strip():
            raise ValueError("image_glob must not be empty.")
        if not self.location_field.strip():
            raise ValueError("location_field must not be empty.")


def create_vector_index(config: VectorIndexBuildConfig) -> Path:
    """Create a new raster vector index; never overwrite an existing index."""
    from osgeo import gdal

    gdal.UseExceptions()

    if not config.data_dir.is_dir():
        raise FileNotFoundError(f"Raster data directory does not exist: {config.data_dir}")
    if config.index_path.exists():
        raise FileExistsError(
            f"Raster vector index already exists: {config.index_path}. "
            "Remove or archive it explicitly before rebuilding."
        )
    raster_paths = sorted(config.data_dir.glob(config.image_glob))
    if not raster_paths:
        raise FileNotFoundError(
            f"No rasters matched {config.image_glob!r} in {config.data_dir}"
        )

    output_srs = (
        load_lunar_geographic_wkt()
        if config.output_srs_path == LUNAR_GEOGRAPHIC_WKT_PATH
        else config.output_srs_path.read_text(encoding="utf-8").strip()
    )
    format_name = "GPKG" if config.index_path.suffix.lower() == ".gpkg" else "ESRI Shapefile"
    layer_name = config.layer_name or config.index_path.stem
    options = gdal.TileIndexOptions(
        format=format_name,
        layerName=layer_name,
        locationFieldName=config.location_field,
        outputSRS=output_srs,
    )
    dataset = gdal.TileIndex(
        str(config.index_path),
        [str(path) for path in raster_paths],
        options=options,
    )
    if dataset is None:
        error = gdal.GetLastErrorMsg()
        detail = f": {error}" if error else ""
        raise RuntimeError(f"Could not create raster vector index{detail}")
    dataset = None
    return config.index_path


__all__ = ["VectorIndexBuildConfig", "create_vector_index"]
