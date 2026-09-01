"""Lunar tiling, chip creation, and supporting geospatial utilities."""

from .tiling_config import (
    BandNoDataOverride,
    TileConfig,
    TileSourceConfig,
    tile_config_from_dict,
)
from .lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH, load_lunar_geographic_wkt
from .vector_index import IndexedRaster, query_source_index
from .vector_index_builder import VectorIndexBuildConfig, create_vector_index
from .tiling_results import (
    MissingRequiredSourceError,
    TileCubeRecord,
    TileSourceError,
)
from .tiling import (
    create_tiles_for_aoi,
    create_tiles_for_index,
    create_tiles_for_point,
)
from .static_band_contract import (
    MINIRF_PRESERVE_SOURCE_NODATA_BANDS,
    MINIRF_STATIC_NODATA,
    STATIC_BAND_NAMES,
    STATIC_DEFAULT_NODATA,
)

__all__ = [
    "BandNoDataOverride",
    "LUNAR_GEOGRAPHIC_WKT_PATH",
    "IndexedRaster",
    "MissingRequiredSourceError",
    "MINIRF_PRESERVE_SOURCE_NODATA_BANDS",
    "MINIRF_STATIC_NODATA",
    "STATIC_BAND_NAMES",
    "STATIC_DEFAULT_NODATA",
    "TileConfig",
    "TileCubeRecord",
    "TileSourceError",
    "TileSourceConfig",
    "VectorIndexBuildConfig",
    "create_vector_index",
    "create_tiles_for_aoi",
    "create_tiles_for_index",
    "create_tiles_for_point",
    "tile_config_from_dict",
    "load_lunar_geographic_wkt",
    "query_source_index",
]
