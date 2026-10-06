"""Format-independent vector-index access for tiling raster sources."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .grid_registry import GeographicCoverage
from .tiling_config import TileSourceConfig


@dataclass(frozen=True)
class IndexedRaster:
    """One raster path returned by a spatial vector-index query."""

    path: Path
    feature_id: int | None = None


def resolve_indexed_raster_path(data_dir: Path, stored_path: object) -> Path:
    """Resolve an index field value against its source data directory."""
    if stored_path is None or not str(stored_path).strip():
        raise ValueError("Raster index contains an empty raster path.")
    path = Path(str(stored_path).strip())
    return path if path.is_absolute() else Path(data_dir) / path


def open_vector_layer(
    index_path: str | Path,
    *,
    layer_name: str | None = None,
):
    """Open a read-only OGR vector dataset and return its selected layer.

    The returned tuple is ``(dataset, layer)``. Callers must retain the dataset
    for at least as long as they use the layer because OGR layers are owned by
    their containing dataset.
    """
    from osgeo import gdal

    gdal.UseExceptions()

    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Raster vector index does not exist: {index_path}")

    dataset = gdal.OpenEx(
        str(index_path),
        gdal.OF_VECTOR | gdal.OF_READONLY,
    )
    if dataset is None:
        error = gdal.GetLastErrorMsg()
        detail = f": {error}" if error else ""
        raise RuntimeError(f"Could not open raster vector index {index_path}{detail}")

    if layer_name is None:
        layer = dataset.GetLayer(0)
    else:
        layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        available = [
            dataset.GetLayer(index).GetName()
            for index in range(dataset.GetLayerCount())
        ]
        requested = layer_name if layer_name is not None else "layer 0"
        raise KeyError(
            f"Could not find {requested!r} in {index_path}. "
            f"Available layers: {available}"
        )
    return dataset, layer


def _require_location_field(layer: Any, source: TileSourceConfig) -> None:
    definition = layer.GetLayerDefn()
    if definition.GetFieldIndex(source.location_field) >= 0:
        return
    available = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    raise KeyError(
        f"Raster index {source.index_path} layer {layer.GetName()!r} does not "
        f"contain location field {source.location_field!r}. "
        f"Available fields: {available}"
    )


def query_source_index(
    source: TileSourceConfig,
    *,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
) -> list[IndexedRaster]:
    """Return rasters whose index footprints intersect a geographic AOI."""
    return query_source_index_envelopes(
        source,
        (
            GeographicCoverage(
                south=lr_lat,
                west=ul_lon,
                north=ul_lat,
                east=lr_lon,
            ),
        ),
    )


def query_source_index_envelopes(
    source: TileSourceConfig,
    envelopes: tuple[GeographicCoverage, ...],
) -> list[IndexedRaster]:
    """Query one or more non-wrapping envelopes and deduplicate records."""
    if not envelopes:
        return []
    dataset, layer = open_vector_layer(
        source.index_path,
        layer_name=source.index_layer,
    )
    try:
        _require_location_field(layer, source)
        records: dict[str, IndexedRaster] = {}
        for envelope in envelopes:
            layer.SetSpatialFilterRect(
                envelope.west,
                envelope.south,
                envelope.east,
                envelope.north,
            )
            layer.ResetReading()
            for feature in layer:
                feature_id = (
                    int(feature.GetFID())
                    if feature.GetFID() is not None
                    else None
                )
                record = IndexedRaster(
                    path=resolve_indexed_raster_path(
                        source.data_dir,
                        feature.GetField(source.location_field),
                    ),
                    feature_id=feature_id,
                )
                key = str(record.path)
                previous = records.get(key)
                previous_fid = (
                    previous.feature_id
                    if previous is not None and previous.feature_id is not None
                    else math.inf
                )
                record_fid = feature_id if feature_id is not None else math.inf
                if previous is None or record_fid < previous_fid:
                    records[key] = record
    finally:
        layer.SetSpatialFilter(None)
        layer = None
        dataset = None
    return sorted(
        records.values(),
        key=lambda record: (
            str(record.path),
            record.feature_id if record.feature_id is not None else -1,
        ),
    )


__all__ = [
    "IndexedRaster",
    "open_vector_layer",
    "query_source_index",
    "query_source_index_envelopes",
    "resolve_indexed_raster_path",
]
