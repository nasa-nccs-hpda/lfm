"""Configuration-driven creation of modality-neutral lunar datacubes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .grid_router import route_aoi
from .grid_tile_def import tile_definition_for_grid
from .raster_cube import indexed_band_catalog, warp_source_to_tile, write_tile_cube
from .grid_registry import GeographicCoverage
from .tiling_config import TileConfig, TileSourceConfig
from .tiling_policy import (
    select_source_rasters,
    validate_source_selectors,
)
from .tiling_results import (
    MissingRequiredSourceError,
    TileCubeRecord,
    TileSourceError,
    tile_cube_filename,
)
from .vector_index import query_source_index_envelopes


class ConfiguredTiler:
    """Create grid-aligned cubes for every source in a :class:`TileConfig`."""

    def __init__(
        self,
        config: TileConfig,
        *,
        selectors: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.selectors = validate_source_selectors(config.sources, selectors)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._coverage_catalogs: dict[str, dict] = {}

    def _coverage_catalog(self, source: TileSourceConfig) -> dict:
        # Lazy, per-run cache: only metadata, no GDAL objects or shared writes.
        if source.name not in self._coverage_catalogs:
            inventory = query_source_index_envelopes(source, (
                GeographicCoverage(south=-90, west=-180, north=90, east=180),))
            selected = select_source_rasters(source, inventory,
                                             selector=self.selectors.get(source.name))
            self._coverage_catalogs[source.name] = indexed_band_catalog(
                source, [record.path for record in selected])
        return self._coverage_catalogs[source.name]

    def _product_id(self, source: TileSourceConfig) -> str | None:
        return (
            self.selectors[source.name]
            if source.selection_mode == "product_id"
            else None
        )

    def _output_path(
        self,
        source: TileSourceConfig,
        *,
        grid_id: str,
        tile_x: int,
        tile_y: int,
    ) -> Path:
        return self.config.output_dir / tile_cube_filename(
            source_name=source.name,
            grid_id=grid_id,
            zoom_level=self.config.zoom_level,
            tile_x=tile_x,
            tile_y=tile_y,
            product_id=self._product_id(source),
        )

    def run_tile_index(
        self,
        tile_x: int,
        tile_y: int,
        grid_id: str,
    ) -> list[TileCubeRecord]:
        tile_def = tile_definition_for_grid(grid_id, self.config.zoom_level)
        tile_def.validateTileIndex(tile_x, tile_y)
        ulx, uly, lrx, lry = tile_def.getTileBbox(tile_x, tile_y)
        query_envelopes = tile_def.geographic_query_envelopes(tile_x, tile_y)

        records: list[TileCubeRecord] = []
        for source in self.config.sources:
            try:
                indexed = query_source_index_envelopes(
                    source,
                    query_envelopes,
                )
                selected = select_source_rasters(
                    source,
                    indexed,
                    selector=self.selectors.get(source.name),
                )
                if not selected and not source.band_names:
                    if source.required:
                        raise MissingRequiredSourceError(
                            f"Required source {source.name!r} has no indexed data "
                            f"for grid {grid_id} tile ({tile_x}, {tile_y}).",
                            source_name=source.name,
                            zone=grid_id,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            completed_records=tuple(records),
                            product_id=self._product_id(source),
                        )
                    continue
                bands = warp_source_to_tile(
                    source,
                    [record.path for record in selected],
                    tile_def=tile_def,
                    bounds=(ulx, uly, lrx, lry),
                    coverage_catalog=lambda: self._coverage_catalog(source),
                )
                if not bands:
                    if source.required:
                        raise MissingRequiredSourceError(
                            f"Required source {source.name!r} has no valid bands "
                            f"for grid {grid_id} tile ({tile_x}, {tile_y}).",
                            source_name=source.name,
                            zone=grid_id,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            completed_records=tuple(records),
                            product_id=self._product_id(source),
                        )
                    continue
                output_path = self._output_path(
                    source,
                    grid_id=grid_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
                records.append(
                    write_tile_cube(
                        output_path,
                        bands,
                        source=source,
                        product_id=self._product_id(source),
                        zone=grid_id,
                        zoom_level=self.config.zoom_level,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        tile_def=tile_def,
                        ulx=ulx,
                        uly=uly,
                    )
                )
            except TileSourceError:
                raise
            except Exception as exc:
                raise TileSourceError(
                    f"Source {source.name!r} failed for grid {grid_id} tile "
                    f"({tile_x}, {tile_y}): {exc}",
                    source_name=source.name,
                    zone=grid_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    completed_records=tuple(records),
                    product_id=self._product_id(source),
                ) from exc
        return records

    def run_point(
        self,
        lat: float,
        lon: float,
        grid_id: str,
    ) -> list[TileCubeRecord]:
        tile_def = tile_definition_for_grid(grid_id, self.config.zoom_level)
        tile_index = tile_def.llToTileIndex(lat, lon)
        if tile_index is None:
            return []
        tile_x, tile_y = tile_index
        return self.run_tile_index(tile_x, tile_y, grid_id)

    def run_aoi(
        self,
        ul_lat: float,
        ul_lon: float,
        lr_lat: float,
        lr_lon: float,
    ) -> list[TileCubeRecord]:
        tile_indexes: set[tuple[str, int, int]] = set()
        for part in route_aoi(
            ul_lat=ul_lat,
            ul_lon=ul_lon,
            lr_lat=lr_lat,
            lr_lon=lr_lon,
        ):
            tile_def = tile_definition_for_grid(
                part.grid_id,
                self.config.zoom_level,
            )
            indices = tile_def.getOverlappingTiles(
                part.ul_lat,
                part.ul_lon,
                part.lr_lat,
                part.lr_lon,
            )
            tile_indexes.update(
                (part.grid_id, tile_x, tile_y)
                for tile_x, tile_y in indices
            )
        records: list[TileCubeRecord] = []
        for grid_id, tile_x, tile_y in sorted(
            tile_indexes,
            key=lambda item: (item[0], item[2], item[1]),
        ):
            try:
                records.extend(
                    self.run_tile_index(
                        tile_x,
                        tile_y,
                        grid_id,
                    )
                )
            except TileSourceError as exc:
                exc.completed_records = (
                    tuple(records) + tuple(exc.completed_records)
                )
                raise
        return records


__all__ = ["ConfiguredTiler"]
