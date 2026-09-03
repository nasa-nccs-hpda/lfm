"""Configuration-driven creation of modality-neutral LTM datacubes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .TmsIntersector import TmsIntersector
from .TmsTileDef import TmsTileDef
from .raster_cube import warp_source_to_tile, write_tile_cube
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
from .vector_index import query_source_index


class ConfiguredTiler:
    """Create LTM datacubes for every source declared in a :class:`TileConfig`."""

    def __init__(
        self,
        config: TileConfig,
        *,
        selectors: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.selectors = validate_source_selectors(config.sources, selectors)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _output_path(
        self,
        source: TileSourceConfig,
        *,
        zone: str,
        tile_x: int,
        tile_y: int,
    ) -> Path:
        return self.config.output_dir / tile_cube_filename(
            source_name=source.name,
            zone=zone,
            zoom_level=self.config.zoom_level,
            tile_x=tile_x,
            tile_y=tile_y,
            product_id=(
                self.selectors[source.name]
                if source.selection_mode == "product_id"
                else None
            ),
        )

    def run_tile_index(
        self,
        tile_x: int,
        tile_y: int,
        zone: str,
    ) -> list[TileCubeRecord]:
        tile_def = TmsTileDef.initFromParams(zone, self.config.zoom_level)
        ulx, uly, lrx, lry = tile_def.getTileBbox(tile_x, tile_y)
        ul_lat, ul_lon = tile_def.ltmToLatLon(ulx, uly)
        lr_lat, lr_lon = tile_def.ltmToLatLon(lrx, lry)

        records: list[TileCubeRecord] = []
        for source in self.config.sources:
            try:
                indexed = query_source_index(
                    source,
                    ul_lat=ul_lat,
                    ul_lon=ul_lon,
                    lr_lat=lr_lat,
                    lr_lon=lr_lon,
                )
                selected = select_source_rasters(
                    source,
                    indexed,
                    selector=self.selectors.get(source.name),
                )
                if not selected:
                    if source.required:
                        raise MissingRequiredSourceError(
                            f"Required source {source.name!r} has no indexed data "
                            f"for LTM{zone} tile ({tile_x}, {tile_y}).",
                            source_name=source.name,
                            zone=zone,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            completed_records=tuple(records),
                        )
                    continue
                bands = warp_source_to_tile(
                    source,
                    [record.path for record in selected],
                    tile_def=tile_def,
                    bounds=(ulx, uly, lrx, lry),
                )
                if not bands:
                    if source.required:
                        raise MissingRequiredSourceError(
                            f"Required source {source.name!r} has no valid bands "
                            f"for LTM{zone} tile ({tile_x}, {tile_y}).",
                            source_name=source.name,
                            zone=zone,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            completed_records=tuple(records),
                        )
                    continue
                output_path = self._output_path(
                    source,
                    zone=zone,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
                records.append(
                    write_tile_cube(
                        output_path,
                        bands,
                        source=source,
                        product_id=(
                            self.selectors[source.name]
                            if source.selection_mode == "product_id"
                            else None
                        ),
                        zone=zone,
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
                    f"Source {source.name!r} failed for LTM{zone} tile "
                    f"({tile_x}, {tile_y}): {exc}",
                    source_name=source.name,
                    zone=zone,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    completed_records=tuple(records),
                ) from exc
        return records

    def run_point(self, lat: float, lon: float, zone: str) -> list[TileCubeRecord]:
        tile_def = TmsTileDef.initFromParams(zone, self.config.zoom_level)
        tile_index = tile_def.llToTileIndex(lat, lon)
        if tile_index is None:
            return []
        tile_x, tile_y = tile_index
        return self.run_tile_index(tile_x, tile_y, zone)

    def run_aoi(
        self,
        ul_lat: float,
        ul_lon: float,
        lr_lat: float,
        lr_lon: float,
    ) -> list[TileCubeRecord]:
        tile_indexes = TmsIntersector().getTids(
            ul_lat,
            ul_lon,
            lr_lat,
            lr_lon,
            self.config.zoom_level,
        )
        records: list[TileCubeRecord] = []
        for index in sorted(
            tile_indexes,
            key=lambda item: (item["zone"], item["tileY"], item["tileX"]),
        ):
            records.extend(
                self.run_tile_index(
                    index["tileX"],
                    index["tileY"],
                    index["zone"],
                )
            )
        return records


__all__ = ["ConfiguredTiler"]
