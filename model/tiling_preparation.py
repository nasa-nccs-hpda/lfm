"""High-level source-index preparation before read-only tile generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TextIO

from .lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH
from .tiling_config import TileConfig, TileSourceConfig
from .vector_index_builder import (
    VectorIndexBuildConfig,
    VectorIndexValidationResult,
    ensure_vector_index,
)


@dataclass(frozen=True)
class TileSourcePreparation:
    """Declare how one prospective tile source prepares its raster index."""

    source: TileSourceConfig
    enabled: bool = True
    image_glob: str = "*.tif"
    output_srs_path: Path = LUNAR_GEOGRAPHIC_WKT_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.source, TileSourceConfig):
            raise TypeError("source must be a TileSourceConfig.")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        image_glob = str(self.image_glob).strip()
        if not image_glob:
            raise ValueError("image_glob must not be empty.")
        object.__setattr__(self, "image_glob", image_glob)
        object.__setattr__(self, "output_srs_path", Path(self.output_srs_path))

    def index_config(self) -> VectorIndexBuildConfig:
        """Return the builder contract derived from the prospective source."""
        return VectorIndexBuildConfig(
            data_dir=self.source.data_dir,
            index_path=self.source.index_path,
            image_glob=self.image_glob,
            layer_name=self.source.index_layer,
            location_field=self.source.location_field,
            output_srs_path=self.output_srs_path,
        )


@dataclass(frozen=True)
class TilePreparationResult:
    """A low-level tile configuration and its validated source indexes."""

    config: TileConfig
    indexes: tuple[VectorIndexValidationResult, ...]


def prepare_tile_config(
    *,
    output_dir: str | Path,
    zoom_level: int,
    sources: Sequence[TileSourcePreparation],
    debug: bool = False,
    logger: logging.Logger | None = None,
    stdout: TextIO | None = None,
) -> TilePreparationResult:
    """Prepare enabled indexes, then assemble the read-only tiling config.

    Disabled preparations are not discovered, indexed, or validated. The
    resulting :class:`TileConfig` can be passed to the existing low-level
    ``create_tiles_*`` functions, which remain free of index mutations.
    """
    preparations = tuple(sources)
    if any(not isinstance(item, TileSourcePreparation) for item in preparations):
        raise TypeError("sources must contain TileSourcePreparation objects.")
    enabled = tuple(item for item in preparations if item.enabled)
    if not enabled:
        raise ValueError("At least one tile source must be enabled.")
    names = [item.source.name for item in enabled]
    if len(set(names)) != len(names):
        raise ValueError("Enabled tile source names must be unique.")

    index_results = tuple(
        ensure_vector_index(
            item.index_config(),
            logger=logger,
            stdout=stdout,
        )
        for item in enabled
    )
    config = TileConfig(
        output_dir=Path(output_dir),
        zoom_level=zoom_level,
        sources=tuple(item.source for item in enabled),
        debug=debug,
    )
    return TilePreparationResult(config=config, indexes=index_results)


__all__ = [
    "TilePreparationResult",
    "TileSourcePreparation",
    "prepare_tile_config",
]
