"""Canonical static-source configuration without model-training dependencies."""

from pathlib import Path

from .tiling_config import BandNoDataOverride, TileSourceConfig
from .static_band_contract import (
    MINIRF_SOURCE_NODATA, MINIRF_SOURCE_NODATA_BANDS,
    STATIC_BAND_NAMES, STATIC_OUTPUT_NODATA,
)


def make_static_source(
    *,
    data_dir: str | Path,
    index_path: str | Path,
    index_layer: str | None = None,
    location_field: str = "location",
    required: bool = True,
) -> TileSourceConfig:
    """Create the canonical 63-band static lunar tiling source config.

    Tiling retains uncovered channels as -32768 after verifying indexed band
    metadata; required still rejects genuinely missing or unreadable inputs.
    """
    return TileSourceConfig(
        name="static",
        data_dir=Path(data_dir),
        index_path=Path(index_path),
        index_layer=index_layer,
        location_field=location_field,
        selection_mode="all_intersecting",
        band_names=STATIC_BAND_NAMES,
        resampling="bilinear",
        output_nodata=STATIC_OUTPUT_NODATA,
        band_nodata_overrides=tuple(
            BandNoDataOverride(
                band_name=name,
                source_value=MINIRF_SOURCE_NODATA,
            )
            for name in MINIRF_SOURCE_NODATA_BANDS
        ),
        required=required,
    )


__all__ = ["make_static_source"]
