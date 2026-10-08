"""Canonical output-band contract for the built-in WAC modality."""

WAC_VIS_BAND_NAMES = (
    "vis.mos-0",
    "vis.mos-1",
    "vis.mos-2",
    "vis.mos-3",
    "vis.mos-4",
)

WAC_UV_BAND_NAMES = (
    "uv.mos-0",
    "uv.mos-1",
)

WAC_BAND_NAMES = (*WAC_VIS_BAND_NAMES, *WAC_UV_BAND_NAMES)


__all__ = [
    "WAC_BAND_NAMES",
    "WAC_UV_BAND_NAMES",
    "WAC_VIS_BAND_NAMES",
]
