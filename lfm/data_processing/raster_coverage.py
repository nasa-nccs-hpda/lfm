"""Metadata-only lunar raster coverage for the NAC AOI discovery notebook.

Workers live in an importable module so a spawn process pool works in Jupyter.
Coverage describes the affine raster perimeter, including any NoData margins.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import transform


def _edge_points(corners, edge_segments):
    """Sample straight edges before a nonlinear coordinate transformation."""
    points = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        for fraction in np.linspace(0, 1, edge_segments, endpoint=False):
            points.append(tuple(a + fraction * (b - a) for a, b in zip(start, end)))
    return points + points[:1]


@lru_cache(maxsize=8)
def coverage_context(aoi_nwse, lunar_wkt, edge_segments=64):
    """Prepare a small, nonpolar, non-wrapping AOI and lunar equal-area CRS.

    AOI order is north, west, south, east, in IAU:30100 degrees. The notebook
    loads ``lunar_wkt`` from the repository's TMS/IAU_30100_2015.wkt.
    """
    north, west, south, east = aoi_nwse
    if not all(np.isfinite(aoi_nwse)) or not (
        -82 < south < north < 82 and -180 <= west < east <= 180
        and east - west < 180
    ):
        raise ValueError("Use a nonpolar, non-wrapping AOI in north/west/south/east order.")
    if not isinstance(edge_segments, int) or edge_segments < 4:
        raise ValueError("edge_segments must be an integer >= 4.")
    geographic = CRS.from_wkt(lunar_wkt)
    radius = geographic.ellipsoid.semi_major_metre
    center_lon, center_lat = (west + east) / 2, (north + south) / 2
    equal_area = CRS.from_proj4(
        f"+proj=laea +lat_0={center_lat} +lon_0={center_lon} "
        f"+R={radius} +units=m +no_defs +type=crs"
    )
    to_area = Transformer.from_crs(geographic, equal_area, always_xy=True)
    aoi = Polygon(_edge_points(
        [(west, north), (east, north), (east, south), (west, south)], edge_segments
    ))
    area_m2 = transform(to_area.transform, aoi).area
    return geographic, aoi, to_area, area_m2, center_lon


def inspect_raster_coverage(path, *, aoi_nwse, lunar_wkt, edge_segments=64):
    """Return filename, precise AOI coverage percent, and any per-file error.

    No bands or masks are read. Edges follow the affine pixel perimeter,
    including rotation, and are densified before reprojection. Geographic
    intersection precedes equal-area projection to avoid projecting remote
    scenes near the antipode of the AOI-centered projection. Ordinary NAC
    scene footprints are supported; polar/global footprints are not.
    """
    filename = str(Path(path))
    try:
        geographic, aoi, to_area, aoi_area, center_lon = coverage_context(
            tuple(aoi_nwse), lunar_wkt, edge_segments
        )
        with rasterio.open(filename) as dataset:
            if dataset.crs is None:
                raise ValueError("Raster has no CRS.")
            if dataset.transform.is_identity:
                raise ValueError("Raster lacks a usable affine geotransform.")
            corners = [(0, 0), (dataset.width, 0),
                       (dataset.width, dataset.height), (0, dataset.height)]
            boundary = _edge_points(corners, edge_segments)
            xy = [dataset.transform * point for point in boundary]
            to_geographic = Transformer.from_crs(
                dataset.crs, geographic, always_xy=True
            )
            longitude, latitude = to_geographic.transform(
                *zip(*xy), errcheck=True
            )
        longitude = np.rad2deg(np.unwrap(np.deg2rad(longitude)))
        if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
            raise ValueError("Nonfinite footprint coordinates.")
        if np.ptp(longitude) >= 180 or max(np.abs(latitude)) >= 90:
            raise ValueError("Polar/global footprint requires a different coverage method.")
        footprint = Polygon(zip(longitude, latitude))
        if footprint.is_empty or not footprint.is_valid or footprint.area <= 0:
            raise ValueError("Invalid transformed raster footprint.")
        # Match 0..360 and -180..180 source conventions to this AOI.
        offset = 360 * round((center_lon - footprint.centroid.x) / 360)
        footprint = translate(footprint, xoff=offset)
        overlap = footprint.intersection(aoi)
        area = 0.0 if overlap.is_empty else transform(to_area.transform, overlap).area
        percent = float(np.clip(100 * area / aoi_area, 0, 100))
        return {"filename": filename, "overlap_percent": percent, "error": None}
    except Exception as exc:
        # A malformed file must not abort a collection scan or masquerade as 0%.
        return {"filename": filename, "overlap_percent": None,
                "error": f"{type(exc).__name__}: {exc}"}
