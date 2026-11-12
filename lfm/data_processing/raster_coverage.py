"""Metadata-only lunar raster coverage for the WAC/NAC AOI discovery notebook.

Workers live in an importable module so a spawn process pool works in Jupyter.
Coverage describes the affine raster perimeter, including any NoData margins.
"""

from __future__ import annotations

from functools import lru_cache
from heapq import nsmallest
from pathlib import Path

import numpy as np
from pyproj import CRS, Geod, Transformer
import rasterio
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import transform
from shapely.strtree import STRtree
from shapely.wkb import loads as load_wkb


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


def _read_lunar_footprint(filename, geographic, edge_segments):
    """Read and densify a raster's affine perimeter in lunar longitude/latitude."""
    if not isinstance(edge_segments, int) or edge_segments < 4:
        raise ValueError("edge_segments must be an integer >= 4.")
    with rasterio.open(filename) as dataset:
        if dataset.crs is None:
            raise ValueError("Raster has no CRS.")
        if dataset.transform.is_identity:
            raise ValueError("Raster lacks a usable affine geotransform.")
        corners = [(0, 0), (dataset.width, 0),
                   (dataset.width, dataset.height), (0, dataset.height)]
        boundary = _edge_points(corners, edge_segments)
        xy = [dataset.transform * point for point in boundary]
        to_geographic = Transformer.from_crs(dataset.crs, geographic, always_xy=True)
        longitude, latitude = to_geographic.transform(*zip(*xy), errcheck=True)
    longitude = np.rad2deg(np.unwrap(np.deg2rad(longitude)))
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Nonfinite footprint coordinates.")
    if np.ptp(longitude) >= 180 or max(np.abs(latitude)) >= 90:
        raise ValueError("Polar/global footprint requires a different coverage method.")
    footprint = Polygon(zip(longitude, latitude))
    if footprint.is_empty or not footprint.is_valid or footprint.area <= 0:
        raise ValueError("Invalid transformed raster footprint.")
    # Keep ordinary scenes contiguous, including those crossing the antimeridian.
    offset = -360 * np.floor((footprint.centroid.x + 180) / 360)
    return translate(footprint, xoff=offset)


def inspect_raster_footprint(path, *, lunar_wkt, edge_segments=64):
    """Return a full geographic footprint, without clipping to a search AOI.

    WKB hex is safe to pass between processes and save in JSON. This is an
    affine footprint, including NoData margins. Longitude is unwrapped across
    the antimeridian. Pole-enclosing/global footprints are reported as errors.
    """
    filename = str(Path(path))
    try:
        footprint = _read_lunar_footprint(filename, CRS.from_wkt(lunar_wkt), edge_segments)
        return {"filename": filename, "footprint_wkb_hex": footprint.wkb_hex, "error": None}
    except Exception as exc:
        return {"filename": filename, "footprint_wkb_hex": None,
                "error": f"{type(exc).__name__}: {exc}"}


def inspect_raster_coverage(path, *, aoi_nwse, lunar_wkt, edge_segments=64):
    """Return filename, precise AOI coverage percent, and any per-file error.

    No bands or masks are read. Edges follow the affine pixel perimeter,
    including rotation, and are densified before reprojection. Geographic
    intersection precedes equal-area projection to avoid projecting remote
    scenes near the antipode of the AOI-centered projection. Ordinary WAC/NAC
    scene footprints are supported; polar/global footprints are not.
    """
    filename = str(Path(path))
    try:
        geographic, aoi, to_area, aoi_area, center_lon = coverage_context(
            tuple(aoi_nwse), lunar_wkt, edge_segments
        )
        footprint = _read_lunar_footprint(filename, geographic, edge_segments)
        # Match 0..360 and -180..180 source conventions to this AOI.
        offset = 360 * round((center_lon - footprint.centroid.x) / 360)
        footprint = translate(footprint, xoff=offset)
        overlap = footprint.intersection(aoi)
        area_geometry = transform(to_area.transform, overlap)
        area = area_geometry.area
        percent = float(np.clip(100 * area / aoi_area, 0, 100))
        return {"filename": filename, "overlap_percent": percent, "error": None}
    except Exception as exc:
        # A malformed file must not abort a collection scan or masquerade as 0%.
        return {"filename": filename, "overlap_percent": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _surface_area(geometry, geod):
    """Positive lunar surface area, ignoring line/point-only contacts."""
    if geometry.is_empty:
        return 0.0
    if geometry.geom_type == "Polygon":
        return abs(geod.geometry_area_perimeter(orient(geometry, sign=1.0))[0])
    if hasattr(geometry, "geoms"):
        return sum(_surface_area(part, geod) for part in geometry.geoms)
    return 0.0


def rank_raster_pairs(wac_footprints, nac_footprints, *, lunar_wkt, limit=5):
    """Rank full-scene pairs by percentage of each WAC footprint covered by NAC.

    Inputs map filenames to geographic ``footprint_wkb_hex`` from inspection.
    Surface areas use the repository lunar sphere, not square degrees. A
    spatial index with shifted longitude copies handles antimeridian scenes
    and skips expensive intersections for disjoint pairs. The returned nested
    dictionary has M x N entries: 0.0 for disjoint readable files, None when
    either footprint is None (failed inspection), and precise percentages for
    overlaps. Requires Shapely >= 2. No AOI restriction is applied.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    radius = CRS.from_wkt(lunar_wkt).ellipsoid.semi_major_metre
    geod = Geod(a=radius, b=radius)
    nac_entries = sorted(nac_footprints.items())
    nac_names = [name for name, geometry in nac_entries if geometry]
    nac_geometries = [load_wkb(nac_footprints[name], hex=True) for name in nac_names]
    indexed_geometries = [translate(geometry, xoff=offset)
                          for geometry in nac_geometries for offset in (-360, 0, 360)]
    tree = STRtree(indexed_geometries)
    overlap_percent_by_wac = {}
    pair_count = 0

    def pairs():
        nonlocal pair_count
        for wac_name, geometry_hex in sorted(wac_footprints.items()):
            overlap_percent_by_wac[wac_name] = {
                nac_name: 0.0 if geometry_hex and nac_hex else None
                for nac_name, nac_hex in nac_entries
            }
            if not geometry_hex:
                continue
            wac_geometry = load_wkb(geometry_hex, hex=True)
            wac_area = _surface_area(wac_geometry, geod)
            if wac_area <= 0:
                raise ValueError(f"Nonpositive lunar surface area for {wac_name}")
            # Each ordinary scene spans <180 degrees, so at most one shifted
            # copy of a given NAC scene can have positive area in this WAC.
            for index in tree.query(wac_geometry, predicate="intersects"):
                area = _surface_area(wac_geometry.intersection(indexed_geometries[index]), geod)
                if area > 0:
                    pair_count += 1
                    pair = {
                        "wac_filename": wac_name,
                        "nac_filename": nac_names[index // 3],
                        "wac_coverage_percent": float(np.clip(100 * area / wac_area, 0, 100)),
                        "shared_area_km2": area / 1_000_000,
                        "wac_area_km2": wac_area / 1_000_000,
                    }
                    overlap_percent_by_wac[wac_name][pair["nac_filename"]] = pair["wac_coverage_percent"]
                    yield pair

    def pair_key(pair):
        return (-pair["wac_coverage_percent"], pair["wac_filename"], pair["nac_filename"])

    top_pairs = nsmallest(limit, pairs(), key=pair_key)
    return {"top_pairs": top_pairs, "overlap_percent_by_wac": overlap_percent_by_wac,
            "positive_pair_count": pair_count}
