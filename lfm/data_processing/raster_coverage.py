"""Metadata-only lunar raster coverage for the WAC/NAC AOI discovery notebook.

Workers live in an importable module so a spawn process pool works in Jupyter.
Coverage describes the affine raster perimeter, including any NoData margins.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from heapq import nsmallest
import multiprocessing
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import perf_counter

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


def _pair_key(pair):
    return (-pair["wac_coverage_percent"], pair["wac_filename"], pair["nac_filename"])


def _build_pair_context(nac_entries, lunar_wkt, limit):
    started = perf_counter()
    radius = CRS.from_wkt(lunar_wkt).ellipsoid.semi_major_metre
    geod = Geod(a=radius, b=radius)
    nac_names = [name for name, geometry in nac_entries if geometry]
    nac_geometries = [load_wkb(geometry, hex=True) for _, geometry in nac_entries if geometry]
    indexed_geometries = [translate(geometry, xoff=offset)
                          for geometry in nac_geometries for offset in (-360, 0, 360)]
    tree = STRtree(indexed_geometries)
    return {"geod": geod, "nac_names": nac_names, "geometries": indexed_geometries,
            "tree": tree, "limit": limit, "index_seconds": perf_counter() - started}


_PAIR_WORKER_CONTEXT = None


def _initialize_pair_worker(nac_cache_path, lunar_wkt, limit):
    """Build the read-only NAC index once in each spawned process."""
    global _PAIR_WORKER_CONTEXT
    # This pickle is created by the parent for this invocation, never supplied
    # by a user. A small initializer argument lets all workers start together;
    # sending a large NAC inventory through spawn's pipes serializes startup.
    with open(nac_cache_path, "rb") as stream:
        nac_entries = pickle.load(stream)
    _PAIR_WORKER_CONTEXT = _build_pair_context(nac_entries, lunar_wkt, limit)


def _compare_wac_footprint(wac_item, context):
    """Return only positive overlaps and this WAC's best pairs, never a dense row."""
    started = perf_counter()
    wac_name, geometry_hex = wac_item
    wac_geometry = load_wkb(geometry_hex, hex=True)
    geod = context["geod"]
    wac_area = _surface_area(wac_geometry, geod)
    if wac_area <= 0:
        raise ValueError(f"Nonpositive lunar surface area for {wac_name}")
    overlaps = {}

    def pairs():
        # Each ordinary scene spans <180 degrees, so at most one shifted copy
        # of a NAC scene can have positive area in this WAC.
        for index in context["tree"].query(wac_geometry, predicate="intersects"):
            area = _surface_area(wac_geometry.intersection(context["geometries"][index]), geod)
            if area > 0:
                nac_name = context["nac_names"][index // 3]
                percent = float(np.clip(100 * area / wac_area, 0, 100))
                overlaps[nac_name] = percent
                yield {"wac_filename": wac_name, "nac_filename": nac_name,
                       "wac_coverage_percent": percent, "shared_area_km2": area / 1_000_000,
                       "wac_area_km2": wac_area / 1_000_000}

    top_pairs = nsmallest(context["limit"], pairs(), key=_pair_key)
    return {"wac_filename": wac_name, "overlaps": overlaps, "top_pairs": top_pairs,
            "intersection_seconds": perf_counter() - started,
            "worker_pid": os.getpid(), "index_seconds": context["index_seconds"]}


def _compare_wac_worker(wac_item):
    return _compare_wac_footprint(wac_item, _PAIR_WORKER_CONTEXT)


def rank_raster_pairs(
    wac_footprints, nac_footprints, *, lunar_wkt, limit=5, max_workers=1,
    progress=False, progress_callback=None,
):
    """Rank full-scene pairs by percentage of each WAC footprint covered by NAC.

    Inputs map filenames to geographic WKB hex or None for failed inspection.
    Spawn workers each build one NAC spatial index, then compare WAC files.
    At most twice the worker count of jobs are pending; only positive overlaps
    and each WAC's top pairs cross process boundaries. The parent constructs
    the complete M x N matrix with 0.0 for disjoint pairs and None for failures.

    max_workers=1 keeps a serial path; the notebook explicitly requests 10.
    Results and tie ordering are independent of worker completion order.
    Timings distinguish total elapsed time, parent matrix work, and cumulative
    worker index/intersection time (worker seconds are not wall-clock stages).
    progress_callback(completed, total) runs only in the parent, initially and
    after each completed WAC, so notebook progress bars need not be pickled.
    Requires Shapely >= 2. No AOI restriction is applied.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")
    started = perf_counter()
    wac_entries = sorted(wac_footprints.items())
    nac_entries = sorted(nac_footprints.items())
    tasks = [entry for entry in wac_entries if entry[1]]
    has_nac = any(geometry for _, geometry in nac_entries)
    workers_used = min(max_workers, len(tasks)) if has_nac else 0
    overlap_percent_by_wac = {}
    top_pairs = []
    pair_count = 0
    completed = 0
    intersection_seconds = 0.0
    index_times = {}
    matrix_started = perf_counter()
    valid_row = {name: 0.0 if geometry else None for name, geometry in nac_entries}
    failed_row = dict.fromkeys(valid_row)
    matrix_seconds = perf_counter() - matrix_started
    last_progress = started

    def report_progress(force=False):
        nonlocal last_progress
        now = perf_counter()
        if progress and (force or now - last_progress >= 5):
            print(f"Ranking: {completed:,}/{len(wac_entries):,} WAC complete; "
                  f"{pair_count:,} positive pairs; {now - started:.1f}s elapsed; "
                  f"{matrix_seconds:.1f}s constructing matrix", flush=True)
            last_progress = now

    def collect(result, *, readable=True):
        nonlocal completed, pair_count, top_pairs, matrix_seconds, intersection_seconds
        row_started = perf_counter()
        row = (valid_row if readable else failed_row).copy()
        row.update(result["overlaps"])
        overlap_percent_by_wac[result["wac_filename"]] = row
        matrix_seconds += perf_counter() - row_started
        pair_count += len(result["overlaps"])
        top_pairs = nsmallest(limit, top_pairs + result["top_pairs"], key=_pair_key)
        if "worker_pid" in result:
            intersection_seconds += result["intersection_seconds"]
            index_times[result["worker_pid"]] = result["index_seconds"]
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, len(wac_entries))
        report_progress(force=completed == 1)

    if progress:
        print(f"Ranking {len(wac_entries):,} WAC against {len(nac_entries):,} NAC "
              f"with {workers_used} worker(s); building each worker's NAC index once.", flush=True)
    report_progress(force=True)
    if progress_callback is not None:
        progress_callback(0, len(wac_entries))
    for name, geometry in wac_entries:
        if not geometry or not has_nac:
            collect({"wac_filename": name, "overlaps": {}, "top_pairs": []}, readable=bool(geometry))

    if workers_used == 1:
        context = _build_pair_context(nac_entries, lunar_wkt, limit)
        for task in tasks:
            collect(_compare_wac_footprint(task, context))
    elif workers_used > 1:
        # Keep the cache until the pool has joined; cleanup also runs on errors.
        with TemporaryDirectory(prefix="lfm_ranking_") as cache_dir:
            nac_cache_path = str(Path(cache_dir) / "nac_footprints.pkl")
            with open(nac_cache_path, "wb") as stream:
                pickle.dump(nac_entries, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with ProcessPoolExecutor(
                max_workers=workers_used, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_pair_worker, initargs=(nac_cache_path, lunar_wkt, limit),
            ) as pool:
                remaining = iter(tasks)
                pending = {}

                def submit_next():
                    task = next(remaining, None)
                    if task is not None:
                        pending[pool.submit(_compare_wac_worker, task)] = task[0]

                for _ in range(2 * workers_used):
                    submit_next()
                try:
                    while pending:
                        done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                        for future in done:
                            name = pending.pop(future)
                            try:
                                collect(future.result())
                            except Exception as exc:
                                raise RuntimeError(f"Overlap ranking failed for WAC {name}: {exc}") from exc
                            submit_next()
                        report_progress()
                finally:
                    for future in pending:
                        future.cancel()

    # Reorder references only; do not duplicate the dense rows.
    matrix_started = perf_counter()
    overlap_percent_by_wac = {name: overlap_percent_by_wac[name] for name, _ in wac_entries}
    matrix_seconds += perf_counter() - matrix_started
    report_progress(force=True)
    timings = {"ranking_wall_seconds": perf_counter() - started,
               "matrix_build_seconds": matrix_seconds,
               "worker_intersection_seconds": intersection_seconds,
               "worker_index_seconds": sum(index_times.values())}

    return {"top_pairs": top_pairs, "overlap_percent_by_wac": overlap_percent_by_wac,
            "positive_pair_count": pair_count, "timings": timings,
            "workers_used": workers_used}
