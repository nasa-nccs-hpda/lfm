#!/usr/bin/env python3
"""Read-only search for likely NAC TIFFs with lunar polar raster footprints.

No imagery pixels are read. Bounds describe georeferenced coverage, not valid
data coverage. NAC identity is a path/name heuristic, not instrument metadata.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path("/explore/nobackup/projects/lfm")
NAC_PATTERN = r"nac|(?:^|[/_])M\d+[LR][EC](?:[._/]|$)"


def longitude_span(longitudes):
    """Shortest circular span, so crossing 180 degrees does not inflate width."""
    values = sorted(lon % 360 for lon in longitudes)
    gaps = [b - a for a, b in zip(values, values[1:])]
    return 360 - max(gaps + [values[0] + 360 - values[-1]])


def worker_count(override=None):
    value = int(os.environ.get("SLURM_CPUS_PER_TASK", "1") if override is None else override)
    if value < 1:
        raise ValueError("workers must be positive")
    return value


def inspect_batch(paths):
    """Workers only read metadata; the parent owns reporting and output files."""
    results, errors = [], []
    for path in paths:
        try:
            results.append(inspect_raster(path))
        except Exception as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    return results, errors


def inspection_waves(paths, workers, batch_size):
    """Bound pending work to one batch per worker; finish a wave before yielding."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch size must be positive")
    if workers == 1:
        for start in range(0, len(paths), batch_size):
            yield inspect_batch(paths[start:start + batch_size])
        return
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(paths), workers * batch_size):
            futures = [pool.submit(inspect_batch, paths[i:i + batch_size])
                       for i in range(start, min(start + workers * batch_size, len(paths)), batch_size)]
            results, errors = [], []
            for future in futures:
                batch_results, batch_errors = future.result()
                results.extend(batch_results)
                errors.extend(batch_errors)
            yield results, errors


def matches_filters(item, hemisphere, min_longitude_span):
    return (bool(item["hemispheres"])
            and (hemisphere == "both" or hemisphere in item["hemispheres"])
            and item["longitude_span"] >= min_longitude_span)


def candidates(roots, max_depth, pattern, errors):
    """Root files have depth 1 (find -maxdepth semantics); no symlink descent."""
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            errors.append({"path": str(root), "error": "Missing or unreadable search directory"})
            continue

        def onerror(exc):
            errors.append({"path": str(exc.filename), "error": str(exc)})

        for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            relative_depth = len(Path(directory).relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
            if relative_depth + 1 >= max_depth:
                dirs[:] = []
            if relative_depth + 1 > max_depth:
                continue
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if pattern is not None and not pattern.search(str(path)):
                    continue
                identity = path.resolve()
                if identity not in seen:
                    seen.add(identity)
                    yield path


def inspect_raster(path, edge_samples=65):
    from osgeo import gdal, osr

    gdal.UseExceptions()
    ds = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
    if ds is None:
        raise ValueError("GDAL could not open raster")
    try:
        source = ds.GetSpatialRef()
        affine = ds.GetGeoTransform(can_return_null=True)
        if source is None or affine is None:
            raise ValueError("Missing CRS or affine georeferencing")
        if not all(math.isclose(value, 1737400, rel_tol=0, abs_tol=.01)
                   for value in (source.GetSemiMajor(), source.GetSemiMinor())):
            raise ValueError("Not the repository's spherical lunar CRS (radius 1737400 m)")
        geographic = osr.SpatialReference()
        geographic.ImportFromWkt((REPO_ROOT / "TMS/IAU_30100_2015.wkt").read_text())
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        with osr.ExceptionMgr(useExceptions=False):
            forward = osr.CoordinateTransformation(source, geographic)
            reverse = osr.CoordinateTransformation(geographic, source)
        if forward is None or reverse is None:
            raise ValueError("Could not construct lunar coordinate transformation")
        width, height = ds.RasterXSize, ds.RasterYSize
        pixels = []
        for i in range(edge_samples):
            fraction = i / (edge_samples - 1)
            pixels.extend(((width * fraction, 0), (width * fraction, height),
                           (0, height * fraction), (width, height * fraction)))
        pixels.extend((width * x / 4, height * y / 4) for x in range(1, 4) for y in range(1, 4))
        coordinates = []
        for col, row in pixels:
            x, y = gdal.ApplyGeoTransform(affine, col, row)
            lon, lat, *_ = forward.TransformPoint(x, y)
            if not math.isfinite(lon) or not math.isfinite(lat) or not -90 <= lat <= 90:
                raise ValueError("Nonfinite or invalid footprint transformation")
            coordinates.append((lon, lat))
        # Perimeter-only sampling misses poles contained inside a rectangle.
        inverse = gdal.InvGeoTransform(affine)
        if inverse is None:
            raise ValueError("Noninvertible raster affine")
        poles = []
        interior_pole = False
        for pole in (-90, 90):
            try:
                x, y, *_ = reverse.TransformPoint(0, pole)
                col, row = gdal.ApplyGeoTransform(inverse, x, y)
                if math.isfinite(col) and math.isfinite(row) and 0 <= col <= width and 0 <= row <= height:
                    poles.append(pole)
                    interior_pole |= 0 < col < width and 0 < row < height
            except RuntimeError:
                pass  # An opposite pole may be outside the projection domain.
        latitudes = [lat for _, lat in coordinates] + poles
        north, south = max(latitudes), min(latitudes)
        return dict(path=str(path), latitude_min=south, latitude_max=north,
                    longitude_span=360.0 if interior_pole else longitude_span([lon for lon, _ in coordinates]),
                    hemispheres=[name for name, matches in (("north", north >= 82), ("south", south <= -82)) if matches],
                    width=width, height=height, bands=ds.RasterCount,
                    crs_name=source.GetName(), crs_wkt=source.ExportToWkt(),
                    transform=list(affine), contains_poles=poles,
                    sample_center_lon_lat=list(forward.TransformPoint(
                        *gdal.ApplyGeoTransform(affine, width / 2, height / 2))[:2]))
    finally:
        ds = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[DEFAULT_ROOT / p for p in ("data", "processed_data", "rawdata")])
    parser.add_argument("--max-depth", type=int, default=10, help="Root files have depth 1; default 10")
    parser.add_argument("--hemisphere", choices=("north", "south", "both"), default="both")
    parser.add_argument("--name-regex", default=NAC_PATTERN, help="Case-insensitive path filter for likely NAC products")
    parser.add_argument("--all-tifs", action="store_true", help="Disable name filter; results are NOT necessarily NAC")
    parser.add_argument("--limit", type=int, default=20, help="Stop after this many matches; 0 scans all candidates")
    parser.add_argument("--report", type=Path, help="Optional NEW JSON report (refuses overwrite)")
    parser.add_argument("--workers", type=int, help="Default: SLURM_CPUS_PER_TASK, or 1")
    parser.add_argument("--batch-size", type=int, default=32, help="TIFFs per worker batch (default 32)")
    parser.add_argument("--min-longitude-span", type=float, default=10,
                        help="Minimum whole-footprint longitude span in degrees (default 10)")
    args = parser.parse_args()
    if args.max_depth < 1 or args.limit < 0:
        parser.error("max-depth must be positive and limit nonnegative")
    if args.batch_size < 1 or not 0 <= args.min_longitude_span <= 360:
        parser.error("batch-size must be positive; min-longitude-span must be between 0 and 360")
    try:
        workers = worker_count(args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        pattern = None if args.all_tifs else re.compile(args.name_regex, re.I)
    except re.error as exc:
        parser.error(str(exc))
    if args.report and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    errors, matches, inspected, stopped = [], [], 0, False
    print(f"Roots: {', '.join(map(str, args.roots))}\nMax depth: {args.max_depth}\n"
          f"Name filter: {pattern.pattern if pattern else 'ALL TIFFS'}", flush=True)
    paths = sorted(candidates(args.roots, args.max_depth, pattern, errors))
    print(f"Discovered {len(paths)} TIFFs; workers={workers}; batch-size={args.batch_size}; "
          f"minimum longitude span={args.min_longitude_span} degrees", flush=True)
    waves = inspection_waves(paths, workers, args.batch_size)
    try:
        for results, failures in waves:
            inspected += len(results) + len(failures)
            errors.extend(failures)
            for failure in failures:
                print(f"SKIP {failure['path']}: {failure['error']}", file=sys.stderr, flush=True)
            for item in results:
                if matches_filters(item, args.hemisphere, args.min_longitude_span):
                    if args.limit and len(matches) >= args.limit:
                        continue
                    matches.append(item)
                    print(f"MATCH {item['path']}\n  latitude {item['latitude_min']:.6f} .. "
                          f"{item['latitude_max']:.6f}; longitude span {item['longitude_span']:.6f}; "
                          f"{item['width']} x {item['height']}; {item['bands']} band(s); {item['crs_name']}", flush=True)
            print(f"Inspected {inspected} candidate TIFFs; {len(matches)} matches", flush=True)
            if args.limit and len(matches) >= args.limit:
                stopped = True
                break
    finally:
        waves.close()
    print(f"Done: {inspected} candidates inspected; {len(matches)} matches; {len(errors)} errors; limit reached={stopped}")
    print("Footprints are sampled metadata coverage, not proof of valid imagery. NAC identity is inferred from names.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(dict(roots=list(map(str, args.roots)), max_depth=args.max_depth,
                           name_regex=None if pattern is None else pattern.pattern,
                           hemisphere=args.hemisphere, stopped_at_limit=stopped,
                           workers=workers, batch_size=args.batch_size, discovered=len(paths),
                           min_longitude_span=args.min_longitude_span,
                           inspected=inspected, matches=matches, errors=errors), stream, indent=2, allow_nan=False)
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
