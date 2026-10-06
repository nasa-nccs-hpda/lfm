#!/usr/bin/env python3
"""Read-only search for likely NAC TIFFs with lunar polar raster footprints.

No imagery pixels are read. Bounds describe georeferenced coverage, not valid
data coverage. NAC identity is a path/name heuristic, not instrument metadata.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path("/explore/nobackup/projects/lfm")
NAC_PATTERN = r"nac|(?:^|[/_])M\d+[LR][EC](?:[._/]|$)"


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
        for pole in (-90, 90):
            try:
                x, y, *_ = reverse.TransformPoint(0, pole)
                col, row = gdal.ApplyGeoTransform(inverse, x, y)
                if math.isfinite(col) and math.isfinite(row) and 0 <= col <= width and 0 <= row <= height:
                    poles.append(pole)
            except RuntimeError:
                pass  # An opposite pole may be outside the projection domain.
        latitudes = [lat for _, lat in coordinates] + poles
        north, south = max(latitudes), min(latitudes)
        return dict(path=str(path), latitude_min=south, latitude_max=north,
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
    args = parser.parse_args()
    if args.max_depth < 1 or args.limit < 0:
        parser.error("max-depth must be positive and limit nonnegative")
    try:
        pattern = None if args.all_tifs else re.compile(args.name_regex, re.I)
    except re.error as exc:
        parser.error(str(exc))
    if args.report and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    from osgeo import gdal
    gdal.UseExceptions()
    errors, matches, inspected, stopped = [], [], 0, False
    print(f"Roots: {', '.join(map(str, args.roots))}\nMax depth: {args.max_depth}\n"
          f"Name filter: {pattern.pattern if pattern else 'ALL TIFFS'}", flush=True)
    for path in candidates(args.roots, args.max_depth, pattern, errors):
        inspected += 1
        try:
            item = inspect_raster(path)
        except Exception as exc:
            errors.append(dict(path=str(path), error=str(exc)))
            print(f"SKIP {path}: {exc}", file=sys.stderr, flush=True)
            continue
        if item["hemispheres"] and (args.hemisphere == "both" or args.hemisphere in item["hemispheres"]):
            matches.append(item)
            print(f"MATCH {path}\n  latitude {item['latitude_min']:.6f} .. {item['latitude_max']:.6f}; "
                  f"{item['width']} x {item['height']}; {item['bands']} band(s); {item['crs_name']}", flush=True)
            if args.limit and len(matches) >= args.limit:
                stopped = True
                break
        if inspected % 100 == 0:
            print(f"Inspected {inspected} candidate TIFFs; {len(matches)} matches", flush=True)
    print(f"Done: {inspected} candidates inspected; {len(matches)} matches; {len(errors)} errors; limit reached={stopped}")
    print("Footprints are sampled metadata coverage, not proof of valid imagery. NAC identity is inferred from names.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(dict(roots=list(map(str, args.roots)), max_depth=args.max_depth,
                           name_regex=None if pattern is None else pattern.pattern,
                           hemisphere=args.hemisphere, stopped_at_limit=stopped,
                           inspected=inspected, matches=matches, errors=errors), stream, indent=2, allow_nan=False)
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
