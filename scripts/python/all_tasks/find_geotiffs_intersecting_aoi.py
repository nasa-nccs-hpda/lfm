#!/usr/bin/env python3
"""List GeoTIFFs from a directory whose footprints intersect a lunar AOI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import TileSourceConfig, query_source_index  # noqa: E402


DEFAULT_INDEX_NAMES = ("output_index.shp", "output_index.gpkg")
GEOTIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find indexed or directly scanned GeoTIFF footprints that "
            "intersect a lunar latitude/longitude AOI."
        )
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing candidate GeoTIFFs.",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        required=True,
        metavar=("UL_LAT", "UL_LON", "LR_LAT", "LR_LON"),
        help="AOI bounds in lunar latitude/longitude.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help=(
            "Existing .shp or .gpkg raster index. In auto mode, the script "
            "otherwise looks for DATA_DIR/output_index.shp or .gpkg."
        ),
    )
    parser.add_argument("--index-layer", default=None)
    parser.add_argument("--location-field", default="location")
    parser.add_argument(
        "--mode",
        choices=("auto", "index", "scan"),
        default="auto",
        help=(
            "Use an existing index, scan raster metadata directly, or choose "
            "the index when available and scan otherwise (default: auto)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search DATA_DIR when direct raster scanning is used.",
    )
    parser.add_argument(
        "--edge-samples",
        type=int,
        default=21,
        help="Footprint samples per raster edge in scan mode (default: 21).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def explore_path(path: Path) -> Path:
    text = str(path)
    return Path(
        text.replace(
            "/panfs/ccds02/nobackup/",
            "/explore/nobackup/",
            1,
        )
        if text.startswith("/panfs/ccds02/nobackup/")
        else text
    )


def normalize_longitude_interval(west: float, east: float) -> tuple[float, float]:
    west = float(west)
    east = float(east)
    while east < west:
        east += 360.0
    while west < -180.0:
        west += 360.0
        east += 360.0
    while west >= 180.0:
        west -= 360.0
        east -= 360.0
    if east - west > 360.0:
        raise ValueError("AOI longitude span must not exceed 360 degrees.")
    return west, east


def conventional_longitude_intervals(
    west: float,
    east: float,
) -> list[tuple[float, float]]:
    west, east = normalize_longitude_interval(west, east)
    if east <= 180.0:
        return [(west, east)]
    return [(west, 180.0), (-180.0, east - 360.0)]


def longitude_intersects(
    first_west: float,
    first_east: float,
    second_west: float,
    second_east: float,
) -> bool:
    first_west, first_east = normalize_longitude_interval(first_west, first_east)
    second_west, second_east = normalize_longitude_interval(
        second_west,
        second_east,
    )
    return any(
        max(first_west, second_west + shift)
        <= min(first_east, second_east + shift)
        for shift in (-360.0, 0.0, 360.0)
    )


def find_default_index(data_dir: Path) -> Path | None:
    return next(
        (
            candidate
            for name in DEFAULT_INDEX_NAMES
            if (candidate := data_dir / name).is_file()
        ),
        None,
    )


def indexed_matches(
    *,
    data_dir: Path,
    index_path: Path,
    index_layer: str | None,
    location_field: str,
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    ul_lat, ul_lon, lr_lat, lr_lon = bounds
    source = TileSourceConfig(
        name="search",
        data_dir=data_dir,
        index_path=index_path,
        index_layer=index_layer,
        location_field=location_field,
        selection_mode="all_intersecting",
        required=False,
    )
    by_path: dict[Path, dict[str, Any]] = {}
    for interval_west, interval_east in conventional_longitude_intervals(
        ul_lon,
        lr_lon,
    ):
        records = query_source_index(
            source,
            ul_lat=ul_lat,
            ul_lon=interval_west,
            lr_lat=lr_lat,
            lr_lon=interval_east,
        )
        for record in records:
            path = explore_path(record.path)
            if path.suffix.lower() not in GEOTIFF_SUFFIXES:
                continue
            if not path.is_file():
                raise FileNotFoundError(
                    f"Raster index references a missing GeoTIFF: {path}"
                )
            by_path[path] = {
                "filename": path.name,
                "path": str(path),
                "feature_id": record.feature_id,
            }
    return [by_path[path] for path in sorted(by_path, key=str)]


def geotiff_paths(data_dir: Path, *, recursive: bool) -> list[Path]:
    candidates = data_dir.rglob("*") if recursive else data_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in GEOTIFF_SUFFIXES
        ),
        key=str,
    )


def scanned_matches(
    *,
    data_dir: Path,
    bounds: tuple[float, float, float, float],
    recursive: bool,
    edge_samples: int,
) -> tuple[list[dict[str, Any]], int]:
    from extract_datacube_aoi import extract_aoi

    ul_lat, ul_lon, lr_lat, lr_lon = bounds
    query_west, query_east = normalize_longitude_interval(ul_lon, lr_lon)
    paths = geotiff_paths(data_dir, recursive=recursive)
    matches: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        print(f"Scanning raster {index}/{len(paths)}: {path.name}", flush=True)
        footprint = extract_aoi(path, edge_samples=edge_samples)["aoi"]
        latitude_overlap = (
            footprint["ul_lat"] >= lr_lat and footprint["lr_lat"] <= ul_lat
        )
        longitude_overlap = longitude_intersects(
            query_west,
            query_east,
            footprint["ul_lon"],
            footprint["lr_lon"],
        )
        if latitude_overlap and longitude_overlap:
            output_path = explore_path(path)
            matches.append(
                {
                    "filename": output_path.name,
                    "path": str(output_path),
                    "feature_id": None,
                    "footprint_aoi": footprint,
                }
            )
    return matches, len(paths)


def main() -> int:
    args = parse_args()
    data_dir = explore_path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"GeoTIFF directory does not exist: {data_dir}")
    if args.edge_samples < 2:
        raise ValueError("--edge-samples must be at least 2.")

    bounds = tuple(float(value) for value in args.bounds)
    ul_lat, ul_lon, lr_lat, lr_lon = bounds
    if ul_lat < lr_lat:
        raise ValueError(
            f"UL_LAT must be north of LR_LAT, got {ul_lat} < {lr_lat}."
        )
    normalize_longitude_interval(ul_lon, lr_lon)

    explicit_index = explore_path(args.index_path) if args.index_path else None
    discovered_index = explicit_index or find_default_index(data_dir)
    if explicit_index is not None and not explicit_index.is_file():
        raise FileNotFoundError(f"Vector index does not exist: {explicit_index}")
    if args.mode == "index" and discovered_index is None:
        candidates = ", ".join(DEFAULT_INDEX_NAMES)
        raise FileNotFoundError(
            f"Index mode requires --index-path or one of {candidates} in {data_dir}."
        )

    use_index = args.mode == "index" or (
        args.mode == "auto" and discovered_index is not None
    )
    if use_index:
        mode = "index"
        index_path = discovered_index
        print(f"Querying existing raster index: {index_path}", flush=True)
        matches = indexed_matches(
            data_dir=data_dir,
            index_path=index_path,
            index_layer=args.index_layer,
            location_field=args.location_field,
            bounds=bounds,
        )
        candidate_count = None
    else:
        mode = "scan"
        index_path = None
        print(
            "No existing default index selected; scanning GeoTIFF metadata "
            "directly.",
            flush=True,
        )
        matches, candidate_count = scanned_matches(
            data_dir=data_dir,
            bounds=bounds,
            recursive=args.recursive,
            edge_samples=args.edge_samples,
        )

    report = {
        "status": "passed",
        "mode": mode,
        "data_dir": str(data_dir),
        "index_path": str(index_path) if index_path is not None else None,
        "index_layer": args.index_layer,
        "location_field": args.location_field if mode == "index" else None,
        "aoi": {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
        },
        "candidate_count": candidate_count,
        "match_count": len(matches),
        "matches": matches,
    }

    print(f"\nIntersecting GeoTIFFs: {len(matches)}")
    for match in matches:
        print(f"  {match['filename']}")
    print("\nFull paths:")
    for match in matches:
        print(f"  {match['path']}")

    if args.report_path is not None:
        report_path = explore_path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
