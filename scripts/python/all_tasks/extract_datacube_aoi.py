#!/usr/bin/env python3
"""Extract a lunar latitude/longitude AOI from a georeferenced datacube."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import LUNAR_GEOGRAPHIC_WKT_PATH, load_lunar_geographic_wkt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform a datacube footprint from its embedded raster CRS to "
            "the repository IAU:30100 lunar geographic CRS."
        )
    )
    parser.add_argument(
        "datacube",
        type=Path,
        help="Path to a georeferenced datacube readable by GDAL.",
    )
    parser.add_argument(
        "--edge-samples",
        type=int,
        default=21,
        help=(
            "Points sampled along each footprint edge when calculating the "
            "geographic envelope (default: 21)."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="Decimal places printed for coordinates (default: 10).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path for a detailed JSON report.",
    )
    return parser.parse_args()


def pixel_to_projected(
    geotransform: tuple[float, ...],
    pixel: float,
    line: float,
) -> tuple[float, float]:
    x = geotransform[0] + pixel * geotransform[1] + line * geotransform[2]
    y = geotransform[3] + pixel * geotransform[4] + line * geotransform[5]
    return float(x), float(y)


def perimeter_pixels(
    width: int,
    height: int,
    samples_per_edge: int,
) -> list[tuple[float, float]]:
    fractions = [
        index / (samples_per_edge - 1) for index in range(samples_per_edge)
    ]
    points = [
        *((fraction * width, 0.0) for fraction in fractions),
        *((float(width), fraction * height) for fraction in fractions[1:]),
        *(
            ((1.0 - fraction) * width, float(height))
            for fraction in fractions[1:]
        ),
        *((0.0, (1.0 - fraction) * height) for fraction in fractions[1:-1]),
    ]
    return points


def longitude_envelope(longitudes: list[float]) -> tuple[float, float, float, bool]:
    """Return the smallest continuous west/east interval containing longitudes."""
    if not longitudes:
        raise ValueError("At least one longitude is required.")
    normalized = sorted(float(value) % 360.0 for value in longitudes)
    if len(normalized) == 1:
        west = normalized[0] if normalized[0] <= 180.0 else normalized[0] - 360.0
        return west, west, 0.0, False
    gaps = [
        normalized[index + 1] - normalized[index]
        for index in range(len(normalized) - 1)
    ]
    gaps.append(normalized[0] + 360.0 - normalized[-1])
    largest_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    start = normalized[(largest_gap_index + 1) % len(normalized)]
    span = 360.0 - gaps[largest_gap_index]
    west = start if start <= 180.0 else start - 360.0
    east = west + span
    crosses_antimeridian = east > 180.0
    return west, east, span, crosses_antimeridian


def longitude_in_interval(longitude: float, west: float) -> float:
    value = float(longitude)
    while value < west:
        value += 360.0
    while value >= west + 360.0:
        value -= 360.0
    return value


def create_transformation(source_srs, target_srs):
    from osgeo import osr

    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # Explore's custom-authority LTM definitions require legacy return-code
    # behavior while the transformation object is constructed.
    with osr.ExceptionMgr(useExceptions=False):
        transformation = osr.CoordinateTransformation(source_srs, target_srs)
    if transformation is None:
        raise RuntimeError(
            "Could not construct a transformation from the datacube CRS to "
            "the repository IAU:30100 CRS."
        )
    return transformation


def transform_point(transformation, x: float, y: float) -> tuple[float, float]:
    transformed = transformation.TransformPoint(x, y)
    if transformed is None or len(transformed) < 2:
        raise RuntimeError(f"Coordinate transformation failed for ({x}, {y}).")
    longitude, latitude = float(transformed[0]), float(transformed[1])
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise RuntimeError(
            "Coordinate transformation returned non-finite longitude/latitude "
            f"for ({x}, {y}): ({longitude}, {latitude})."
        )
    return latitude, longitude


def extract_aoi(datacube: Path, *, edge_samples: int) -> dict[str, Any]:
    from osgeo import gdal, gdalconst, osr

    gdal.UseExceptions()
    dataset = gdal.Open(str(datacube), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"GDAL could not open datacube: {datacube}")
    try:
        source_srs = dataset.GetSpatialRef()
        if source_srs is None:
            raise ValueError(f"Datacube has no embedded CRS: {datacube}")
        source_srs = source_srs.Clone()
        geotransform = dataset.GetGeoTransform(can_return_null=True)
        if geotransform is None:
            raise ValueError(f"Datacube has no geotransform: {datacube}")
        geotransform = tuple(float(value) for value in geotransform)
        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)
        band_count = int(dataset.RasterCount)

        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(load_lunar_geographic_wkt())
        # The datacube source is expected to be a projected LTM zone. Legacy
        # LTM WKT definitions do not always compare equal to the repository's
        # modern IAU:30100 geographic WKT through CloneGeogCS(), even though
        # PROJ can construct the intended LTM-to-IAU transformation. Use the
        # embedded projected CRS directly instead of rejecting it by name or
        # geographic-base metadata.
        transformation = create_transformation(source_srs, target_srs)

        named_corner_pixels = {
            "upper_left": (0.0, 0.0),
            "upper_right": (float(width), 0.0),
            "lower_right": (float(width), float(height)),
            "lower_left": (0.0, float(height)),
        }
        corners: list[dict[str, Any]] = []
        for name, (pixel, line) in named_corner_pixels.items():
            x, y = pixel_to_projected(geotransform, pixel, line)
            latitude, longitude = transform_point(transformation, x, y)
            corners.append(
                {
                    "name": name,
                    "pixel": pixel,
                    "line": line,
                    "projected_x": x,
                    "projected_y": y,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

        perimeter_geographic: list[tuple[float, float]] = []
        for pixel, line in perimeter_pixels(width, height, edge_samples):
            x, y = pixel_to_projected(geotransform, pixel, line)
            perimeter_geographic.append(transform_point(transformation, x, y))

        latitudes = [latitude for latitude, _ in perimeter_geographic]
        longitudes = [longitude for _, longitude in perimeter_geographic]
        west, east, longitude_span, crosses_antimeridian = longitude_envelope(
            longitudes
        )
        for corner in corners:
            corner["longitude"] = longitude_in_interval(corner["longitude"], west)

        return {
            "datacube": str(datacube),
            "raster": {
                "width": width,
                "height": height,
                "band_count": band_count,
                "geotransform": list(geotransform),
                "source_crs_name": source_srs.GetName(),
                "source_crs_wkt": source_srs.ExportToWkt(),
            },
            "target_crs": {
                "name": target_srs.GetName(),
                "wkt_path": str(LUNAR_GEOGRAPHIC_WKT_PATH),
                "wkt": target_srs.ExportToWkt(),
            },
            "edge_samples_per_edge": edge_samples,
            "corners": corners,
            "aoi": {
                "ul_lat": max(latitudes),
                "ul_lon": west,
                "lr_lat": min(latitudes),
                "lr_lon": east,
                "longitude_span_degrees": longitude_span,
                "crosses_antimeridian": crosses_antimeridian,
            },
        }
    finally:
        dataset = None


def print_report(report: dict[str, Any], *, precision: int) -> None:
    raster = report["raster"]
    aoi = report["aoi"]
    coordinate_format = f".{{precision}}f"

    def formatted(value: float) -> str:
        return format(value, coordinate_format.format(precision=precision))

    print(f"Datacube: {report['datacube']}")
    print(
        f"Raster: {raster['width']} x {raster['height']} pixels, "
        f"{raster['band_count']} band(s)"
    )
    print(f"Source CRS: {raster['source_crs_name']}")
    print(f"Target CRS: {report['target_crs']['name']} (IAU:30100, 2015)")
    print(f"Edge samples per edge: {report['edge_samples_per_edge']}")
    print("\nCorner coordinates (latitude, longitude):")
    for corner in report["corners"]:
        print(
            f"  {corner['name']:>11}: "
            f"({formatted(corner['latitude'])}, "
            f"{formatted(corner['longitude'])})"
        )

    print("\nAOI_BOUNDS = {")
    print(f"    \"ul_lat\": {formatted(aoi['ul_lat'])},")
    print(f"    \"ul_lon\": {formatted(aoi['ul_lon'])},")
    print(f"    \"lr_lat\": {formatted(aoi['lr_lat'])},")
    print(f"    \"lr_lon\": {formatted(aoi['lr_lon'])},")
    print("}")
    print(f"Longitude span: {formatted(aoi['longitude_span_degrees'])} degrees")
    if aoi["crosses_antimeridian"]:
        print(
            "Note: this footprint crosses the antimeridian; lr_lon is printed "
            "above 180 degrees to preserve a small continuous AOI interval."
        )


def main() -> int:
    args = parse_args()
    if not args.datacube.is_file():
        raise FileNotFoundError(f"Datacube does not exist: {args.datacube}")
    if args.edge_samples < 2:
        raise ValueError("--edge-samples must be at least 2.")
    if args.precision < 0 or args.precision > 15:
        raise ValueError("--precision must be between 0 and 15.")

    report = extract_aoi(args.datacube, edge_samples=args.edge_samples)
    print_report(report, precision=args.precision)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
