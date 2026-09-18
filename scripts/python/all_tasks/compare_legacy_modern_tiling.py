#!/usr/bin/env python3
"""Compare representative legacy and modern WAC/static LTM cubes.

The same geographic AOI is run through the legacy ``Pipeline`` compatibility
path and the configuration-driven tiler. Outputs are paired by modality, LTM
zone, zoom, and tile coordinates. Bands are compared by their ``Name``
metadata so an intentional ordering change does not hide pixel differences.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, gdalconst


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from lfm.model import (  # noqa: E402
    BandNoDataOverride,
    MINIRF_SOURCE_NODATA,
    MINIRF_SOURCE_NODATA_BANDS,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
    TileConfig,
    TileSourceConfig,
    create_tiles_for_aoi,
)
from lfm.model.Pipeline import Pipeline  # noqa: E402


gdal.UseExceptions()

DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)
DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")
DEFAULT_PRODUCT_ID = "M1187363083CE"
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)

LEGACY_FILENAME_RE = re.compile(
    r"^(?P<prefix>StaticCube|Cube)-LTM(?P<zone>\d+[NS])"
    r"_Zoom-(?P<zoom>\d+)_Tile-(?P<tile_x>\d+)-(?P<tile_y>\d+)"
    r"(?:_ProdId-(?P<product_id>.+))?\.tif$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and modern WAC/static LTM cube outputs."
    )
    parser.add_argument("--wac-data-dir", type=Path, default=DEFAULT_WAC_DATA_DIR)
    parser.add_argument("--wac-index", type=Path, default=None)
    parser.add_argument("--wac-index-layer", default=None)
    parser.add_argument("--static-data-dir", type=Path, default=DEFAULT_STATIC_DATA_DIR)
    parser.add_argument("--static-index", type=Path, default=None)
    parser.add_argument("--static-index-layer", default=None)
    parser.add_argument("--location-field", default="location")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-id", default=DEFAULT_PRODUCT_ID)
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("UL_LAT", "UL_LON", "LR_LAT", "LR_LON"),
        default=DEFAULT_BOUNDS,
    )
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument("--expected-tile-count", type=int, default=2)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of jointly valid pixels outside numeric tolerance.",
    )
    parser.add_argument(
        "--max-mask-mismatch-fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of pixels with differing valid/NoData masks.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def cube_key(
    source_name: str,
    zone: str,
    zoom_level: int,
    tile_x: int,
    tile_y: int,
) -> tuple[str, str, int, int, int]:
    return source_name, zone, int(zoom_level), int(tile_x), int(tile_y)


def key_text(key: tuple[str, str, int, int, int]) -> str:
    source, zone, zoom, tile_x, tile_y = key
    return f"{source}:LTM{zone}:z{zoom}:x{tile_x}:y{tile_y}"


def parse_legacy_key(path: Path) -> tuple[str, str, int, int, int]:
    match = LEGACY_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized legacy cube filename: {path.name}")
    source_name = "static" if match.group("prefix") == "StaticCube" else "wac"
    return cube_key(
        source_name,
        match.group("zone"),
        int(match.group("zoom")),
        int(match.group("tile_x")),
        int(match.group("tile_y")),
    )


def same_optional_number(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return math.isclose(first, second, rel_tol=1.0e-7, abs_tol=0.0)


def invalid_mask(array: np.ndarray, nodata: float | None) -> np.ndarray:
    invalid = ~np.isfinite(array)
    if nodata is None:
        return invalid
    if math.isnan(nodata):
        return invalid | np.isnan(array)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(nodata, dtype=array.dtype).item()
    except (OverflowError, TypeError, ValueError):
        return invalid
    if np.isfinite(converted):
        invalid |= array == converted
    return invalid


def band_inventory(dataset: Any, path: Path) -> tuple[list[str], dict[str, int]]:
    names: list[str] = []
    indexes: dict[str, int] = {}
    duplicates: list[str] = []
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        name = band.GetMetadataItem("Name") or band.GetDescription()
        if not name:
            name = f"band_{index}"
        names.append(name)
        if name in indexes:
            duplicates.append(name)
        else:
            indexes[name] = index
    if duplicates:
        raise ValueError(f"Duplicate band names in {path}: {sorted(set(duplicates))}")
    return names, indexes


def compare_band(
    *,
    name: str,
    legacy_dataset: Any,
    legacy_index: int,
    modern_dataset: Any,
    modern_index: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_outlier_fraction: float,
    max_mask_mismatch_fraction: float,
) -> dict[str, Any]:
    legacy_band = legacy_dataset.GetRasterBand(legacy_index)
    modern_band = modern_dataset.GetRasterBand(modern_index)
    legacy_nodata = legacy_band.GetNoDataValue()
    modern_nodata = modern_band.GetNoDataValue()
    legacy_nodata = float(legacy_nodata) if legacy_nodata is not None else None
    modern_nodata = float(modern_nodata) if modern_nodata is not None else None

    legacy_array = legacy_band.ReadAsArray()
    modern_array = modern_band.ReadAsArray()
    if legacy_array is None or modern_array is None:
        return {
            "band_name": name,
            "passed": False,
            "failure": "GDAL could not read one or both bands.",
        }
    legacy = np.asarray(legacy_array)
    modern = np.asarray(modern_array)
    if legacy.shape != modern.shape:
        return {
            "band_name": name,
            "passed": False,
            "failure": f"shape mismatch: {legacy.shape} != {modern.shape}",
        }

    legacy_invalid = invalid_mask(legacy, legacy_nodata)
    modern_invalid = invalid_mask(modern, modern_nodata)
    mask_mismatch_count = int(np.count_nonzero(legacy_invalid ^ modern_invalid))
    mask_mismatch_fraction = mask_mismatch_count / legacy.size
    jointly_valid = ~legacy_invalid & ~modern_invalid
    jointly_valid_count = int(np.count_nonzero(jointly_valid))

    if jointly_valid_count:
        legacy_valid = legacy[jointly_valid].astype(np.float64, copy=False)
        modern_valid = modern[jointly_valid].astype(np.float64, copy=False)
        difference = np.abs(legacy_valid - modern_valid)
        close = np.isclose(
            legacy_valid,
            modern_valid,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        outlier_count = int(np.count_nonzero(~close))
        outlier_fraction = outlier_count / jointly_valid_count
        maximum_absolute_difference = float(difference.max())
        mean_absolute_difference = float(difference.mean())
        root_mean_square_difference = float(
            np.sqrt(np.mean(np.square(difference)))
        )
    else:
        outlier_count = 0
        outlier_fraction = 0.0
        maximum_absolute_difference = None
        mean_absolute_difference = None
        root_mean_square_difference = None

    nodata_matches = same_optional_number(legacy_nodata, modern_nodata)
    passed = (
        nodata_matches
        and mask_mismatch_fraction <= max_mask_mismatch_fraction
        and outlier_fraction <= max_outlier_fraction
    )
    return {
        "band_name": name,
        "passed": passed,
        "legacy_index": legacy_index,
        "modern_index": modern_index,
        "legacy_dtype": str(legacy.dtype),
        "modern_dtype": str(modern.dtype),
        "dtype_matches": legacy.dtype == modern.dtype,
        "legacy_nodata": legacy_nodata,
        "modern_nodata": modern_nodata,
        "nodata_matches": nodata_matches,
        "pixel_count": int(legacy.size),
        "legacy_invalid_count": int(np.count_nonzero(legacy_invalid)),
        "modern_invalid_count": int(np.count_nonzero(modern_invalid)),
        "mask_mismatch_count": mask_mismatch_count,
        "mask_mismatch_fraction": mask_mismatch_fraction,
        "jointly_valid_count": jointly_valid_count,
        "outlier_count": outlier_count,
        "outlier_fraction": outlier_fraction,
        "maximum_absolute_difference": maximum_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "root_mean_square_difference": root_mean_square_difference,
    }


def compare_cube_pair(
    *,
    key: tuple[str, str, int, int, int],
    legacy_path: Path,
    modern_path: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_outlier_fraction: float,
    max_mask_mismatch_fraction: float,
) -> dict[str, Any]:
    legacy = gdal.Open(str(legacy_path), gdalconst.GA_ReadOnly)
    modern = gdal.Open(str(modern_path), gdalconst.GA_ReadOnly)
    if legacy is None or modern is None:
        raise RuntimeError(
            f"Could not open comparison pair: {legacy_path}, {modern_path}"
        )
    try:
        legacy_names, legacy_indexes = band_inventory(legacy, legacy_path)
        modern_names, modern_indexes = band_inventory(modern, modern_path)
        missing_in_legacy = [name for name in modern_names if name not in legacy_indexes]
        legacy_only = [name for name in legacy_names if name not in modern_indexes]
        common_names = [name for name in modern_names if name in legacy_indexes]

        source_name = key[0]
        band_set_acceptable = not missing_in_legacy and (
            source_name == "static" or not legacy_only
        )
        band_order_matches = legacy_names == modern_names

        shape_matches = (
            legacy.RasterXSize == modern.RasterXSize
            and legacy.RasterYSize == modern.RasterYSize
        )
        geotransform_matches = all(
            math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-8)
            for first, second in zip(
                legacy.GetGeoTransform(),
                modern.GetGeoTransform(),
                strict=True,
            )
        )
        legacy_srs = legacy.GetSpatialRef()
        modern_srs = modern.GetSpatialRef()
        crs_matches = bool(
            legacy_srs is not None
            and modern_srs is not None
            and legacy_srs.IsSame(modern_srs)
        )

        band_results = [
            compare_band(
                name=name,
                legacy_dataset=legacy,
                legacy_index=legacy_indexes[name],
                modern_dataset=modern,
                modern_index=modern_indexes[name],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                max_outlier_fraction=max_outlier_fraction,
                max_mask_mismatch_fraction=max_mask_mismatch_fraction,
            )
            for name in common_names
        ]
        failed_bands = [row["band_name"] for row in band_results if not row["passed"]]
        passed = (
            shape_matches
            and geotransform_matches
            and crs_matches
            and band_set_acceptable
            and not failed_bands
        )
        intentional_differences: list[str] = []
        if not band_order_matches and not missing_in_legacy:
            intentional_differences.append(
                "Modern bands use declared order; comparison aligned bands by name."
            )
        if source_name == "static" and legacy_only:
            intentional_differences.append(
                "Modern static output selects the canonical 63-band contract; "
                "legacy-only indexed bands were not compared."
            )
        if any(not row.get("dtype_matches", True) for row in band_results):
            intentional_differences.append(
                "Output dtypes differ for at least one band; numeric tolerance "
                "determines equivalence."
            )

        return {
            "key": key_text(key),
            "source_name": source_name,
            "zone": key[1],
            "zoom_level": key[2],
            "tile_x": key[3],
            "tile_y": key[4],
            "passed": passed,
            "legacy_path": str(legacy_path),
            "modern_path": str(modern_path),
            "legacy_file_size_bytes": legacy_path.stat().st_size,
            "modern_file_size_bytes": modern_path.stat().st_size,
            "legacy_shape": [legacy.RasterYSize, legacy.RasterXSize],
            "modern_shape": [modern.RasterYSize, modern.RasterXSize],
            "shape_matches": shape_matches,
            "geotransform_matches": geotransform_matches,
            "crs_matches": crs_matches,
            "legacy_band_count": len(legacy_names),
            "modern_band_count": len(modern_names),
            "band_order_matches": band_order_matches,
            "missing_in_legacy": missing_in_legacy,
            "legacy_only_bands": legacy_only,
            "failed_bands": failed_bands,
            "intentional_differences": intentional_differences,
            "legacy_band_names": legacy_names,
            "modern_band_names": modern_names,
            "bands": band_results,
        }
    finally:
        legacy = None
        modern = None


def require_empty_cube_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("*.tif"))
    if existing:
        raise FileExistsError(
            f"Comparison directory already contains GeoTIFFs: {path}. "
            "Use a fresh --output-dir."
        )


def run_legacy(
    *,
    wac_index: Path,
    static_index: Path,
    output_dir: Path,
    product_id: str,
    bounds: tuple[float, float, float, float],
    zoom_level: int,
) -> list[Path]:
    original_static_db = Pipeline.STATIC_FILE_DB
    try:
        Pipeline.STATIC_FILE_DB = static_index
        pipeline = Pipeline(
            wac_index,
            output_dir,
            targetProductID=product_id,
        )
        return [Path(path) for path in pipeline.run(*bounds, zoom_level)]
    finally:
        Pipeline.STATIC_FILE_DB = original_static_db


def run_modern(
    *,
    wac_data_dir: Path,
    wac_index: Path,
    wac_index_layer: str | None,
    static_data_dir: Path,
    static_index: Path,
    static_index_layer: str | None,
    location_field: str,
    output_dir: Path,
    product_id: str,
    bounds: tuple[float, float, float, float],
    zoom_level: int,
) -> list[Any]:
    wac_source = TileSourceConfig(
        name="wac",
        data_dir=wac_data_dir,
        index_path=wac_index,
        index_layer=wac_index_layer,
        location_field=location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    static_source = TileSourceConfig(
        name="static",
        data_dir=static_data_dir,
        index_path=static_index,
        index_layer=static_index_layer,
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
    )
    config = TileConfig(
        output_dir=output_dir,
        zoom_level=zoom_level,
        sources=(wac_source, static_source),
    )
    return create_tiles_for_aoi(
        config,
        ul_lat=bounds[0],
        ul_lon=bounds[1],
        lr_lat=bounds[2],
        lr_lon=bounds[3],
        selectors={"wac": product_id},
    )


def main() -> int:
    args = parse_args()
    if args.expected_tile_count < 1:
        raise ValueError("--expected-tile-count must be positive.")
    for name in (
        "absolute_tolerance",
        "relative_tolerance",
        "max_outlier_fraction",
        "max_mask_mismatch_fraction",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if args.max_outlier_fraction > 1 or args.max_mask_mismatch_fraction > 1:
        raise ValueError("Mismatch fractions must not exceed 1.")

    wac_index = args.wac_index or args.wac_data_dir / "output_index.shp"
    static_index = args.static_index or args.static_data_dir / "db2.shp"
    report_path = args.report_path or args.output_dir / "comparison_report.json"
    for name, path in (
        ("WAC data directory", args.wac_data_dir),
        ("static data directory", args.static_data_dir),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for name, path in (("WAC index", wac_index), ("static index", static_index)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    legacy_dir = args.output_dir / "legacy"
    modern_dir = args.output_dir / "modern"
    require_empty_cube_dir(legacy_dir)
    require_empty_cube_dir(modern_dir)
    bounds = tuple(float(value) for value in args.bounds)

    print("Running legacy Pipeline comparison baseline...", flush=True)
    legacy_paths = run_legacy(
        wac_index=wac_index,
        static_index=static_index,
        output_dir=legacy_dir,
        product_id=args.product_id,
        bounds=bounds,
        zoom_level=args.zoom_level,
    )
    print("Running modern TileConfig implementation...", flush=True)
    modern_records = run_modern(
        wac_data_dir=args.wac_data_dir,
        wac_index=wac_index,
        wac_index_layer=args.wac_index_layer,
        static_data_dir=args.static_data_dir,
        static_index=static_index,
        static_index_layer=args.static_index_layer,
        location_field=args.location_field,
        output_dir=modern_dir,
        product_id=args.product_id,
        bounds=bounds,
        zoom_level=args.zoom_level,
    )

    legacy_by_key: dict[tuple[str, str, int, int, int], Path] = {}
    for path in legacy_paths:
        key = parse_legacy_key(path)
        if key in legacy_by_key:
            raise ValueError(f"Duplicate legacy cube key: {key_text(key)}")
        legacy_by_key[key] = path
    modern_by_key: dict[tuple[str, str, int, int, int], Path] = {}
    for record in modern_records:
        key = cube_key(
            record.source_name,
            record.zone,
            record.zoom_level,
            record.tile_x,
            record.tile_y,
        )
        if key in modern_by_key:
            raise ValueError(f"Duplicate modern cube key: {key_text(key)}")
        modern_by_key[key] = record.path

    expected_cube_count = args.expected_tile_count * 2
    missing_legacy_keys = sorted(set(modern_by_key) - set(legacy_by_key))
    missing_modern_keys = sorted(set(legacy_by_key) - set(modern_by_key))
    common_keys = sorted(set(legacy_by_key).intersection(modern_by_key))
    comparisons = [
        compare_cube_pair(
            key=key,
            legacy_path=legacy_by_key[key],
            modern_path=modern_by_key[key],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            max_outlier_fraction=args.max_outlier_fraction,
            max_mask_mismatch_fraction=args.max_mask_mismatch_fraction,
        )
        for key in common_keys
    ]

    overall_passed = (
        len(legacy_by_key) == expected_cube_count
        and len(modern_by_key) == expected_cube_count
        and not missing_legacy_keys
        and not missing_modern_keys
        and all(comparison["passed"] for comparison in comparisons)
    )
    payload = {
        "status": "passed" if overall_passed else "differences_detected",
        "query": {
            "ul_lat": bounds[0],
            "ul_lon": bounds[1],
            "lr_lat": bounds[2],
            "lr_lon": bounds[3],
            "zoom_level": args.zoom_level,
            "product_id": args.product_id,
        },
        "tolerances": {
            "absolute": args.absolute_tolerance,
            "relative": args.relative_tolerance,
            "max_outlier_fraction": args.max_outlier_fraction,
            "max_mask_mismatch_fraction": args.max_mask_mismatch_fraction,
        },
        "expected_cube_count_per_implementation": expected_cube_count,
        "legacy_cube_count": len(legacy_by_key),
        "modern_cube_count": len(modern_by_key),
        "missing_legacy_keys": [key_text(key) for key in missing_legacy_keys],
        "missing_modern_keys": [key_text(key) for key in missing_modern_keys],
        "intentional_contract_differences": [
            "Modern output filenames identify the configured source and return "
            "structured TileCubeRecord results.",
            "Modern static output uses the explicitly declared canonical 63-band "
            "order; comparisons align legacy bands by Name metadata.",
            "Every static output band uses -32768 NoData while source-specific "
            "sentinels are applied only during bilinear warping.",
        ],
        "comparisons": comparisons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nLegacy/modern comparison summary:")
    for comparison in comparisons:
        print(
            f"  {comparison['key']}: "
            f"{'PASS' if comparison['passed'] else 'DIFFERENCES'}; "
            f"bands={comparison['modern_band_count']}, "
            f"failed_bands={len(comparison['failed_bands'])}, "
            f"order_matches={comparison['band_order_matches']}"
        )
        for difference in comparison["intentional_differences"]:
            print(f"    intentional: {difference}")
    print(f"Report: {report_path}")
    if not overall_passed:
        raise AssertionError(
            "Legacy/modern tiling differences exceeded the configured acceptance "
            f"criteria. Inspect {report_path}."
        )
    print("Legacy/modern tiling comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
