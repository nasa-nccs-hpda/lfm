#!/usr/bin/env python3
"""Run modern and legacy WAC/static tiling for AOIs declared in JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(
    str(Path(__file__).resolve().parents[1]).replace(
        "/panfs/ccds02/nobackup",
        "/explore/nobackup",
    )
)
if not (REPO_ROOT / "lfm").is_dir() or not (REPO_ROOT / "model").is_dir():
    raise FileNotFoundError(
        f"Could not find the lfm/ and model/ directories beneath {REPO_ROOT}."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import TileConfig, TileSourceConfig, create_tiles_for_aoi  # noqa: E402
from model.Pipeline import Pipeline  # noqa: E402
from lfm.all_models.all_tasks.tiling_utils import (  # noqa: E402
    DEFAULT_WAC_BAND_NUMBER,
    DEFAULT_ZOOM_LEVEL,
    RUN_ID,
    make_static_source,
    validate_path_pairs,
)
from lfm.all_models.all_tasks.viz import (  # noqa: E402
    pair_dynamic_and_static,
    plot_modern_legacy_cube_comparison,
    print_record_summary,
)


PROJECT_DATA_DIR = Path("/explore/nobackup/projects/lfm")
WAC_DATA_DIR = PROJECT_DATA_DIR / "processed_data/Lunar/LRO_WAC_Pho_Sites"
STATIC_DATA_DIR = PROJECT_DATA_DIR / "staticLinks"
WAC_INDEX = WAC_DATA_DIR / "output_index.shp"
STATIC_INDEX = STATIC_DATA_DIR / "db2.shp"
LOCATION_FIELD = "location"

ZOOM_LEVEL = DEFAULT_ZOOM_LEVEL
WAC_BAND_NUMBER = DEFAULT_WAC_BAND_NUMBER
STATIC_BAND_TO_PLOT = "lola_kaguya_60mpp_elv"
MAX_PLOT_TILES = 2

AOI_KEYS = ("ul_lat", "ul_lon", "lr_lat", "lr_lon")
LEGACY_CUBE_NAME = re.compile(
    r"^(?P<prefix>StaticCube|Cube)-LTM(?P<zone>\d+[NS])"
    r"_Zoom-(?P<zoom>\d+)_Tile-(?P<tile_x>\d+)-(?P<tile_y>\d+)"
    r"(?:_ProdId-.+)?\.tif$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create modern WAC/static LTM cubes and plots plus legacy Pipeline "
            "cubes for every product/AOI entry in a JSON configuration file."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="JSON file containing an 'entries' list of product IDs and AOIs.",
    )
    return parser.parse_args()


def load_entries(config_path: Path) -> list[dict[str, Any]]:
    """Load and validate product-scoped geographic AOIs from JSON."""
    if not config_path.is_file():
        raise FileNotFoundError(f"JSON configuration does not exist: {config_path}")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    raw_entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            "The JSON configuration must contain a nonempty 'entries' list."
        )

    entries: list[dict[str, Any]] = []
    seen_product_ids: set[str] = set()
    for number, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Entry {number} must be a JSON object.")

        product_id = raw_entry.get("product_id")
        if (
            not isinstance(product_id, str)
            or not product_id.startswith("M")
            or not product_id.isalnum()
        ):
            raise ValueError(
                f"Entry {number} product_id must be a pure alphanumeric M... ID."
            )
        if product_id in seen_product_ids:
            raise ValueError(f"Duplicate product_id in entry {number}: {product_id}")
        seen_product_ids.add(product_id)

        raw_aoi = raw_entry.get("aoi")
        if not isinstance(raw_aoi, dict):
            raise TypeError(f"Entry {number} must contain an 'aoi' object.")
        missing = [key for key in AOI_KEYS if key not in raw_aoi]
        extra = sorted(set(raw_aoi) - set(AOI_KEYS))
        if missing or extra:
            raise ValueError(
                f"Entry {number} AOI keys are invalid; "
                f"missing={missing}, extra={extra}."
            )

        try:
            aoi = {key: float(raw_aoi[key]) for key in AOI_KEYS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Entry {number} AOI values must be numeric.") from exc
        if not all(math.isfinite(value) for value in aoi.values()):
            raise ValueError(f"Entry {number} AOI values must be finite.")
        if not (-90.0 <= aoi["lr_lat"] < aoi["ul_lat"] <= 90.0):
            raise ValueError(
                f"Entry {number} must have valid latitudes with ul_lat > lr_lat."
            )
        if aoi["ul_lon"] >= aoi["lr_lon"]:
            raise ValueError(
                f"Entry {number} must have ul_lon < lr_lon; "
                "antimeridian-crossing AOIs are not supported by this script."
            )

        entries.append({"product_id": product_id, "aoi": aoi})
    return entries


def run_legacy_tiling(
    *,
    product_id: str,
    aoi: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    """Run the legacy WAC/static Pipeline for one product-scoped AOI."""
    output_dir.mkdir(parents=True, exist_ok=False)
    original_static_index = Pipeline.STATIC_FILE_DB
    try:
        Pipeline.STATIC_FILE_DB = STATIC_INDEX
        pipeline = Pipeline(
            WAC_INDEX,
            output_dir,
            targetProductID=product_id,
        )
        return [
            Path(path)
            for path in pipeline.run(
                aoi["ul_lat"],
                aoi["ul_lon"],
                aoi["lr_lat"],
                aoi["lr_lon"],
                ZOOM_LEVEL,
            )
        ]
    finally:
        Pipeline.STATIC_FILE_DB = original_static_index


def match_modern_and_legacy_cubes(
    modern_pairs: list[tuple[Any, Any]],
    legacy_paths: list[Path],
) -> list[tuple[Any, Any, Path, Path]]:
    """Match current and legacy WAC/static cubes by structured tile identity."""
    legacy_by_tile: dict[tuple[str, int, int, int], dict[str, Path]] = {}
    for path in legacy_paths:
        match = LEGACY_CUBE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized legacy cube filename: {path.name}")
        tile = (
            match.group("zone"),
            int(match.group("zoom")),
            int(match.group("tile_x")),
            int(match.group("tile_y")),
        )
        source_name = "static" if match.group("prefix") == "StaticCube" else "wac"
        sources = legacy_by_tile.setdefault(tile, {})
        if source_name in sources:
            raise ValueError(f"Duplicate legacy {source_name} cube for tile {tile}.")
        sources[source_name] = path

    comparisons: list[tuple[Any, Any, Path, Path]] = []
    missing: list[str] = []
    for current_wac, current_static in modern_pairs:
        tile = (
            current_wac.zone,
            current_wac.zoom_level,
            current_wac.tile_x,
            current_wac.tile_y,
        )
        legacy_sources = legacy_by_tile.get(tile, {})
        if "wac" not in legacy_sources or "static" not in legacy_sources:
            missing.append(str(tile))
            continue
        comparisons.append(
            (
                current_wac,
                current_static,
                legacy_sources["wac"],
                legacy_sources["static"],
            )
        )
    if missing:
        raise ValueError(
            "Legacy WAC/static counterparts are missing for current tiles: "
            + ", ".join(missing)
        )
    if not comparisons:
        raise ValueError("No current and legacy WAC/static tile sets matched.")
    return comparisons


def resolve_band_number(
    source: rasterio.io.DatasetReader,
    *,
    band_number: int | None = None,
    band_name: str | None = None,
) -> int:
    """Resolve exactly one 1-based band number or metadata name."""
    if (band_number is None) == (band_name is None):
        raise ValueError("Provide exactly one of band_number or band_name.")
    if band_name is not None:
        matches = [
            number
            for number in range(1, source.count + 1)
            if source.tags(number).get("Name") == band_name
            or source.descriptions[number - 1] == band_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one band named {band_name!r} in {source.name}, "
                f"found {len(matches)}."
            )
        return matches[0]
    if band_number is None or not 1 <= band_number <= source.count:
        raise IndexError(
            f"Band {band_number} is outside the 1..{source.count} range "
            f"for {source.name}."
        )
    return band_number


def read_band_diagnostics(
    path: Path,
    *,
    band_number: int | None = None,
    band_name: str | None = None,
) -> tuple[dict[str, Any], np.ma.MaskedArray]:
    """Inspect raw values, declared NoData, and the effective raster mask."""
    with rasterio.open(path) as source:
        number = resolve_band_number(
            source,
            band_number=band_number,
            band_name=band_name,
        )
        raw = source.read(number, masked=False)
        masked = source.read(number, masked=True)
        name = (
            source.tags(number).get("Name")
            or source.descriptions[number - 1]
            or f"band_{number}"
        )
        nodata = source.nodatavals[number - 1]

    mask = np.ma.getmaskarray(masked)
    if nodata is None:
        nodata_pixels = np.zeros(raw.shape, dtype=bool)
        nodata_value: float | str | None = None
    elif math.isnan(float(nodata)):
        nodata_pixels = np.isnan(raw)
        nodata_value = "NaN"
    else:
        converted_nodata = np.asarray(nodata, dtype=raw.dtype).item()
        nodata_pixels = raw == converted_nodata
        nodata_value = float(nodata)

    raw_finite = np.asarray(raw, dtype=np.float64)
    raw_finite = raw_finite[np.isfinite(raw_finite)]
    valid_values = np.asarray(masked.compressed(), dtype=np.float64)
    valid_values = valid_values[np.isfinite(valid_values)]
    nodata_count = int(np.count_nonzero(nodata_pixels))
    nodata_masked_count = int(np.count_nonzero(nodata_pixels & mask))
    nodata_unmasked_count = int(np.count_nonzero(nodata_pixels & ~mask))
    raw_min = float(raw_finite.min()) if raw_finite.size else None
    raw_max = float(raw_finite.max()) if raw_finite.size else None
    nodata_is_raw_extreme = None
    if nodata_count and nodata is not None and math.isfinite(float(nodata)):
        nodata_is_raw_extreme = bool(
            float(nodata) == raw_min or float(nodata) == raw_max
        )

    result = {
        "path": str(path),
        "band_number": number,
        "band_name": name,
        "dtype": str(raw.dtype),
        "declared_nodata": nodata_value,
        "raw_finite_min": raw_min,
        "raw_finite_max": raw_max,
        "valid_finite_min": (
            float(valid_values.min()) if valid_values.size else None
        ),
        "valid_finite_max": (
            float(valid_values.max()) if valid_values.size else None
        ),
        "masked_pixel_count": int(np.count_nonzero(mask)),
        "nodata_pixel_count": nodata_count,
        "nodata_masked_count": nodata_masked_count,
        "nodata_unmasked_count": nodata_unmasked_count,
        "all_nodata_occurrences_masked": nodata_unmasked_count == 0,
        "nodata_is_raw_extreme": nodata_is_raw_extreme,
    }
    return result, masked


def compare_diagnostic_bands(
    current: np.ma.MaskedArray,
    legacy: np.ma.MaskedArray,
    *,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare current and legacy masks and jointly valid pixel values."""
    if current.shape != legacy.shape:
        return {
            "passed": False,
            "shape_matches": False,
            "current_shape": list(current.shape),
            "legacy_shape": list(legacy.shape),
        }

    current_mask = np.ma.getmaskarray(current)
    legacy_mask = np.ma.getmaskarray(legacy)
    mask_mismatch = current_mask ^ legacy_mask
    jointly_valid = ~current_mask & ~legacy_mask
    current_values = np.asarray(current.data[jointly_valid], dtype=np.float64)
    legacy_values = np.asarray(legacy.data[jointly_valid], dtype=np.float64)
    close = np.isclose(
        current_values,
        legacy_values,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=False,
    )
    differences = np.abs(current_values - legacy_values)
    mask_mismatch_count = int(np.count_nonzero(mask_mismatch))
    value_mismatch_count = int(np.count_nonzero(~close))
    return {
        "passed": mask_mismatch_count == 0 and value_mismatch_count == 0,
        "shape_matches": True,
        "current_shape": list(current.shape),
        "legacy_shape": list(legacy.shape),
        "mask_mismatch_count": mask_mismatch_count,
        "jointly_valid_pixel_count": int(current_values.size),
        "value_mismatch_count": value_mismatch_count,
        "maximum_absolute_difference": (
            float(differences.max()) if differences.size else None
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def diagnose_modern_legacy_cubes(
    *,
    product_id: str,
    comparisons: list[tuple[Any, Any, Path, Path]],
) -> dict[str, Any]:
    """Diagnose plotted bands and compare current outputs to legacy outputs."""
    tile_results: list[dict[str, Any]] = []
    for current_wac, current_static, legacy_wac, legacy_static in comparisons:
        tile = {
            "zone": current_wac.zone,
            "zoom_level": current_wac.zoom_level,
            "tile_x": current_wac.tile_x,
            "tile_y": current_wac.tile_y,
        }
        modalities: dict[str, Any] = {}
        for modality, current_path, legacy_path, selector in (
            (
                "wac",
                current_wac.path,
                legacy_wac,
                {"band_number": WAC_BAND_NUMBER},
            ),
            (
                "static",
                current_static.path,
                legacy_static,
                {"band_name": STATIC_BAND_TO_PLOT},
            ),
        ):
            current_stats, current_band = read_band_diagnostics(
                current_path,
                **selector,
            )
            legacy_stats, legacy_band = read_band_diagnostics(
                legacy_path,
                **selector,
            )
            comparison = compare_diagnostic_bands(current_band, legacy_band)
            passed = bool(
                current_stats["all_nodata_occurrences_masked"]
                and legacy_stats["all_nodata_occurrences_masked"]
                and comparison["passed"]
            )
            modalities[modality] = {
                "passed": passed,
                "current": current_stats,
                "legacy": legacy_stats,
                "comparison": comparison,
            }
            print(
                f"Diagnostic {product_id} LTM{tile['zone']} "
                f"z{tile['zoom_level']} tile ({tile['tile_x']}, "
                f"{tile['tile_y']}) {modality.upper()}: "
                f"{'PASS' if passed else 'FAIL'}"
            )
            for label, stats in (("current", current_stats), ("legacy", legacy_stats)):
                print(
                    f"  {label}: nodata={stats['declared_nodata']}, "
                    f"raw=[{stats['raw_finite_min']}, "
                    f"{stats['raw_finite_max']}], "
                    f"nodata_pixels={stats['nodata_pixel_count']}, "
                    f"unmasked_nodata={stats['nodata_unmasked_count']}"
                )
            print(
                f"  comparison: mask_mismatches="
                f"{comparison.get('mask_mismatch_count')}, "
                f"value_mismatches={comparison.get('value_mismatch_count')}, "
                f"max_abs_diff={comparison.get('maximum_absolute_difference')}"
            )
        tile_results.append(
            {
                "tile": tile,
                "passed": all(result["passed"] for result in modalities.values()),
                "modalities": modalities,
            }
        )
    return {
        "product_id": product_id,
        "passed": all(result["passed"] for result in tile_results),
        "tiles": tile_results,
    }


def log_iteration_traceback(
    *,
    traceback_path: Path,
    phase: str,
    request_number: int,
    request_count: int,
    product_id: str,
    aoi: dict[str, float],
) -> None:
    """Append the active exception traceback with request context."""
    traceback_path.parent.mkdir(parents=True, exist_ok=True)
    with traceback_path.open("a", encoding="utf-8") as error_log:
        error_log.write("=" * 80)
        error_log.write("\n")
        error_log.write(
            f"Request {request_number}/{request_count}: WAC product {product_id}\n"
        )
        error_log.write(f"Phase: {phase}\n")
        error_log.write(f"AOI: {json.dumps(aoi, sort_keys=True)}\n\n")
        error_log.write(traceback.format_exc())
        error_log.write("\n")


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    entries = load_entries(config_path)

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    validate_path_pairs(
        {
            "WAC data directory": WAC_DATA_DIR,
            "static data directory": STATIC_DATA_DIR,
        },
        path_type="directory",
    )
    validate_path_pairs(
        {
            "WAC index": WAC_INDEX,
            "static index": STATIC_INDEX,
        },
        path_type="file",
    )

    output_dir = REPO_ROOT / "outputs" / "tiling" / RUN_ID
    output_dir.mkdir(parents=True, exist_ok=False)

    wac_source = TileSourceConfig(
        name="wac",
        data_dir=WAC_DATA_DIR,
        index_path=WAC_INDEX,
        location_field=LOCATION_FIELD,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    static_source = make_static_source(
        data_dir=STATIC_DATA_DIR,
        index_path=STATIC_INDEX,
        location_field=LOCATION_FIELD,
    )

    print(f"Repository root: {REPO_ROOT}")
    print(f"Configuration: {config_path}")
    print(f"Requests: {len(entries)}")
    print(f"Outputs: {output_dir}")

    plot_paths: list[Path] = []
    failed_steps: list[str] = []
    traceback_path = output_dir / "errors" / "tracebacks.log"
    for number, entry in enumerate(entries, start=1):
        product_id = entry["product_id"]
        aoi = entry["aoi"]
        print(f"\n[{number}/{len(entries)}] Tiling WAC product {product_id}")
        print(f"AOI: {aoi}")

        modern_records = None
        modern_pairs = None
        try:
            tile_config = TileConfig(
                output_dir=output_dir / "modern_cubes" / product_id,
                zoom_level=ZOOM_LEVEL,
                sources=(wac_source, static_source),
            )
            modern_records = create_tiles_for_aoi(
                tile_config,
                **aoi,
                selectors={"wac": product_id},
            )
            print("Modern tiler records:")
            print_record_summary(modern_records)

            modern_pairs = pair_dynamic_and_static(modern_records, "wac")
            print(f"Modern WAC/static tile pairs: {len(modern_pairs)}")
        except Exception:
            failed_steps.append(f"{product_id} (modern)")
            log_iteration_traceback(
                traceback_path=traceback_path,
                phase="modern tiling",
                request_number=number,
                request_count=len(entries),
                product_id=product_id,
                aoi=aoi,
            )
            plt.close("all")
            print(
                f"MODERN FAILED for {product_id}; attempting legacy tiling. "
                f"Traceback appended to {traceback_path}",
                file=sys.stderr,
            )

        legacy_paths = None
        try:
            legacy_paths = run_legacy_tiling(
                product_id=product_id,
                aoi=aoi,
                output_dir=output_dir / "legacy_cubes" / product_id,
            )
            print(f"Legacy Pipeline created {len(legacy_paths)} cube(s):")
            for legacy_path in legacy_paths:
                print(f"  {legacy_path}")
        except Exception:
            failed_steps.append(f"{product_id} (legacy)")
            log_iteration_traceback(
                traceback_path=traceback_path,
                phase="legacy Pipeline tiling",
                request_number=number,
                request_count=len(entries),
                product_id=product_id,
                aoi=aoi,
            )
            print(
                f"LEGACY FAILED for {product_id}; continuing with the next AOI. "
                f"Traceback appended to {traceback_path}",
                file=sys.stderr,
            )

        if modern_records is not None and legacy_paths is not None:
            print(
                "Cube inventory for comparison: "
                f"modern={len(modern_records)}, legacy={len(legacy_paths)}"
            )
        if modern_pairs is not None and legacy_paths is not None:
            try:
                comparisons = match_modern_and_legacy_cubes(
                    modern_pairs,
                    legacy_paths,
                )
                diagnostic_report = diagnose_modern_legacy_cubes(
                    product_id=product_id,
                    comparisons=comparisons,
                )
                diagnostic_path = (
                    output_dir / "diagnostics" / f"{product_id}.json"
                )
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostic_path.write_text(
                    json.dumps(diagnostic_report, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Diagnostic report: {diagnostic_path}")
                plot_path = (
                    output_dir
                    / "plots"
                    / f"wac_static_comparison_{product_id}.png"
                )
                figure = plot_modern_legacy_cube_comparison(
                    comparisons,
                    product_id=product_id,
                    dynamic_band_number=WAC_BAND_NUMBER,
                    static_band_name=STATIC_BAND_TO_PLOT,
                    output_path=plot_path,
                    max_tiles=MAX_PLOT_TILES,
                )
                plt.close(figure)
                plot_paths.append(plot_path)
            except Exception:
                failed_steps.append(f"{product_id} (diagnostics/visualization)")
                log_iteration_traceback(
                    traceback_path=traceback_path,
                    phase="modern/legacy comparison diagnostics/visualization",
                    request_number=number,
                    request_count=len(entries),
                    product_id=product_id,
                    aoi=aoi,
                )
                plt.close("all")
                print(
                    f"COMPARISON PLOT FAILED for {product_id}; continuing with "
                    f"the next AOI. Traceback appended to {traceback_path}",
                    file=sys.stderr,
                )
        else:
            print(
                f"Skipping comparison plot for {product_id} because one tiler "
                "did not produce a complete result.",
                file=sys.stderr,
            )

    print(f"\nProcessed {len(entries)} WAC/static tiling requests.")
    print(f"Created {len(plot_paths)} plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
    if failed_steps:
        print(
            f"Failed {len(failed_steps)} tiling step(s): "
            f"{', '.join(failed_steps)}",
            file=sys.stderr,
        )
        print(f"Full tracebacks: {traceback_path}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
