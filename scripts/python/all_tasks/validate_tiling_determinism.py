#!/usr/bin/env python3
"""Validate deterministic modern tiling without parsing output filenames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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
    TileCubeRecord,
    TileSourceConfig,
    create_tiles_for_aoi,
)


DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)
DEFAULT_STATIC_DATA_DIR = Path("/explore/nobackup/projects/lfm/staticLinks")
DEFAULT_PRODUCT_ID = "M1187363083CE"
DEFAULT_BOUNDS = (1.3, 149.7, 1.1, 149.9)
INDEX_SUFFIXES = {
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".sbn",
    ".sbx",
    ".gpkg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run modern mixed-source tiling twice and verify deterministic "
            "structured results, files, and read-only index behavior."
        )
    )
    parser.add_argument("--wac-data-dir", type=Path, default=DEFAULT_WAC_DATA_DIR)
    parser.add_argument("--wac-index", type=Path, default=None)
    parser.add_argument("--wac-index-layer", default=None)
    parser.add_argument(
        "--static-data-dir",
        type=Path,
        default=DEFAULT_STATIC_DATA_DIR,
    )
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
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_index_artifact(path: Path) -> bool:
    return path.suffix.lower() in INDEX_SUFFIXES or path.name.lower().endswith(
        (".shp.xml", ".gpkg-wal", ".gpkg-shm", ".gpkg-journal")
    )


def index_family(index_path: Path) -> tuple[Path, ...]:
    if index_path.suffix.lower() == ".gpkg":
        related = [
            index_path,
            Path(f"{index_path}-wal"),
            Path(f"{index_path}-shm"),
            Path(f"{index_path}-journal"),
        ]
        return tuple(path for path in related if path.exists())
    return tuple(
        sorted(
            path
            for path in index_path.parent.glob(f"{index_path.stem}.*")
            if is_index_artifact(path)
        )
    )


def snapshot_paths(paths: tuple[Path, ...]) -> dict[str, dict[str, int | str]]:
    return {
        str(path): {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def visible_index_artifacts(directory: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and is_index_artifact(path)
        )
    )


def require_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.iterdir())
    if existing:
        raise FileExistsError(
            f"Determinism output directory must start empty: {path}; "
            f"found {existing[:5]}."
        )


def record_identity(record: TileCubeRecord) -> dict[str, Any]:
    """Return identity from structured fields, never from filename parsing."""
    return {
        "source_name": record.source_name,
        "product_id": record.product_id,
        "zone": record.zone,
        "zoom_level": record.zoom_level,
        "tile_x": record.tile_x,
        "tile_y": record.tile_y,
    }


def record_details(record: TileCubeRecord) -> dict[str, Any]:
    if not record.path.is_file():
        raise AssertionError(f"Structured result path does not exist: {record.path}")
    return {
        "identity": record_identity(record),
        "filename": record.path.name,
        "band_names": list(record.band_names),
        "crs_wkt": record.crs_wkt,
        "nodata_values": list(record.nodata_values),
        "file_size_bytes": record.path.stat().st_size,
        "sha256": sha256_file(record.path),
    }


def expected_order_key(
    details: dict[str, Any],
    source_order: dict[str, int],
) -> tuple[str, int, int, int]:
    identity = details["identity"]
    return (
        identity["zone"],
        identity["tile_y"],
        identity["tile_x"],
        source_order[identity["source_name"]],
    )


def build_config(
    *,
    output_dir: Path,
    zoom_level: int,
    wac_data_dir: Path,
    wac_index: Path,
    wac_index_layer: str | None,
    static_data_dir: Path,
    static_index: Path,
    static_index_layer: str | None,
    location_field: str,
) -> TileConfig:
    return TileConfig(
        output_dir=output_dir,
        zoom_level=zoom_level,
        sources=(
            TileSourceConfig(
                name="wac",
                data_dir=wac_data_dir,
                index_path=wac_index,
                index_layer=wac_index_layer,
                location_field=location_field,
                selection_mode="product_id",
                resampling="bilinear",
                preserve_source_nodata=True,
            ),
            TileSourceConfig(
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
            ),
        ),
    )


def run_once(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    wac_index: Path,
    static_index: Path,
) -> list[dict[str, Any]]:
    config = build_config(
        output_dir=output_dir,
        zoom_level=args.zoom_level,
        wac_data_dir=args.wac_data_dir,
        wac_index=wac_index,
        wac_index_layer=args.wac_index_layer,
        static_data_dir=args.static_data_dir,
        static_index=static_index,
        static_index_layer=args.static_index_layer,
        location_field=args.location_field,
    )
    if any(source.resampling != "bilinear" for source in config.sources):
        raise AssertionError("Every tiling source must use bilinear resampling.")
    records = create_tiles_for_aoi(
        config,
        ul_lat=args.bounds[0],
        ul_lon=args.bounds[1],
        lr_lat=args.bounds[2],
        lr_lon=args.bounds[3],
        selectors={"wac": args.product_id},
    )
    expected_record_count = args.expected_tile_count * len(config.sources)
    if len(records) != expected_record_count:
        raise AssertionError(
            f"Expected {expected_record_count} records, got {len(records)}."
        )
    source_order = {
        source.name: index for index, source in enumerate(config.sources)
    }
    details = [record_details(record) for record in records]
    if details != sorted(
        details,
        key=lambda item: expected_order_key(item, source_order),
    ):
        raise AssertionError("TileCubeRecord order does not follow tile/source order.")
    counts = Counter(item["identity"]["source_name"] for item in details)
    expected_counts = {
        source.name: args.expected_tile_count for source in config.sources
    }
    if dict(counts) != expected_counts:
        raise AssertionError(
            f"Unexpected per-source record counts: {dict(counts)}; "
            f"expected {expected_counts}."
        )
    return details


def main() -> int:
    args = parse_args()
    if args.expected_tile_count < 1:
        raise ValueError("--expected-tile-count must be positive.")
    wac_index = args.wac_index or args.wac_data_dir / "output_index.shp"
    static_index = args.static_index or args.static_data_dir / "db2.shp"
    report_path = args.report_path or args.output_dir / "determinism_report.json"

    for name, directory in (
        ("WAC data directory", args.wac_data_dir),
        ("static data directory", args.static_data_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {directory}")
    for name, index_path in (
        ("WAC index", wac_index),
        ("static index", static_index),
    ):
        if not index_path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {index_path}")

    run_one_dir = args.output_dir / "run_1"
    run_two_dir = args.output_dir / "run_2"
    require_empty_directory(run_one_dir)
    require_empty_directory(run_two_dir)

    declared_index_paths_before = tuple(
        dict.fromkeys(index_family(wac_index) + index_family(static_index))
    )
    indexes_before = snapshot_paths(declared_index_paths_before)
    visible_indexes_before = {
        str(args.wac_data_dir): visible_index_artifacts(args.wac_data_dir),
        str(args.static_data_dir): visible_index_artifacts(args.static_data_dir),
    }

    print("Running deterministic tiling pass 1...", flush=True)
    run_one = run_once(
        output_dir=run_one_dir,
        args=args,
        wac_index=wac_index,
        static_index=static_index,
    )
    print("Running deterministic tiling pass 2...", flush=True)
    run_two = run_once(
        output_dir=run_two_dir,
        args=args,
        wac_index=wac_index,
        static_index=static_index,
    )

    declared_index_paths_after = tuple(
        dict.fromkeys(index_family(wac_index) + index_family(static_index))
    )
    indexes_after = snapshot_paths(declared_index_paths_after)
    visible_indexes_after = {
        str(args.wac_data_dir): visible_index_artifacts(args.wac_data_dir),
        str(args.static_data_dir): visible_index_artifacts(args.static_data_dir),
    }
    output_index_artifacts = sorted(
        str(path.relative_to(args.output_dir))
        for path in args.output_dir.rglob("*")
        if path.is_file() and is_index_artifact(path)
    )

    record_order_matches = [item["identity"] for item in run_one] == [
        item["identity"] for item in run_two
    ]
    record_metadata_matches = run_one == run_two
    file_hashes_match = [item["sha256"] for item in run_one] == [
        item["sha256"] for item in run_two
    ]
    declared_indexes_unchanged = indexes_before == indexes_after
    visible_indexes_unchanged = visible_indexes_before == visible_indexes_after
    no_output_indexes_created = not output_index_artifacts
    passed = all(
        (
            record_order_matches,
            record_metadata_matches,
            file_hashes_match,
            declared_indexes_unchanged,
            visible_indexes_unchanged,
            no_output_indexes_created,
        )
    )

    report = {
        "status": "passed" if passed else "differences_detected",
        "query": {
            "bounds": list(args.bounds),
            "zoom_level": args.zoom_level,
            "product_id": args.product_id,
        },
        "contract": {
            "result_pairing": "TileCubeRecord fields (no output filename parsing)",
            "resampling": "bilinear",
            "declared_indexes": [str(wac_index), str(static_index)],
            "index_creation": "forbidden during tiling",
        },
        "checks": {
            "record_order_matches": record_order_matches,
            "record_metadata_matches": record_metadata_matches,
            "file_sha256_matches": file_hashes_match,
            "declared_indexes_unchanged": declared_indexes_unchanged,
            "visible_index_inventory_unchanged": visible_indexes_unchanged,
            "no_output_indexes_created": no_output_indexes_created,
        },
        "index_state_before": indexes_before,
        "index_state_after": indexes_after,
        "visible_index_inventory_before": visible_indexes_before,
        "visible_index_inventory_after": visible_indexes_after,
        "output_index_artifacts": output_index_artifacts,
        "run_1": run_one,
        "run_2": run_two,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nDeterminism validation summary:")
    for name, value in report["checks"].items():
        print(f"  {name}: {'PASS' if value else 'FAIL'}")
    print(f"Report: {report_path}")
    if not passed:
        raise AssertionError(
            "Repeated modern tiling runs were not deterministic or modified "
            f"index state. Inspect {report_path}."
        )
    print("Tiling determinism validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
