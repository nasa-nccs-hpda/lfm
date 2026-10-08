#!/usr/bin/env python3
"""Compare LFM raster indexes with the supported ``gdaltindex`` CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from osgeo import gdal, ogr, osr


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt
from lfm.data_processing.tiling.vector_index_builder import (
    VectorIndexBuildConfig,
    _raster_footprint,
    create_vector_index,
)


BOUNDS_ABSOLUTE_TOLERANCE = 0.02
DEFAULT_GDALTINDEX_AREA_RELATIVE_TOLERANCE = 0.02
DENSE_REFERENCE_EDGE_SAMPLES = 201
DENSE_REFERENCE_BOUNDS_ABSOLUTE_TOLERANCE = 0.001
DENSE_REFERENCE_AREA_RELATIVE_TOLERANCE = 0.001
QueryBounds = tuple[float, float, float, float]
QuerySpec = tuple[str, QueryBounds, tuple[str, ...]]


@dataclass(frozen=True)
class Fixture:
    name: str
    raster_paths: tuple[Path, ...]
    query_rectangles: tuple[QuerySpec, ...]
    gdaltindex_area_is_acceptance_gate: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _load_tms_crs(repo_root: Path, filename: str) -> str:
    definition = json.loads(
        (repo_root / "TMS" / "RG" / filename).read_text(encoding="utf-8")
    )
    return str(definition["crs"])


def _write_raster(
    path: Path,
    *,
    projection: str,
    geotransform: tuple[float, float, float, float, float, float],
) -> Path:
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        100,
        100,
        1,
        gdal.GDT_Byte,
    )
    if dataset is None:
        raise RuntimeError(f"Could not create fixture raster: {path}")
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geotransform)
    dataset.GetRasterBand(1).Fill(1)
    dataset = None
    return path.resolve()


def _build_fixtures(root: Path, repo_root: Path) -> tuple[Fixture, ...]:
    geographic_wkt = load_lunar_geographic_wkt()
    ltm_wkt = _load_tms_crs(repo_root, "tms_LTM_1NRG.json")
    polar_wkt = _load_tms_crs(repo_root, "tms_LPS_NRG.json")

    ltm_dir = root / "ordinary_ltm"
    ltm_dir.mkdir()
    ltm_paths = tuple(
        sorted(
            (
                _write_raster(
                    ltm_dir / "b_ltm.tif",
                    projection=ltm_wkt,
                    geotransform=(270_000.0, 100.0, 0.0, 120_000.0, 0.0, -100.0),
                ),
                _write_raster(
                    ltm_dir / "a_ltm.tif",
                    projection=ltm_wkt,
                    geotransform=(240_000.0, 100.0, 0.0, 120_000.0, 0.0, -100.0),
                ),
            ),
            key=str,
        )
    )

    polar_dir = root / "polar"
    polar_dir.mkdir()
    polar_paths = (
        _write_raster(
            polar_dir / "polar_north.tif",
            projection=polar_wkt,
            geotransform=(550_000.0, 1_000.0, 0.0, 650_000.0, 0.0, -1_000.0),
        ),
    )

    seam_dir = root / "longitude_seam"
    seam_dir.mkdir()
    seam_paths = tuple(
        sorted(
            (
                _write_raster(
                    seam_dir / "west_of_seam.tif",
                    projection=geographic_wkt,
                    geotransform=(-180.0, 0.008, 0.0, 1.0, 0.0, -0.02),
                ),
                _write_raster(
                    seam_dir / "east_of_seam.tif",
                    projection=geographic_wkt,
                    geotransform=(179.2, 0.008, 0.0, 1.0, 0.0, -0.02),
                ),
            ),
            key=str,
        )
    )
    seam_queries = (
        (
            "east_side",
            (179.4, -0.5, 179.8, 0.5),
            ("east_of_seam.tif",),
        ),
        (
            "west_side",
            (-179.8, -0.5, -179.4, 0.5),
            ("west_of_seam.tif",),
        ),
    )
    return (
        Fixture("ordinary_ltm", ltm_paths, ()),
        Fixture(
            "polar",
            polar_paths,
            (),
            False,
        ),
        Fixture("longitude_seam", seam_paths, seam_queries),
    )


def _run_gdaltindex(
    executable: str,
    *,
    index_path: Path,
    raster_paths: tuple[Path, ...],
    output_wkt: str,
) -> dict[str, object]:
    command = [
        executable,
        "-f",
        "GPKG",
        "-lyr_name",
        "oracle",
        "-tileindex",
        "location",
        "-write_absolute_path",
        "-t_srs",
        output_wkt,
        str(index_path),
        *(str(path) for path in raster_paths),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    details = {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"gdaltindex failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return details


def _vertex_count(geometry) -> int:
    child_count = geometry.GetGeometryCount()
    if child_count:
        return sum(
            _vertex_count(geometry.GetGeometryRef(index))
            for index in range(child_count)
        )
    return int(geometry.GetPointCount())


def _geometry_record(location: str, geometry) -> dict[str, Any]:
    envelope = geometry.GetEnvelope()
    return {
        "location": location,
        "valid": bool(geometry.IsValid()),
        "empty": bool(geometry.IsEmpty()),
        "bounds": [
            float(envelope[0]),
            float(envelope[2]),
            float(envelope[1]),
            float(envelope[3]),
        ],
        "area": float(geometry.GetArea()),
        "vertex_count": _vertex_count(geometry),
    }


def _read_index(index_path: Path, *, layer_name: str) -> dict[str, Any]:
    dataset = gdal.OpenEx(str(index_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    if dataset is None:
        raise RuntimeError(f"Could not open vector index: {index_path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Could not open layer {layer_name!r} in {index_path}")
    spatial_reference = layer.GetSpatialRef()
    if spatial_reference is None:
        raise RuntimeError(f"Layer {layer_name!r} has no CRS: {index_path}")
    spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    records: list[dict[str, Any]] = []
    layer.ResetReading()
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            raise RuntimeError(f"Feature {feature.GetFID()} has no geometry")
        records.append(
            _geometry_record(
                str(feature.GetField("location")),
                geometry,
            )
        )
    result = {
        "driver": dataset.GetDriver().ShortName,
        "layer_name": layer.GetName(),
        "feature_count": int(layer.GetFeatureCount()),
        "crs_wkt": spatial_reference.ExportToWkt(),
        "records": records,
    }
    layer = None
    dataset = None
    return result


def _dense_reference_records(
    raster_paths: tuple[Path, ...],
) -> dict[str, dict[str, Any]]:
    output_srs = osr.SpatialReference()
    if output_srs.ImportFromWkt(load_lunar_geographic_wkt()) != 0:
        raise ValueError("Could not import the repository lunar geographic CRS.")
    output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    records: dict[str, dict[str, Any]] = {}
    for path in raster_paths:
        geometry = _raster_footprint(
            path,
            output_srs=output_srs,
            gdal=gdal,
            ogr=ogr,
            osr=osr,
            samples_per_edge=DENSE_REFERENCE_EDGE_SAMPLES,
        )
        records[str(path)] = _geometry_record(str(path), geometry)
    return records


def _query_index(
    index_path: Path,
    *,
    layer_name: str,
    bounds: QueryBounds,
) -> tuple[str, ...]:
    dataset = gdal.OpenEx(str(index_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    if dataset is None:
        raise RuntimeError(f"Could not open vector index: {index_path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Could not open layer {layer_name!r} in {index_path}")
    min_x, min_y, max_x, max_y = bounds
    layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
    layer.ResetReading()
    selected = tuple(
        sorted(Path(str(feature.GetField("location"))).name for feature in layer)
    )
    layer.SetSpatialFilter(None)
    layer = None
    dataset = None
    return selected


def _spatial_references_match(first_wkt: str, second_wkt: str) -> bool:
    first = osr.SpatialReference()
    second = osr.SpatialReference()
    if first.ImportFromWkt(first_wkt) != 0 or second.ImportFromWkt(second_wkt) != 0:
        return False
    first.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    second.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return bool(first.IsSame(second))


def _automatic_queries(
    records: list[dict[str, Any]],
) -> tuple[QuerySpec, ...]:
    min_x = min(record["bounds"][0] for record in records)
    min_y = min(record["bounds"][1] for record in records)
    max_x = max(record["bounds"][2] for record in records)
    max_y = max(record["bounds"][3] for record in records)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    expected_all = tuple(sorted(Path(record["location"]).name for record in records))
    individual_queries = []
    for record in records:
        record_min_x, record_min_y, record_max_x, record_max_y = record["bounds"]
        record_width = max(record_max_x - record_min_x, 1e-6)
        record_height = max(record_max_y - record_min_y, 1e-6)
        center_x = (record_min_x + record_max_x) / 2.0
        center_y = (record_min_y + record_max_y) / 2.0
        individual_queries.append(
            (
                f"center_{Path(record['location']).name}",
                (
                    center_x - record_width * 0.05,
                    center_y - record_height * 0.05,
                    center_x + record_width * 0.05,
                    center_y + record_height * 0.05,
                ),
                (Path(record["location"]).name,),
            )
        )
    return (
        (
            "all_records",
            (
                min_x - width * 0.05,
                min_y - height * 0.05,
                max_x + width * 0.05,
                max_y + height * 0.05,
            ),
            expected_all,
        ),
        (
            "outside",
            (
                max_x + width,
                max_y + height,
                max_x + width * 1.5,
                max_y + height * 1.5,
            ),
            (),
        ),
        *individual_queries,
    )


def _compare_fixture(
    fixture: Fixture,
    *,
    gdaltindex_executable: str,
) -> dict[str, Any]:
    directory = fixture.raster_paths[0].parent
    custom_path = directory / "lfm_index.gpkg"
    oracle_path = directory / "gdaltindex_oracle.gpkg"
    config = VectorIndexBuildConfig(
        data_dir=directory,
        index_path=custom_path,
        layer_name="lfm",
    )
    create_vector_index(
        config,
        raster_paths=fixture.raster_paths,
        progress=False,
    )
    gdaltindex_run = _run_gdaltindex(
        gdaltindex_executable,
        index_path=oracle_path,
        raster_paths=fixture.raster_paths,
        output_wkt=load_lunar_geographic_wkt(),
    )
    custom = _read_index(custom_path, layer_name="lfm")
    oracle = _read_index(oracle_path, layer_name="oracle")
    dense_reference_by_location = _dense_reference_records(fixture.raster_paths)
    failures: list[str] = []

    output_wkt = load_lunar_geographic_wkt()
    driver_matches = custom["driver"] == oracle["driver"] == "GPKG"
    if not driver_matches:
        failures.append("driver differs")
    crs_matches = (
        _spatial_references_match(custom["crs_wkt"], oracle["crs_wkt"])
        and _spatial_references_match(custom["crs_wkt"], output_wkt)
        and _spatial_references_match(oracle["crs_wkt"], output_wkt)
    )
    if not crs_matches:
        failures.append("CRS differs")
    feature_count_matches = (
        custom["feature_count"]
        == oracle["feature_count"]
        == len(fixture.raster_paths)
    )
    if not feature_count_matches:
        failures.append("feature count differs")

    custom_locations = tuple(record["location"] for record in custom["records"])
    oracle_locations = tuple(record["location"] for record in oracle["records"])
    expected_locations = tuple(str(path) for path in fixture.raster_paths)
    location_values_match = (
        set(custom_locations)
        == set(oracle_locations)
        == set(expected_locations)
    )
    record_order_matches = (
        custom_locations == oracle_locations == expected_locations
    )
    if not location_values_match:
        failures.append("location values differ")
    if not record_order_matches:
        failures.append("record order differs")

    oracle_by_location = {
        record["location"]: record for record in oracle["records"]
    }
    record_comparisons: list[dict[str, Any]] = []
    for custom_record in custom["records"]:
        oracle_record = oracle_by_location.get(custom_record["location"])
        if oracle_record is None:
            continue
        bound_differences = [
            abs(first - second)
            for first, second in zip(
                custom_record["bounds"],
                oracle_record["bounds"],
            )
        ]
        maximum_bound_difference = max(bound_differences)
        area_denominator = max(
            abs(custom_record["area"]),
            abs(oracle_record["area"]),
            1e-12,
        )
        relative_area_difference = (
            abs(custom_record["area"] - oracle_record["area"])
            / area_denominator
        )
        dense_reference = dense_reference_by_location[custom_record["location"]]
        reference_bound_differences = [
            abs(first - second)
            for first, second in zip(
                custom_record["bounds"],
                dense_reference["bounds"],
            )
        ]
        reference_maximum_bound_difference = max(reference_bound_differences)
        reference_area_denominator = max(
            abs(custom_record["area"]),
            abs(dense_reference["area"]),
            1e-12,
        )
        reference_relative_area_difference = (
            abs(custom_record["area"] - dense_reference["area"])
            / reference_area_denominator
        )
        comparison = {
            "location": custom_record["location"],
            "custom": custom_record,
            "gdaltindex": oracle_record,
            "dense_reference": dense_reference,
            "maximum_bound_difference": maximum_bound_difference,
            "relative_area_difference": relative_area_difference,
            "bounds_within_tolerance": (
                maximum_bound_difference <= BOUNDS_ABSOLUTE_TOLERANCE
            ),
            "gdaltindex_area_is_acceptance_gate": (
                fixture.gdaltindex_area_is_acceptance_gate
            ),
            "gdaltindex_area_within_default_tolerance": (
                relative_area_difference
                <= DEFAULT_GDALTINDEX_AREA_RELATIVE_TOLERANCE
            ),
            "gdaltindex_area_acceptance_passes": (
                not fixture.gdaltindex_area_is_acceptance_gate
                or relative_area_difference
                <= DEFAULT_GDALTINDEX_AREA_RELATIVE_TOLERANCE
            ),
            "dense_reference_maximum_bound_difference": (
                reference_maximum_bound_difference
            ),
            "dense_reference_relative_area_difference": (
                reference_relative_area_difference
            ),
            "dense_reference_bounds_within_tolerance": (
                reference_maximum_bound_difference
                <= DENSE_REFERENCE_BOUNDS_ABSOLUTE_TOLERANCE
            ),
            "dense_reference_area_within_tolerance": (
                reference_relative_area_difference
                <= DENSE_REFERENCE_AREA_RELATIVE_TOLERANCE
            ),
        }
        record_comparisons.append(comparison)
        if not custom_record["valid"] or custom_record["empty"]:
            failures.append(f"invalid LFM geometry: {custom_record['location']}")
        if not oracle_record["valid"] or oracle_record["empty"]:
            failures.append(
                f"invalid gdaltindex geometry: {oracle_record['location']}"
            )
        if not comparison["bounds_within_tolerance"]:
            failures.append(f"bounds tolerance exceeded: {custom_record['location']}")
        if not comparison["gdaltindex_area_acceptance_passes"]:
            failures.append(f"area tolerance exceeded: {custom_record['location']}")
        if not comparison["dense_reference_bounds_within_tolerance"]:
            failures.append(
                "dense-reference bounds tolerance exceeded: "
                f"{custom_record['location']}"
            )
        if not comparison["dense_reference_area_within_tolerance"]:
            failures.append(
                "dense-reference area tolerance exceeded: "
                f"{custom_record['location']}"
            )

    queries = (*_automatic_queries(custom["records"]), *fixture.query_rectangles)
    query_comparisons: list[dict[str, Any]] = []
    for name, bounds, expected in queries:
        custom_selected = _query_index(
            custom_path,
            layer_name="lfm",
            bounds=bounds,
        )
        oracle_selected = _query_index(
            oracle_path,
            layer_name="oracle",
            bounds=bounds,
        )
        comparison = {
            "name": name,
            "bounds": list(bounds),
            "expected": list(expected),
            "custom": list(custom_selected),
            "gdaltindex": list(oracle_selected),
            "matches": custom_selected == oracle_selected == expected,
        }
        query_comparisons.append(comparison)
        if not comparison["matches"]:
            failures.append(f"AOI query differs: {name}")

    return {
        "name": fixture.name,
        "raster_order": [str(path) for path in fixture.raster_paths],
        "gdaltindex_run": gdaltindex_run,
        "gdaltindex_area_is_acceptance_gate": (
            fixture.gdaltindex_area_is_acceptance_gate
        ),
        "driver_matches": driver_matches,
        "crs_matches": crs_matches,
        "feature_count_matches": feature_count_matches,
        "location_values_match": location_values_match,
        "record_order_matches": record_order_matches,
        "record_comparisons": record_comparisons,
        "query_comparisons": query_comparisons,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    gdaltindex_executable = shutil.which("gdaltindex")
    if gdaltindex_executable is None:
        raise RuntimeError("gdaltindex is not available on PATH.")
    repo_root = Path(__file__).resolve().parents[3]
    fixture_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lfm_vector_index_comparison_") as temp:
        fixtures = _build_fixtures(Path(temp), repo_root)
        for fixture in fixtures:
            print(f"Comparing fixture: {fixture.name}", flush=True)
            try:
                fixture_reports.append(
                    _compare_fixture(
                        fixture,
                        gdaltindex_executable=gdaltindex_executable,
                    )
                )
            except Exception as exc:
                fixture_reports.append(
                    {
                        "name": fixture.name,
                        "passed": False,
                        "failures": [f"{type(exc).__name__}: {exc}"],
                        "traceback": traceback.format_exc(),
                    }
                )

    version_result = subprocess.run(
        [gdaltindex_executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    gdaltindex_version = (
        version_result.stdout or version_result.stderr
    ).strip() or None

    report = {
        "gdal_version": gdal.VersionInfo("--version"),
        "gdaltindex_executable": gdaltindex_executable,
        "gdaltindex_version": gdaltindex_version,
        "bounds_absolute_tolerance_degrees": BOUNDS_ABSOLUTE_TOLERANCE,
        "gdaltindex_area_relative_tolerance": (
            DEFAULT_GDALTINDEX_AREA_RELATIVE_TOLERANCE
        ),
        "dense_reference_edge_samples": DENSE_REFERENCE_EDGE_SAMPLES,
        "dense_reference_bounds_absolute_tolerance_degrees": (
            DENSE_REFERENCE_BOUNDS_ABSOLUTE_TOLERANCE
        ),
        "dense_reference_area_relative_tolerance": (
            DENSE_REFERENCE_AREA_RELATIVE_TOLERANCE
        ),
        "fixtures": fixture_reports,
        "passed": all(fixture["passed"] for fixture in fixture_reports),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.report}")
    if not report["passed"]:
        raise AssertionError(
            "Vector-index semantic comparison failed; inspect the JSON report."
        )


if __name__ == "__main__":
    main()
