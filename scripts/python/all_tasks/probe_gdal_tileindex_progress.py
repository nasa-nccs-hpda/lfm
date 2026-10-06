#!/usr/bin/env python3
"""Probe GDAL TileIndex progress-callback behavior in the supported runtime."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from osgeo import gdal


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the JSON probe report.",
    )
    return parser.parse_args()


def _signature_text(callable_object) -> str | None:
    try:
        return str(inspect.signature(callable_object))
    except (TypeError, ValueError):
        return None


def _write_rasters(directory: Path, *, count: int) -> list[Path]:
    paths: list[Path] = []
    wkt = load_lunar_geographic_wkt()
    driver = gdal.GetDriverByName("GTiff")
    for index in range(count):
        path = directory / f"source_{index:02d}.tif"
        dataset = driver.Create(str(path), 4, 4, 1, gdal.GDT_Byte)
        if dataset is None:
            raise RuntimeError(f"Could not create probe raster: {path}")
        dataset.SetProjection(wkt)
        dataset.SetGeoTransform(
            (-20.0 + index, 0.1, 0.0, 85.0, 0.0, -0.1)
        )
        dataset.GetRasterBand(1).Fill(index + 1)
        dataset = None
        paths.append(path)
    return paths


def run_probe() -> dict[str, object]:
    gdal.UseExceptions()
    tileindex = getattr(gdal, "TileIndex", None)
    tileindex_options = getattr(gdal, "TileIndexOptions", None)
    gdaltindex_path = shutil.which("gdaltindex")
    gdaltindex_version = None
    if gdaltindex_path is not None:
        version = subprocess.run(
            [gdaltindex_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        gdaltindex_version = (version.stdout or version.stderr).strip() or None
    report: dict[str, object] = {
        "gdal_version": gdal.VersionInfo("--version"),
        "python_tileindex_available": tileindex is not None,
        "python_tileindex_options_available": tileindex_options is not None,
        "tileindex_signature": (
            _signature_text(tileindex) if tileindex is not None else None
        ),
        "tileindex_options_signature": (
            _signature_text(tileindex_options)
            if tileindex_options is not None
            else None
        ),
        "tileindex_options_doc_mentions_callback": (
            tileindex_options is not None
            and "callback" in (tileindex_options.__doc__ or "").casefold()
        ),
        "gdaltindex_executable": gdaltindex_path,
        "gdaltindex_version": gdaltindex_version,
    }
    if tileindex is None or tileindex_options is None:
        report.update(
            {
                "callback_option_accepted": False,
                "callback_option_error": (
                    "The installed osgeo.gdal bindings do not expose "
                    "TileIndex and TileIndexOptions."
                ),
                "callback_invoked": False,
                "callback_event_count": 0,
                "callback_events": [],
                "index_created": False,
                "feature_count": None,
            }
        )
        return report
    callback_events: list[dict[str, object]] = []

    def progress_callback(complete, message, callback_data):
        event = {
            "complete": float(complete),
            "message": str(message),
            "callback_data_matches": callback_data == "lfm-progress-probe",
        }
        callback_events.append(event)
        print(
            f"callback complete={event['complete']:.6f} "
            f"message={event['message']!r}"
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="lfm_tileindex_probe_") as temporary:
        root = Path(temporary)
        raster_paths = _write_rasters(root, count=5)
        index_path = root / "probe_index.gpkg"
        options_kwargs = {
            "format": "GPKG",
            "layerName": "probe_index",
            "locationFieldName": "location",
            "outputSRS": load_lunar_geographic_wkt(),
            "callback": progress_callback,
            "callback_data": "lfm-progress-probe",
        }
        try:
            options = tileindex_options(**options_kwargs)
        except TypeError as exc:
            report.update(
                {
                    "callback_option_accepted": False,
                    "callback_option_error": str(exc),
                    "callback_invoked": False,
                    "callback_event_count": 0,
                    "callback_events": [],
                    "index_created": False,
                    "feature_count": None,
                }
            )
            return report

        report["callback_option_accepted"] = True
        dataset = tileindex(
            str(index_path),
            [str(path) for path in raster_paths],
            options=options,
        )
        if dataset is None:
            raise RuntimeError(
                "GDAL accepted the callback option but failed to create the index: "
                f"{gdal.GetLastErrorMsg()}"
            )
        layer = dataset.GetLayer(0)
        feature_count = int(layer.GetFeatureCount()) if layer is not None else None
        layer = None
        dataset = None
        report.update(
            {
                "callback_invoked": bool(callback_events),
                "callback_event_count": len(callback_events),
                "callback_events": callback_events,
                "callback_reached_completion": any(
                    event["complete"] >= 1.0 for event in callback_events
                ),
                "callback_distinct_fractions": sorted(
                    {event["complete"] for event in callback_events}
                ),
                "index_created": index_path.is_file(),
                "feature_count": feature_count,
                "expected_feature_count": len(raster_paths),
            }
        )
    return report


def main() -> None:
    args = parse_args()
    report = run_probe()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
