#!/usr/bin/env python
"""Validate complete WAC chips through reference-directory and explicit-AOI APIs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

DEFAULT_WAC_DATA_DIR = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--label-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wac-data-dir", type=Path, default=DEFAULT_WAC_DATA_DIR)
    parser.add_argument("--wac-index", type=Path, default=None)
    parser.add_argument("--location-field", default="location")
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--zoom-level", type=int, default=5)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "live", "log"),
        default="auto",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Default: OUTPUT_ROOT/c8_1_wac_validation.json",
    )
    return parser.parse_args()


def _require_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_config(output_root: Path, args: argparse.Namespace) -> Any:
    from lfm.model import (
        AcquisitionGroupConfig,
        ChipConfig,
        NoSplitConfig,
        OutputModalityConfig,
        TileConfig,
        TileSourceConfig,
    )

    source = TileSourceConfig(
        name="wac",
        data_dir=args.wac_data_dir,
        index_path=args.wac_index,
        location_field=args.location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    return ChipConfig(
        output_root=output_root,
        intermediate_root=output_root / ".intermediate",
        label_source=args.label_source,
        acquisition_groups=(
            AcquisitionGroupConfig(
                "wac_grid",
                TileConfig(
                    output_dir=output_root / ".tiling-template",
                    zoom_level=args.zoom_level,
                    sources=(source,),
                ),
            ),
        ),
        output_modalities=(
            OutputModalityConfig("wac_grid", "wac", "wac"),
        ),
        split_config=NoSplitConfig(),
        sample_limit=args.sample_limit,
        intermediate_retention="on_failure",
    )


def _require_success(batch: Any, case_name: str) -> None:
    failed = [result for result in batch.results if result.status != "success"]
    if not failed:
        return
    details = []
    for result in failed:
        errors = [
            f"{item.stage}/{item.code}: {item.message}"
            for item in result.diagnostics
            if item.severity == "error"
        ]
        details.append(
            f"{result.request.sample_id} [{result.status}]: "
            + ("; ".join(errors) or result.message or "no error diagnostic")
        )
    raise AssertionError(
        f"{case_name} produced {len(failed)} non-successful sample(s):\n  "
        + "\n  ".join(details)
    )


def _same_crs(actual_wkt: str, expected_wkt: str) -> bool:
    from osgeo import osr

    actual = osr.SpatialReference()
    expected = osr.SpatialReference()
    return (
        actual.ImportFromWkt(actual_wkt) == 0
        and expected.ImportFromWkt(expected_wkt) == 0
        and bool(actual.IsSame(expected))
    )


def _read_raster(path: Path, request: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from osgeo import gdal, gdalconst

    dataset = gdal.Open(str(path), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise AssertionError(f"GDAL could not open published chip: {path}")
    grid = request.target_grid
    if (dataset.RasterXSize, dataset.RasterYSize) != (grid.width, grid.height):
        raise AssertionError(
            f"Published chip dimensions disagree with target grid: {path}"
        )
    transform = tuple(float(value) for value in dataset.GetGeoTransform())
    if not np.allclose(transform, grid.transform, rtol=0.0, atol=1e-9):
        raise AssertionError(
            f"Published chip transform disagrees with target grid: {path}"
        )
    projection = dataset.GetProjection()
    if not projection or not _same_crs(projection, grid.crs_wkt):
        raise AssertionError(f"Published chip CRS disagrees with target grid: {path}")
    if dataset.RasterCount != 7:
        raise AssertionError(
            f"Expected seven WAC bands in {path}, got {dataset.RasterCount}."
        )

    arrays = []
    band_reports = []
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        array = np.asarray(band.ReadAsArray())
        arrays.append(array)
        valid = np.isfinite(array)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            if np.isnan(float(nodata)):
                valid &= ~np.isnan(array)
            else:
                valid &= array != nodata
        valid &= np.asarray(band.GetMaskBand().ReadAsArray()) != 0
        values = array[valid]
        if values.size == 0:
            raise AssertionError(f"WAC band {index} contains no valid pixels: {path}")
        reported_nodata: float | str | None = None
        if nodata is not None:
            reported_nodata = float(nodata)
            if not np.isfinite(reported_nodata):
                reported_nodata = str(reported_nodata)
        band_reports.append(
            {
                "index": index,
                "name": band.GetMetadataItem("Name") or band.GetDescription() or None,
                "dtype": str(array.dtype),
                "nodata": reported_nodata,
                "valid_pixels": int(values.size),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean(dtype=np.float64)),
            }
        )
    raster = np.stack(arrays)
    report = {
        "path": str(path),
        "sha256": _sha256(path),
        "file_size_bytes": path.stat().st_size,
        "shape": list(raster.shape),
        "geotransform": list(transform),
        "bands": band_reports,
    }
    dataset = None
    return raster, report


def _read_label(
    path: Path,
    expected_shape: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"])
            count = int(np.asarray(archive["num_craters"]).item())
            present_ids = {int(value) for value in np.unique(mask[mask > 0])}
            report: dict[str, Any] = {
                "kind": "instance",
                "arrays": sorted(archive.files),
                "num_craters": count,
                "present_instance_ids": sorted(present_ids),
                "missing_instance_ids": sorted(
                    set(range(1, count + 1)) - present_ids
                ),
            }
    elif path.suffix.casefold() == ".npy":
        mask = np.asarray(np.load(path, allow_pickle=False))
        report = {
            "kind": "semantic",
            "class_values": [int(value) for value in np.unique(mask)],
        }
    else:
        raise AssertionError(f"Unsupported published label suffix: {path}")
    if tuple(mask.shape) != expected_shape:
        raise AssertionError(
            f"Published label shape {mask.shape} does not match "
            f"{expected_shape}: {path}"
        )
    report.update(
        {
            "path": str(path),
            "sha256": _sha256(path),
            "file_size_bytes": path.stat().st_size,
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
        }
    )
    return mask, report


def _display_limits(array: Any) -> tuple[float, float]:
    import numpy as np

    valid = np.asarray(array)[np.isfinite(array)]
    if valid.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(valid, (2.0, 98.0))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _write_inspection_plot(
    path: Path,
    sample_id: str,
    reference_path: Path,
    generated: Any,
    label: Any,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal, gdalconst

    reference = gdal.Open(str(reference_path), gdalconst.GA_ReadOnly)
    if reference is None:
        raise AssertionError(f"GDAL could not open reference chip: {reference_path}")
    reference_band = np.asarray(reference.GetRasterBand(1).ReadAsArray())
    reference = None
    generated_band = np.asarray(generated[0])
    ref_limits = _display_limits(reference_band)
    generated_limits = _display_limits(generated_band)
    colored_label = np.ma.masked_where(label == 0, np.mod(label, 20))

    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(reference_band, cmap="gray", vmin=ref_limits[0], vmax=ref_limits[1])
    axes[0].set_title("Reference band 1")
    axes[1].imshow(
        generated_band,
        cmap="gray",
        vmin=generated_limits[0],
        vmax=generated_limits[1],
    )
    axes[1].set_title("Generated WAC band 1")
    axes[2].imshow(colored_label, cmap="tab20", interpolation="nearest")
    axes[2].set_title("Label (tab20)")
    axes[3].imshow(
        generated_band,
        cmap="gray",
        vmin=generated_limits[0],
        vmax=generated_limits[1],
    )
    axes[3].imshow(colored_label, cmap="tab20", alpha=0.45, interpolation="nearest")
    axes[3].set_title("Generated + label")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _inspect_batch(
    batch: Any,
    *,
    plot_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for prepared, result in zip(
        batch.prepared_requests,
        batch.results,
        strict=True,
    ):
        if result.chip_path is None or result.label_path is None:
            raise AssertionError(
                f"Successful sample lacks a published pair: {result.request.sample_id}"
            )
        generated, raster_report = _read_raster(result.chip_path, result.request)
        label, label_report = _read_label(
            result.label_path,
            (result.request.target_grid.height, result.request.target_grid.width),
        )
        source_label = prepared.preflight.resolved_label_path
        if source_label is None or _sha256(source_label) != label_report["sha256"]:
            raise AssertionError(
                f"Published label is not a byte-identical copy: {result.label_path}"
            )
        plot_path = None
        if plot_dir is not None:
            if result.request.reference_path is None:
                raise AssertionError(
                    "Inspection plotting requires reference provenance."
                )
            plot_path = plot_dir / f"{result.request.sample_id}.png"
            _write_inspection_plot(
                plot_path,
                result.request.sample_id,
                result.request.reference_path,
                generated,
                label,
            )
        samples[result.request.sample_id] = {
            "status": result.status,
            "elapsed_seconds": result.elapsed_seconds,
            "request": {
                "reference_path": (
                    None
                    if result.request.reference_path is None
                    else str(result.request.reference_path)
                ),
                "split_group_key": result.request.split_group_key,
                "target_grid": {
                    "crs_wkt": result.request.target_grid.crs_wkt,
                    "transform": list(result.request.target_grid.transform),
                    "bounds": list(result.request.target_grid.bounds),
                    "width": result.request.target_grid.width,
                    "height": result.request.target_grid.height,
                },
                "geographic_aoi": {
                    "upper_left_latitude": (
                        result.request.geographic_aoi.upper_left_latitude
                    ),
                    "upper_left_longitude": (
                        result.request.geographic_aoi.upper_left_longitude
                    ),
                    "lower_right_latitude": (
                        result.request.geographic_aoi.lower_right_latitude
                    ),
                    "lower_right_longitude": (
                        result.request.geographic_aoi.lower_right_longitude
                    ),
                },
            },
            "chip": raster_report,
            "label": label_report,
            "label_source_path": str(source_label),
            "inspection_plot": None if plot_path is None else str(plot_path),
            "effective_selectors": [
                {
                    "acquisition_group": selector.acquisition_group,
                    "source_name": selector.source_name,
                    "product_id": selector.product_id,
                }
                for selector in result.effective_selectors
            ],
            "cube_records": [
                {
                    "path": str(record.path),
                    "source_name": record.source_name,
                    "product_id": record.product_id,
                    "zone": record.zone,
                    "zoom_level": record.zoom_level,
                    "tile_x": record.tile_x,
                    "tile_y": record.tile_y,
                    "band_names": list(record.band_names),
                }
                for record in result.cube_records
            ],
            "diagnostics": [
                {
                    "stage": item.stage,
                    "code": item.code,
                    "severity": item.severity,
                    "message": item.message,
                }
                for item in result.diagnostics
            ],
            "_chip_array": generated,
        }
    validation = {
        "sample_count": len(batch.results),
        "status_counts": dict(sorted(Counter(r.status for r in batch.results).items())),
        "worker_count": batch.worker_count,
        "elapsed_seconds": batch.elapsed_seconds,
        "manifest_path": str(batch.manifest_path),
        "manifest_sha256": _sha256(batch.manifest_path),
    }
    return samples, validation


def _public_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if not key.startswith("_")}


def main() -> None:
    args = _parse_args()

    import numpy as np

    from lfm.model import (
        chip_request_from_aoi,
        create_chips,
        create_chips_from_reference_directory,
        validate_dataset_publication,
    )

    args.reference_dir = _require_directory(args.reference_dir, "reference directory")
    args.label_source = _require_directory(args.label_source, "label source")
    args.wac_data_dir = _require_directory(args.wac_data_dir, "WAC data directory")
    args.wac_index = _require_file(
        args.wac_index or args.wac_data_dir / "output_index.shp",
        "WAC vector index",
    )
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive.")
    args.output_root = args.output_root.expanduser().resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Choose a clean --output-root; directory is not empty: {args.output_root}"
        )
    report_path = (
        args.output_root / "c8_1_wac_validation.json"
        if args.report_path is None
        else args.report_path.expanduser().resolve()
    )
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")

    reference_config = _build_config(args.output_root / "reference_api", args)
    reference_batch = create_chips_from_reference_directory(
        args.reference_dir,
        reference_config,
        split_group_key=lambda reference: reference.sample_id.split("_", 1)[0],
        recursive=args.recursive,
        max_workers=args.max_workers,
        progress=args.progress,
        progress_mode=args.progress_mode,
    )
    _require_success(reference_batch, "reference-directory API")
    validate_dataset_publication(
        reference_batch.prepared_requests,
        reference_batch.results,
        reference_batch.split_plan,
        reference_config,
        manifest_path=reference_batch.manifest_path,
    )

    explicit_requests = tuple(
        chip_request_from_aoi(
            sample_id=prepared.request.sample_id,
            crs_wkt=prepared.request.target_grid.crs_wkt,
            bounds=prepared.request.target_grid.bounds,
            width=prepared.request.target_grid.width,
            height=prepared.request.target_grid.height,
            transform=prepared.request.target_grid.transform,
            split_group_key=prepared.request.split_group_key,
        )
        for prepared in reference_batch.prepared_requests
    )
    explicit_config = _build_config(args.output_root / "explicit_aoi_api", args)
    explicit_batch = create_chips(
        explicit_requests,
        explicit_config,
        max_workers=args.max_workers,
        progress=args.progress,
        progress_mode=args.progress_mode,
    )
    _require_success(explicit_batch, "explicit-AOI API")
    validate_dataset_publication(
        explicit_batch.prepared_requests,
        explicit_batch.results,
        explicit_batch.split_plan,
        explicit_config,
        manifest_path=explicit_batch.manifest_path,
    )

    plot_dir = None if args.no_plots else args.output_root / "inspection_plots"
    reference_samples, reference_summary = _inspect_batch(
        reference_batch,
        plot_dir=plot_dir,
    )
    explicit_samples, explicit_summary = _inspect_batch(
        explicit_batch,
        plot_dir=None,
    )
    if set(reference_samples) != set(explicit_samples):
        raise AssertionError("The two API paths produced different sample membership.")
    comparisons = []
    for sample_id in sorted(reference_samples, key=str.casefold):
        reference_sample = reference_samples[sample_id]
        explicit_sample = explicit_samples[sample_id]
        arrays_equal = bool(
            np.array_equal(
                reference_sample["_chip_array"],
                explicit_sample["_chip_array"],
                equal_nan=True,
            )
        )
        labels_equal = (
            reference_sample["label"]["sha256"]
            == explicit_sample["label"]["sha256"]
        )
        if not arrays_equal or not labels_equal:
            raise AssertionError(
                f"Reference and explicit-AOI outputs differ for {sample_id}."
            )
        comparisons.append(
            {
                "sample_id": sample_id,
                "chip_arrays_equal": arrays_equal,
                "published_label_hashes_equal": labels_equal,
            }
        )

    report = {
        "validation_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "inputs": {
            "reference_dir": str(args.reference_dir),
            "label_source": str(args.label_source),
            "wac_data_dir": str(args.wac_data_dir),
            "wac_index": str(args.wac_index),
            "zoom_level": args.zoom_level,
            "sample_limit": args.sample_limit,
            "max_workers": args.max_workers,
        },
        "reference_directory_api": {
            **reference_summary,
            "samples": [
                _public_sample(reference_samples[sample_id])
                for sample_id in sorted(reference_samples, key=str.casefold)
            ],
        },
        "explicit_aoi_api": {
            **explicit_summary,
            "samples": [
                _public_sample(explicit_samples[sample_id])
                for sample_id in sorted(explicit_samples, key=str.casefold)
            ],
        },
        "cross_api_comparison": comparisons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"\nC8.1 WAC chip validation passed. Report: {report_path}")


if __name__ == "__main__":
    main()
