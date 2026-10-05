#!/usr/bin/env python
"""Focused A5 real-data check: two AOIs, one finished GeoPackage, WAC + static.

Use the original pre-tiling WAC VIS TIFF for grid metadata, not a reference
chip. Supply --aoi NORTH WEST SOUTH EAST twice, or --auto-aoi. At least one AOI must intersect
an annotated crater; choose an edge crossing a crater to exercise clipping.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-gpkg", type=Path, required=True)
    parser.add_argument("--layer", default="craters")
    parser.add_argument("--source-raster", type=Path, required=True, help="Original WAC VIS source TIFF.")
    parser.add_argument("--product-id", required=True)
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument("--auto-aoi", action="store_true",
                        help="Derive full-crater and edge-clipping AOIs from the largest annotation.")
    region.add_argument("--aoi", type=float, nargs=4, action="append",
                        metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wac-data-dir", type=Path, default=Path(
        "/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites"))
    parser.add_argument("--static-data-dir", type=Path, default=Path("/explore/nobackup/projects/lfm/staticLinks"))
    parser.add_argument("--max-chip-pixels", type=int, default=512 * 512)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def smoke_aoi_bounds(envelope):
    """Two small geographic envelopes; deliberately exclude dateline/polar cases."""
    import math
    west, east, south, north = envelope
    require(all(math.isfinite(v) for v in envelope), "Nonfinite annotation bounds.")
    width, height = east - west, north - south
    require(0 < width < 180 and height > 0, "Auto AOIs need a non-wrapping polygon.")
    full = [north + .2 * height, west - .2 * width,
            south - .2 * height, east + .2 * width]
    require(-82 < full[2] < full[0] < 82 and -180 < full[1] < full[3] < 180,
            "Auto AOIs support nonpolar, non-antimeridian examples only; supply explicit AOIs.")
    clipped = [full[0], full[1], full[2], (west + east) / 2]
    return [full, clipped]


def derive_smoke_aois(path, layer_name):
    """Read-only selection; IDs remain data, never hard-coded fixture contracts."""
    from osgeo import ogr, osr
    from model.lunar_crs import load_lunar_geographic_wkt

    ds = ogr.Open(str(path), 0)
    require(ds is not None, "Could not open label GeoPackage.")
    layer = ds.GetLayerByName(layer_name)
    require(layer is not None, f"Missing label layer: {layer_name}")
    source = layer.GetSpatialRef()
    require(source is not None, "Label layer has no CRS.")
    source = source.Clone()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geographic = osr.SpatialReference()
    geographic.ImportFromWkt(load_lunar_geographic_wkt())
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, geographic)
    candidates = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        require(geometry is not None and not geometry.IsEmpty() and geometry.IsValid(),
                "Smoke-test annotations must have valid, nonempty geometries.")
        candidates.append((geometry.GetArea(), int(feature.GetField("crater_id")), geometry.Clone()))
    require(bool(candidates), "Auto AOIs require at least one crater.")
    _, instance, geometry = max(candidates, key=lambda item: (item[0], -item[1]))
    # Densify before the nonlinear CRS transform; never mutate the source geometry.
    left, right, bottom, top = geometry.GetEnvelope()
    require(right > left and top > bottom, "Selected crater has no positive extent.")
    geometry.Segmentize(max(right - left, top - bottom) / 256)
    require(geometry.Transform(transform) == 0, "Could not transform crater to IAU:30100.")
    aois = smoke_aoi_bounds(geometry.GetEnvelope())
    ds = None
    return aois, instance


def inspect(result, expected_band_names, plot_path):
    import numpy as np
    from osgeo import gdal
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from model.chip_labels import _crs_is_same

    target = result.request.target_grid
    ds = gdal.Open(str(result.chip_path))
    require((ds.RasterXSize, ds.RasterYSize) == (target.width, target.height), "Wrong chip shape.")
    require(_crs_is_same(ds.GetProjection(), target.crs_wkt), "Wrong chip CRS.")
    np.testing.assert_allclose(ds.GetGeoTransform(), target.transform, rtol=0, atol=1e-9)
    names = tuple(ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1))
    require(names == expected_band_names, f"Wrong VIS/UV/static band order: {names}")
    pixels = ds.ReadAsArray()
    invalid_counts, invalid_union = [], np.zeros((target.height, target.width), dtype=bool)
    for i, array in enumerate(pixels, 1):
        band = ds.GetRasterBand(i)
        invalid = (band.GetMaskBand().ReadAsArray() == 0) | ~np.isfinite(array)
        if band.GetNoDataValue() is not None:
            invalid |= array == band.GetNoDataValue()
        invalid_counts.append(int(invalid.sum()))
        invalid_union |= invalid
    summary = result.imagery_nodata
    require(invalid_counts == [b["invalid_count"] for b in summary["bands"]], "Incorrect NoData counts.")
    require(int(invalid_union.sum()) == summary["union_invalid_count"], "Incorrect NoData union.")
    image = pixels[0].astype(float)
    band = ds.GetRasterBand(1)
    image[band.GetMaskBand().ReadAsArray() == 0] = np.nan
    band = ds = None
    with np.load(result.label_path, allow_pickle=False) as archive:
        mask, boxes, count = archive["mask"], archive["bboxes"], int(archive["num_craters"])
    require(mask.shape == (target.height, target.width), "Label and chip shapes differ.")
    require(boxes.shape == (count, 4), "Wrong box count.")
    mapping = result.prepared_label.instance_id_map
    require(tuple(new for _, new in mapping) == tuple(range(1, count + 1)), "Noncompact output IDs.")
    require(set(np.unique(mask)).issubset(set(range(count + 1))), "Unknown mask IDs.")
    if count:
        require(bool(np.all(boxes[:, :2] >= 0) and np.all(boxes[:, 2:] > 0)), "Invalid boxes.")
        require(bool(np.all(boxes[:, 0] + boxes[:, 2] <= target.width)
                     and np.all(boxes[:, 1] + boxes[:, 3] <= target.height)), "Boxes extend beyond chip.")
    instances = np.ma.masked_where(mask == 0, (mask - 1) % 20)
    vmin, vmax = np.nanpercentile(image, (2, 98))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for ax in (axes[0, 0], axes[0, 2]):
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].imshow(instances, cmap="tab20", vmin=-.5, vmax=19.5)
    axes[0, 2].imshow(instances, cmap="tab20", vmin=-.5, vmax=19.5, alpha=.45)
    for x, y, width, height in boxes:
        axes[0, 2].add_patch(Rectangle((x - .5, y - .5), width, height, fill=False, edgecolor="yellow", lw=.5))
    for ax, title in zip(axes.flat, ("WAC VIS", f"instances: {count}", "overlay + clipped boxes")):
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(result.request.sample_id)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return pixels, {"chip_sha256": sha256(result.chip_path), "label_sha256": sha256(result.label_path),
                    "instance_count": count, "id_mapping": mapping, "target_grid": target.to_dict(),
                    "imagery_nodata": summary, "plot": str(plot_path)}


def run(args, report):
    from osgeo import gdal
    import numpy as np
    from model import (AcquisitionGroupConfig, ChipConfig, GeographicAOI, LabelInput,
                       NoSplitConfig, OutputModalityConfig, SourceSelector, TargetGrid,
                       TileConfig, TileSourceConfig, TileSourcePreparation, WAC_BAND_NAMES,
                       STATIC_BAND_NAMES, chip_request_from_aoi, create_chips,
                       ensure_vector_index, resolve_notebook_source_index)
    from model.chip_requests import raster_bounds
    from lfm.all_models.all_tasks.tiling_utils import make_static_source

    gdal.UseExceptions()
    label_before = sha256(args.label_gpkg)
    selected_instance = None
    if args.auto_aoi:
        args.aoi, selected_instance = derive_smoke_aois(args.label_gpkg, args.layer)
        report["auto_aoi"] = {"source_instance_id": selected_instance,
                              "cases": ["full_crater", "edge_clipping"], "aois_nwse": args.aoi}
        print(f"Auto AOIs around source crater {selected_instance}: {args.aoi}", flush=True)
    source_stat = args.source_raster.stat()
    ds = gdal.Open(str(args.source_raster))
    require(ds.RasterCount == 5, "Use the original five-band WAC VIS source raster for the grid.")
    affine = ds.GetGeoTransform()
    source = TargetGrid(ds.GetProjection(), affine, raster_bounds(affine, ds.RasterXSize, ds.RasterYSize),
                        ds.RasterXSize, ds.RasterYSize)
    ds = None
    requests = tuple(chip_request_from_aoi(
        sample_id=f"aoi_{i}", geographic_aoi=GeographicAOI(*aoi), source_grid=source,
        split_group_key=args.product_id,
        source_selectors=(SourceSelector("wac_grid", "wac", args.product_id),),
        label_input=LabelInput(args.label_gpkg, relation="clip_to_target", layer=args.layer))
        for i, aoi in enumerate(args.aoi, 1))
    require(all(r.target_grid.width * r.target_grid.height <= args.max_chip_pixels for r in requests),
            "AOI exceeds the smoke-test pixel limit; choose smaller AOIs.")
    require(requests[0].target_grid != requests[1].target_grid, "Choose two distinct AOIs.")
    # A known missing input tests failure isolation without altering the scientist's GeoPackage.
    missing = args.output_root / "deliberately_missing.gpkg"
    negative = replace(requests[0], sample_id="00_invalid_label",
                       label_path=missing, label_input=LabelInput(missing, relation="clip_to_target"))
    requests = (negative, *requests)
    sources, indexes = [], {}
    for name, data_dir in (("wac", args.wac_data_dir), ("static", args.static_data_dir)):
        resolution = resolve_notebook_source_index(source_name=name, data_dir=data_dir,
                                                   cache_dir=args.output_root / "indexes")
        source_config = (TileSourceConfig(name, data_dir, resolution.index_path, selection_mode="product_id",
                                         preserve_source_nodata=True) if name == "wac" else
                         make_static_source(data_dir=data_dir, index_path=resolution.index_path))
        before = sha256(resolution.index_path) if resolution.index_path.exists() else None
        ensure_vector_index(TileSourcePreparation(source_config,
            rebuild_invalid_index=resolution.rebuild_invalid_index, worker_count=1).index_config(), stdout=sys.stdout)
        after = sha256(resolution.index_path)
        if resolution.uses_shared_default:
            require(after == before, "Shared index was modified during preparation.")
        indexes[str(resolution.index_path)] = after
        sources.append(source_config)
    report["inputs"] = {"label": str(args.label_gpkg), "label_sha256": label_before,
                        "source_raster": str(args.source_raster), "product_id": args.product_id,
                        "source_grid": source.to_dict(), "source_size": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                        "aois_nwse": args.aoi, "indexes": indexes}
    previous = {}
    total_instances = 0
    for mode, workers in (("serial", 1), ("parallel", 2)):
        output = args.output_root / mode
        config = ChipConfig(output, args.label_gpkg,
            acquisition_groups=(AcquisitionGroupConfig("wac_grid", TileConfig(output / ".unused", 5, tuple(sources))),),
            output_modalities=(OutputModalityConfig("wac_grid", "wac", "wac"),
                               OutputModalityConfig("wac_grid", "static", "static")),
            intermediate_root=output / ".intermediate", intermediate_retention="never", split_config=NoSplitConfig())
        batch = create_chips(requests, config, max_workers=workers, progress=True, progress_mode="log")
        report[mode] = {"elapsed_seconds": batch.elapsed_seconds, "workers": batch.worker_count,
                        "manifest": str(batch.manifest_path), "samples": []}
        manifest = json.loads(batch.manifest_path.read_text())
        require(manifest["manifest_version"] == 2, "Expected A5 provenance schema.")
        for result in batch.results:
            row = {"sample_id": result.request.sample_id, "status": result.status,
                   "message": result.message, "diagnostics": [d.__dict__ for d in result.diagnostics]}
            report[mode]["samples"].append(row)
            if result.request.sample_id == "00_invalid_label":
                require(result.status == "failed" and result.preflight.status == "failed"
                        and not result.cube_records, "Negative request did not fail before acquisition.")
                require(result.chip_path is None and result.label_path is None, "Negative request published a pair.")
                continue
            require(result.status == "success", f"{mode}/{result.request.sample_id}: {result.message}; {row['diagnostics']}")
            pixels, detail = inspect(result, (*WAC_BAND_NAMES, *STATIC_BAND_NAMES),
                                     args.output_root / "inspection_plots" / f"{mode}_{result.request.sample_id}.png")
            row.update(detail)
            row["label_diagnostics"] = [d.__dict__ for d in result.prepared_label.diagnostics]
            if selected_instance is not None:
                require(selected_instance in dict(result.prepared_label.instance_id_map),
                        "Selected crater disappeared from smoke-test labels.")
                clipped = any(d.code == "clipped_instance" and
                              d.message.startswith(f"Source instance {selected_instance}:")
                              for d in result.prepared_label.diagnostics)
                require(clipped == (result.request.sample_id == "aoi_2"),
                        "Auto AOIs did not exercise the expected full/crater-edge cases.")
            total_instances += detail["instance_count"]
            if mode == "serial":
                previous[result.request.sample_id] = (pixels, detail)
            else:
                old_pixels, old = previous[result.request.sample_id]
                np.testing.assert_array_equal(pixels, old_pixels)
                for key in ("label_sha256", "chip_sha256", "id_mapping", "target_grid", "imagery_nodata"):
                    require(detail[key] == old[key], f"Serial/parallel mismatch: {key}")
            require(not (config.intermediate_root / result.request.sample_id).exists(), "Cleanup left sample intermediates.")
    require(total_instances > 0, "Both AOIs were empty labels; choose at least one intersecting an annotated crater.")
    require(sha256(args.label_gpkg) == label_before, "Source GeoPackage changed.")
    require(all(sha256(path) == digest for path, digest in indexes.items()), "Source index changed in workers.")
    after_stat = args.source_raster.stat()
    require((after_stat.st_size, after_stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns),
            "Source raster size/modification time changed.")
    report["preservation_checks"] = {"label_hash_unchanged": True, "index_hashes_unchanged": True,
                                     "source_raster_size_and_mtime_unchanged": True}
    report["visual_review"] = "pending user inspection of saved overlays"
    report["status"] = "passed"


def main():
    args = arguments()
    require(args.auto_aoi or len(args.aoi) == 2, "Supply --aoi exactly twice.")
    require(args.max_chip_pixels > 0, "--max-chip-pixels must be positive.")
    require(args.label_gpkg.is_file() and args.source_raster.is_file(), "Source raster and GeoPackage must exist.")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "failed", "job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        path = args.output_root / "a5_validation.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"A5 report: {path}", flush=True)


if __name__ == "__main__":
    main()
