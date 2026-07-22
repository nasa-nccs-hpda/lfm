"""Preprocess NAC COCO crater annotations into Lunar instance-seg split folders.

The input NAC release is expected to contain:

```
NAC_craters_coco_release_final/
├── PHO/
├── DTM/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

Each split JSON is a COCO annotation file. This script reads NetCDF chips with
xarray, rasterizes COCO polygons into instance-id masks, and writes the paired
folder layout used by the current Lunar datamodules:

```
output_root/{train,val,test}/chips/*_input_nac_chip.npy
output_root/{train,val,test}/labels/*_label.npz
```
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from skimage.draw import polygon
from tqdm import tqdm


@dataclass(frozen=True)
class SplitSummary:
    split: str
    images: int
    annotations: int
    successful: int
    failed: int
    total_craters: int


def min_max_scale_bands(bands: np.ndarray) -> np.ndarray:
    """Min-max scale each band independently to [0, 1]."""
    bands = np.asarray(bands, dtype=np.float32)
    scaled = np.zeros_like(bands, dtype=np.float32)
    for index in range(bands.shape[0]):
        band = bands[index]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        band_min = float(np.nanmin(band))
        band_max = float(np.nanmax(band))
        if band_max > band_min:
            scaled[index] = (band - band_min) / (band_max - band_min)
        else:
            scaled[index] = band
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def read_netcdf_band(path: Path, *, variable: str = "band_data") -> np.ndarray:
    """Read a 2D NetCDF variable with xarray."""
    with xr.open_dataset(path) as dataset:
        if variable in dataset:
            arr = dataset[variable].values
        elif len(dataset.data_vars) == 1:
            arr = next(iter(dataset.data_vars.values())).values
        else:
            available = ", ".join(dataset.data_vars)
            raise KeyError(
                f"{path} does not contain variable {variable!r}. "
                f"Available variables: {available}"
            )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D NetCDF variable in {path}, got {arr.shape}")
    return arr


def dtm_path_for_pho_path(nac_root: Path, pho_filename: str) -> Path:
    """Resolve the matched DTM tile for a PHO COCO filename."""
    dtm_filename = pho_filename.replace("_PHO_", "_DTM_")
    if dtm_filename == pho_filename:
        dtm_filename = pho_filename.replace("_PHO", "_DTM")
    return nac_root / "DTM" / dtm_filename


def segmentation_to_polygons(segmentation: Any) -> list[np.ndarray]:
    """Convert COCO polygon segmentation to arrays of x/y pairs."""
    if not segmentation:
        return []
    polygons: list[np.ndarray] = []
    for raw_polygon in segmentation:
        if not raw_polygon:
            continue
        coords = np.asarray(raw_polygon, dtype=np.float32)
        if coords.ndim != 1 or coords.size < 6 or coords.size % 2:
            continue
        polygons.append(coords.reshape(-1, 2))
    return polygons


def create_mask_from_segmentation(
    segmentation: Any,
    *,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a COCO polygon segmentation to a binary mask."""
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for points in segmentation_to_polygons(segmentation):
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        rr, cc = polygon(y_coords, x_coords, shape=mask_shape)
        mask[rr, cc] = 1
    return mask


def load_split_coco(split_json: Path) -> dict[str, Any]:
    with split_json.open("r") as file:
        data = json.load(file)
    required = {"images", "annotations", "categories"}
    missing = required - data.keys()
    if missing:
        raise KeyError(f"{split_json} is missing COCO keys: {sorted(missing)}")
    return data


def annotations_by_image_id(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def process_one_image(args: tuple[Any, ...]) -> tuple[bool, str, int, str | None]:
    (
        image,
        annotations,
        nac_root,
        split_output_root,
        include_dtm,
        nc_variable,
        scale_mode,
        chip_suffix,
        label_suffix,
    ) = args

    try:
        nac_root = Path(nac_root)
        split_output_root = Path(split_output_root)
        file_name = str(image["file_name"])
        base_name = Path(file_name).stem
        height = int(image.get("height", 256))
        width = int(image.get("width", 256))

        pho_path = nac_root / "PHO" / file_name
        if not pho_path.exists():
            raise FileNotFoundError(f"Missing PHO tile: {pho_path}")

        bands = [read_netcdf_band(pho_path, variable=nc_variable)]
        if include_dtm:
            dtm_path = dtm_path_for_pho_path(nac_root, file_name)
            if not dtm_path.exists():
                raise FileNotFoundError(f"Missing DTM tile: {dtm_path}")
            bands.append(read_netcdf_band(dtm_path, variable=nc_variable))

        chip = np.stack(bands, axis=0).astype(np.float32)
        if chip.shape[-2:] != (height, width):
            raise ValueError(
                f"{pho_path} has shape {chip.shape[-2:]}, "
                f"but COCO image says {(height, width)}"
            )
        if scale_mode == "per-chip-minmax":
            chip = min_max_scale_bands(chip)
        elif scale_mode == "none":
            chip = np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unsupported scale mode: {scale_mode}")

        instance_mask = np.zeros((height, width), dtype=np.uint16)
        bboxes: list[list[float]] = []
        kept_instances = 0

        for annotation in annotations:
            segmentation = annotation.get("segmentation", [])
            single_mask = create_mask_from_segmentation(
                segmentation,
                mask_shape=(height, width),
            )
            if not single_mask.any():
                continue
            kept_instances += 1
            instance_mask[single_mask > 0] = kept_instances
            bbox = annotation.get("bbox")
            if bbox is not None:
                bboxes.append([float(value) for value in bbox[:4]])

        chip_path = split_output_root / "chips" / f"{base_name}{chip_suffix}.npy"
        label_path = split_output_root / "labels" / f"{base_name}{label_suffix}.npz"
        np.save(chip_path, chip)
        np.savez_compressed(
            label_path,
            mask=instance_mask,
            bboxes=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
            num_craters=np.asarray(kept_instances, dtype=np.int64),
        )

        return True, base_name, kept_instances, None
    except Exception as exc:  # noqa: BLE001 - return worker errors to parent
        return False, str(image.get("file_name", "unknown")), 0, repr(exc)


def prepare_output_split(split_output_root: Path, *, overwrite: bool) -> None:
    if split_output_root.exists() and overwrite:
        shutil.rmtree(split_output_root)
    (split_output_root / "chips").mkdir(parents=True, exist_ok=True)
    (split_output_root / "labels").mkdir(parents=True, exist_ok=True)


def process_split(
    *,
    nac_root: Path,
    output_root: Path,
    split: str,
    include_dtm: bool,
    nc_variable: str,
    scale_mode: str,
    chip_suffix: str,
    label_suffix: str,
    max_workers: int,
    limit: int | None,
    overwrite: bool,
) -> SplitSummary:
    split_json = nac_root / "splits" / f"{split}.json"
    coco = load_split_coco(split_json)
    grouped_annotations = annotations_by_image_id(coco)
    images = coco["images"] if limit is None else coco["images"][:limit]
    split_output_root = output_root / split
    prepare_output_split(split_output_root, overwrite=overwrite)

    args_list = [
        (
            image,
            grouped_annotations.get(int(image["id"]), []),
            nac_root,
            split_output_root,
            include_dtm,
            nc_variable,
            scale_mode,
            chip_suffix,
            label_suffix,
        )
        for image in images
    ]

    successful = 0
    failed = 0
    total_craters = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_image, args) for args in args_list]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{split}",
            unit="tile",
        ):
            success, base_name, num_craters, error = future.result()
            if success:
                successful += 1
                total_craters += num_craters
            else:
                failed += 1
                errors.append((base_name, error or "unknown error"))

    if errors:
        print(f"[{split}] First {min(5, len(errors))} error(s):")
        for base_name, error in errors[:5]:
            print(f"  {base_name}: {error}")

    return SplitSummary(
        split=split,
        images=len(images),
        annotations=sum(len(grouped_annotations.get(int(i["id"]), [])) for i in images),
        successful=successful,
        failed=failed,
        total_craters=total_craters,
    )


def write_summary(output_root: Path, summaries: list[SplitSummary], args) -> None:
    summary = {
        "nac_root": str(args.nac_root),
        "output_root": str(output_root),
        "include_dtm": bool(args.include_dtm),
        "nc_variable": args.nc_variable,
        "scale_mode": args.scale_mode,
        "chip_suffix": args.chip_suffix,
        "label_suffix": args.label_suffix,
        "splits": [summary.__dict__ for summary in summaries],
    }
    with (output_root / "preprocess_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    lines = [
        "NAC COCO instance preprocessing summary",
        f"nac_root: {args.nac_root}",
        f"output_root: {output_root}",
        f"include_dtm: {args.include_dtm}",
        f"scale_mode: {args.scale_mode}",
        "",
    ]
    for item in summaries:
        avg = item.total_craters / item.successful if item.successful else math.nan
        lines.extend(
            [
                f"[{item.split}]",
                f"images: {item.images}",
                f"annotations: {item.annotations}",
                f"successful: {item.successful}",
                f"failed: {item.failed}",
                f"total_craters: {item.total_craters}",
                f"avg_craters_per_successful_image: {avg:.2f}",
                "",
            ]
        )
    (output_root / "preprocess_summary.txt").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nac-root",
        type=Path,
        default=Path(
            "/explore/nobackup/projects/lfm/processed_data/Lunar/data_release/"
            "NAC_craters_coco_release_final"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--include-dtm",
        action="store_true",
        help="Stack matched DTM as a second channel after PHO.",
    )
    parser.add_argument(
        "--nc-variable",
        default="band_data",
        help="NetCDF variable to read with xarray.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["per-chip-minmax", "none"],
        default="per-chip-minmax",
    )
    parser.add_argument("--chip-suffix", default="_input_nac_chip")
    parser.add_argument("--label-suffix", default="_label")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in args.splits:
        print(f"\n=== Processing {split} split ===", flush=True)
        summary = process_split(
            nac_root=args.nac_root.resolve(),
            output_root=output_root,
            split=split,
            include_dtm=args.include_dtm,
            nc_variable=args.nc_variable,
            scale_mode=args.scale_mode,
            chip_suffix=args.chip_suffix,
            label_suffix=args.label_suffix,
            max_workers=args.max_workers,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(
            f"[{split}] successful={summary.successful} failed={summary.failed} "
            f"craters={summary.total_craters}",
            flush=True,
        )

    write_summary(output_root, summaries, args)
    elapsed = time.perf_counter() - started
    print(f"\nSaved preprocessed NAC dataset to: {output_root}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print("\nTraining args for this output:")
    print("--image-glob '*.npy' --image-suffix '_input_nac_chip' --label-suffix '_label'")


if __name__ == "__main__":
    main()
