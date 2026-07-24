"""Shared image and label IO helpers for split lunar datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    image_path: Path
    label_path: Path


def path_key(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def find_pair_records(
    chips_dir: str | Path,
    labels_dir: str | Path,
    *,
    image_glob: str = "*.tif",
    label_glob: str = "*_label.*",
    image_suffix: str = "_input_wac_static_chip",
    label_suffix: str = "_label",
    require_all_labels: bool = False,
) -> list[PairRecord]:
    chips_dir = Path(chips_dir)
    labels_dir = Path(labels_dir)
    if not chips_dir.exists():
        raise FileNotFoundError(f"chips_dir does not exist: {chips_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")

    labels_by_key = {
        path_key(path, label_suffix): path
        for path in sorted(labels_dir.glob(label_glob))
    }
    records: list[PairRecord] = []
    missing_labels: list[Path] = []
    for image_path in sorted(chips_dir.glob(image_glob)):
        label_path = labels_by_key.get(path_key(image_path, image_suffix))
        if label_path is None:
            missing_labels.append(image_path)
            continue
        records.append(PairRecord(image_path=image_path, label_path=label_path))

    if require_all_labels and missing_labels:
        examples = "\n".join(str(path) for path in missing_labels[:5])
        raise FileNotFoundError(
            f"{len(missing_labels)} chip files had no matching label. "
            f"First missing examples:\n{examples}"
        )
    if not records:
        raise FileNotFoundError(
            f"No matched pairs found in {chips_dir} and {labels_dir}"
        )
    return records


def _nodata_mask_from_array(
    arr: np.ndarray,
    nodata: float | int | None = None,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr_chw = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] <= 32:
            arr_chw = arr
        elif arr.shape[-1] <= 32:
            arr_chw = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Expected CHW or HWC image array, got {arr.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D image array, got {arr.shape}")

    invalid = ~np.isfinite(arr_chw)
    if nodata is not None:
        invalid = invalid | (arr_chw == nodata)
    return invalid.any(axis=0)


def read_tif_with_nodata_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read()
            nodata = src.nodata
        nodata_mask = _nodata_mask_from_array(arr, nodata)
        if arr.shape[0] == 1:
            arr = arr[0]
        return arr, nodata_mask
    except ImportError:
        import tifffile

        arr = tifffile.imread(path)
        return arr, _nodata_mask_from_array(arr)


def read_tif(path: Path) -> np.ndarray:
    arr, _ = read_tif_with_nodata_mask(path)
    return arr


def read_netcdf(path: Path, *, variable: str = "band_data") -> np.ndarray:
    import xarray as xr

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
    return np.asarray(arr)


def read_image_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if "image" in data:
                return data["image"]
            if "data" in data:
                return data["data"]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise KeyError(
                f"{path} is an image .npz but does not contain 'image' or 'data'. "
                f"Available keys: {data.files}"
            )
    if suffix == ".nc":
        return read_netcdf(path)
    return read_tif(path)


def read_image_file_with_nodata_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_tif_with_nodata_mask(path)
    arr = read_image_file(path)
    return arr, _nodata_mask_from_array(arr)


def read_label_file(
    path: str | Path,
    *,
    npz_key: str = "mask",
) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if npz_key not in data:
                raise KeyError(
                    f"{path} is missing required {npz_key!r} array. "
                    f"Available keys: {data.files}"
                )
            return data[npz_key]
    return read_tif(path)


def image_to_hwc_float(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[0] <= arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got {arr.shape}")
    return arr.astype(np.float32, copy=False)
