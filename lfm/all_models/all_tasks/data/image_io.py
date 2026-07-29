"""Shared image and label IO helpers for split lunar datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    image_path: Path
    label_path: Path


IMAGE_ROLE_TOKENS = {"chip", "chips"}
LABEL_ROLE_TOKENS = {"label", "labels"}
IMAGE_DESCRIPTOR_TOKENS = {"input", "wac", "nac", "static"}


def path_key(path: Path, suffix: str | None) -> str:
    stem = path.stem
    if suffix and stem.lower().endswith(suffix.lower()):
        return stem[: -len(suffix)].lower()
    return stem.lower()


def inferred_path_key(path: Path, *, role_tokens: set[str]) -> str:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", path.stem) if token]
    lowered_tokens = [token.lower() for token in tokens]
    try:
        marker_index = next(
            index for index, token in enumerate(lowered_tokens) if token in role_tokens
        )
    except StopIteration:
        return path.stem.lower()

    key_tokens = lowered_tokens[:marker_index]
    if role_tokens == IMAGE_ROLE_TOKENS:
        while key_tokens and key_tokens[-1] in IMAGE_DESCRIPTOR_TOKENS:
            key_tokens.pop()
    return "\0".join(key_tokens)


def find_pair_records(
    chips_dir: str | Path,
    labels_dir: str | Path,
    *,
    image_glob: str = "*chip*.tif",
    label_glob: str = "*label.*",
    image_suffix: str | None = None,
    label_suffix: str | None = None,
    require_all_labels: bool = False,
) -> list[PairRecord]:
    chips_dir = Path(chips_dir)
    labels_dir = Path(labels_dir)
    if not chips_dir.exists():
        raise FileNotFoundError(f"chips_dir does not exist: {chips_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")

    labels_by_key: dict[str, Path] = {}
    duplicate_label_keys: dict[str, list[Path]] = {}
    for label_path in sorted(labels_dir.glob(label_glob)):
        key = (
            path_key(label_path, label_suffix)
            if label_suffix
            else inferred_path_key(label_path, role_tokens=LABEL_ROLE_TOKENS)
        )
        if key in labels_by_key:
            duplicate_label_keys.setdefault(key, [labels_by_key[key]]).append(
                label_path
            )
            continue
        labels_by_key[key] = label_path
    if duplicate_label_keys:
        examples = "\n".join(
            f"{key!r}: {paths[0]} and {paths[1]}"
            for key, paths in list(duplicate_label_keys.items())[:5]
        )
        raise ValueError(
            "Multiple label files map to the same inferred key. "
            "Pass --label-suffix or a narrower --label-glob to disambiguate. "
            f"First duplicate examples:\n{examples}"
        )

    records: list[PairRecord] = []
    missing_labels: list[Path] = []
    for image_path in sorted(chips_dir.glob(image_glob)):
        image_key = (
            path_key(image_path, image_suffix)
            if image_suffix
            else inferred_path_key(image_path, role_tokens=IMAGE_ROLE_TOKENS)
        )
        label_path = labels_by_key.get(image_key)
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


def read_label_file_with_metadata(
    path: str | Path,
) -> np.ndarray | dict[str, np.ndarray | None]:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "mask" not in data:
                raise KeyError(f"{path} is missing required 'mask' array")
            return {
                "mask": data["mask"],
                "bboxes": data["bboxes"] if "bboxes" in data else None,
                "num_craters": data["num_craters"] if "num_craters" in data else None,
            }
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
