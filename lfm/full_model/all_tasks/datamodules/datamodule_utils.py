"""Shared dataset utilities for Lunar fine-tuning notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


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
        key = path_key(image_path, image_suffix)
        label_path = labels_by_key.get(key)
        if label_path is None:
            missing_labels.append(image_path)
            continue
        records.append(PairRecord(image_path=image_path, label_path=label_path))

    if missing_labels:
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


def read_tif(path: Path) -> np.ndarray:
    """Read a TIFF with rasterio when available, otherwise tifffile."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read()
        if arr.shape[0] == 1:
            return arr[0]
        return arr
    except ImportError:
        import tifffile

        return tifffile.imread(path)


def read_netcdf(path: Path, *, variable: str = "band_data") -> np.ndarray:
    """Read a NetCDF image variable with xarray."""
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


def read_image_file(path: Path) -> np.ndarray:
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


def read_label_file(path: Path) -> np.ndarray | dict[str, np.ndarray | None]:
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


def image_to_chw_float(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # rasterio returns CHW; tifffile commonly returns HWC.
        if (
            arr.shape[0] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            and arr.shape[-1] <= 32
        ):
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.float32)


def mask_to_hw_long(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label array, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.long)


def shift_mask(mask: torch.Tensor, shift_xy: tuple[int, int] | None) -> torch.Tensor:
    """Translate a HW mask by integer pixels, filling exposed pixels with 0.

    ``shift_xy`` is ``(x_shift, y_shift)`` in image coordinates. Positive
    ``x_shift`` moves labels right; positive ``y_shift`` moves labels down.
    """
    if shift_xy is None:
        return mask
    shift_x, shift_y = int(shift_xy[0]), int(shift_xy[1])
    if shift_x == 0 and shift_y == 0:
        return mask
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for shift, got shape {tuple(mask.shape)}")

    height, width = mask.shape
    shifted = torch.zeros_like(mask)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def boxes_to_tensor(arr: np.ndarray | None) -> torch.Tensor:
    if arr is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = np.asarray(arr)
    if arr.size == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 4:
        raise ValueError(
            f"Expected bbox array with at least 4 columns, got {arr.shape}"
        )
    return torch.as_tensor(arr[:, :4], dtype=torch.float32)


def crop_boxes_xywh_to_xyxy(
    boxes: torch.Tensor,
    *,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
    min_size: float = 1.0,
) -> torch.Tensor:
    """Crop COCO ``xywh`` boxes and return clipped ``xyxy`` boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)

    boxes = boxes.to(dtype=torch.float32)
    x1 = boxes[:, 0] - float(left)
    y1 = boxes[:, 1] - float(top)
    x2 = boxes[:, 0] + boxes[:, 2] - float(left)
    y2 = boxes[:, 1] + boxes[:, 3] - float(top)

    cropped = torch.stack(
        [
            x1.clamp(0, crop_w),
            y1.clamp(0, crop_h),
            x2.clamp(0, crop_w),
            y2.clamp(0, crop_h),
        ],
        dim=1,
    )
    keep = (cropped[:, 2] - cropped[:, 0] >= float(min_size)) & (
        cropped[:, 3] - cropped[:, 1] >= float(min_size)
    )
    return cropped[keep]


def crop_boxes_xyxy(
    boxes: torch.Tensor,
    *,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
    min_size: float = 1.0,
) -> torch.Tensor:
    """Crop ``xyxy`` boxes and return clipped ``xyxy`` boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)

    cropped = boxes.clone().to(dtype=torch.float32)
    cropped[:, [0, 2]] -= float(left)
    cropped[:, [1, 3]] -= float(top)
    cropped[:, [0, 2]] = cropped[:, [0, 2]].clamp(0, crop_w)
    cropped[:, [1, 3]] = cropped[:, [1, 3]].clamp(0, crop_h)
    keep = (cropped[:, 2] - cropped[:, 0] >= float(min_size)) & (
        cropped[:, 3] - cropped[:, 1] >= float(min_size)
    )
    return cropped[keep]


def instance_mask_to_object_detection_targets(
    mask: torch.Tensor,
    *,
    box_format: str = "xyxy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an instance-id mask into boxes, labels, and binary masks.

    The input mask uses 0 as background and positive integer values as instance
    ids. The output labels are all class 1 because the crater instance dataset
    is single-class.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, got shape {tuple(mask.shape)}")
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError(f"Unsupported box_format: {box_format}")

    height, width = mask.shape[-2], mask.shape[-1]
    instance_ids = torch.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]

    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for instance_id in instance_ids.tolist():
        instance_mask = mask == int(instance_id)
        ys, xs = torch.where(instance_mask)
        if xs.numel() == 0 or ys.numel() == 0:
            continue

        x1 = xs.min().to(torch.float32)
        y1 = ys.min().to(torch.float32)
        x2 = xs.max().to(torch.float32) + 1.0
        y2 = ys.max().to(torch.float32) + 1.0
        if x2 <= x1 or y2 <= y1:
            continue

        if box_format == "xyxy":
            box = torch.stack([x1, y1, x2, y2])
        else:
            cx = ((x1 + x2) * 0.5) / float(width)
            cy = ((y1 + y2) * 0.5) / float(height)
            box_w = (x2 - x1) / float(width)
            box_h = (y2 - y1) / float(height)
            box = torch.stack([cx, cy, box_w, box_h])

        boxes.append(box)
        labels.append(torch.tensor(1, dtype=torch.long))
        masks.append(instance_mask.to(torch.uint8))

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, height, width), dtype=torch.uint8),
        )

    return (
        torch.stack(boxes).to(torch.float32),
        torch.stack(labels).to(torch.long),
        torch.stack(masks).to(torch.uint8),
    )


def center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int | tuple[int, int],
    *,
    boxes: torch.Tensor | None = None,
    boxes_format: str = "xywh",
    output_boxes_format: str = "xyxy",
    min_box_size: float = 1.0,
    sample_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    old_h, old_w = mask.shape[-2], mask.shape[-1]
    crop_h, crop_w = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if crop_h > old_h or crop_w > old_w:
        raise ValueError(
            f"crop_size {(crop_h, crop_w)} exceeds sample size {(old_h, old_w)} "
            f"for {sample_name}"
        )
    top = (old_h - crop_h) // 2
    left = (old_w - crop_w) // 2
    image = image[:, top : top + crop_h, left : left + crop_w]
    mask = mask[top : top + crop_h, left : left + crop_w]

    if boxes is not None:
        if output_boxes_format != "xyxy":
            raise ValueError(f"Unsupported output_boxes_format: {output_boxes_format}")
        if boxes_format == "xywh":
            boxes = crop_boxes_xywh_to_xyxy(
                boxes,
                left=left,
                top=top,
                crop_w=crop_w,
                crop_h=crop_h,
                min_size=min_box_size,
            )
        elif boxes_format == "xyxy":
            boxes = crop_boxes_xyxy(
                boxes,
                left=left,
                top=top,
                crop_w=crop_w,
                crop_h=crop_h,
                min_size=min_box_size,
            )
        else:
            raise ValueError(f"Unsupported boxes_format: {boxes_format}")

    return image, mask, boxes


def normalize_image(
    image: torch.Tensor,
    means: list[float] | None,
    stds: list[float] | None,
) -> torch.Tensor:
    """Apply per-band z-score normalization to a CHW image tensor.

    ``means`` and ``stds`` should match the post-filtered image channel order.
    They may come from fine-tuning train-split statistics or TerraMind
    pretraining modality metadata.
    """
    if means is None or stds is None:
        return image
    if len(means) != len(stds):
        raise ValueError(
            f"means and stds must have the same length: {len(means)} != {len(stds)}"
        )
    mean = torch.tensor(means, dtype=image.dtype).view(-1, 1, 1)
    std = torch.tensor(stds, dtype=image.dtype).view(-1, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("All normalization stds must be positive.")
    if mean.shape[0] == 1 and image.shape[0] != 1:
        warnings.warn(
            "Expanding single-channel norm stats to "
            f"{image.shape[0]} image channels. "
            "This is only appropriate when all channels share the same "
            "physical modality/range.",
            UserWarning,
            stacklevel=2,
        )
        mean = mean.expand(image.shape[0], -1, -1)
        std = std.expand(image.shape[0], -1, -1)
    elif mean.shape[0] != image.shape[0]:
        raise ValueError(
            f"Normalization stats have {mean.shape[0]} channel(s), "
            f"but image has {image.shape[0]} channel(s)."
        )
    return (image - mean) / std


def collate_semantic_segmentation(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "filename": [item["filename"] for item in batch],
    }


def collate_instance_segmentation(batch: list[dict]) -> dict:
    result = collate_semantic_segmentation(batch)
    if "crater_boxes" in batch[0]:
        result["crater_boxes"] = [item["crater_boxes"] for item in batch]
    if "num_craters" in batch[0]:
        result["num_craters"] = torch.stack([item["num_craters"] for item in batch])
    return result


def collate_object_detection_instance_segmentation(batch: list[dict]) -> dict:
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "masks": [item["masks"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }
    if "mask" in batch[0]:
        result["mask"] = torch.stack([item["mask"] for item in batch])
    if "crater_boxes" in batch[0]:
        result["crater_boxes"] = [item["crater_boxes"] for item in batch]
    if "num_craters" in batch[0]:
        result["num_craters"] = torch.stack([item["num_craters"] for item in batch])
    return result
