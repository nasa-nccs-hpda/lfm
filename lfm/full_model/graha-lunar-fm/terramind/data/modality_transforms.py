# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import math
import os
import random
import re

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from scipy import ndimage

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from einops import rearrange, reduce, repeat

from terramind.data.io_utils import load_netcdf

DEFAULT_BINNING_FILE = "terramind/utils/tokenizer/trained/metadata_binning_config.csv"


def get_transform(mod_name, transforms_dict):
    return transforms_dict.get(mod_name, IdentityTransform())


def get_resample_mode(resample_mode: str) -> TF.InterpolationMode:
    """Returns the torchvision resampling mode for the given resample mode string.

    Args:
        resample_mode: Resampling mode string
    """
    if resample_mode == "bilinear":
        return TF.InterpolationMode.BILINEAR
    elif resample_mode == "bicubic":
        return TF.InterpolationMode.BICUBIC
    elif resample_mode == "nearest":
        return TF.InterpolationMode.NEAREST
    else:
        raise ValueError(f"Resample mode {resample_mode} is not supported.")


def _check_tensor(tensor: torch.Tensor):
    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )


def minmax_op(
    tensor: torch.Tensor, min_value: Sequence[float], max_value: Sequence[float]
):

    _check_tensor(tensor)

    min_tensor = torch.as_tensor(min_value, dtype=tensor.dtype, device=tensor.device)
    max_tensor = torch.as_tensor(max_value, dtype=tensor.dtype, device=tensor.device)

    if min_tensor.ndim == 1:
        min_tensor = min_tensor.view(-1, 1, 1)
    if max_tensor.ndim == 1:
        max_tensor = max_tensor.view(-1, 1, 1)

    tensor_minmax = (tensor - min_tensor) / (max_tensor - min_tensor)
    tensor_minmax = torch.nan_to_num(tensor_minmax, nan=0.0)

    return tensor_minmax


def minmax_op_reverse(
    tensor: torch.Tensor, min_value: Sequence[float], max_value: Sequence[float]
):

    _check_tensor(tensor)

    min_tensor = torch.as_tensor(min_value, dtype=tensor.dtype, device=tensor.device)
    max_tensor = torch.as_tensor(max_value, dtype=tensor.dtype, device=tensor.device)

    if min_tensor.ndim == 1:
        min_tensor = min_tensor.view(-1, 1, 1)
    if max_tensor.ndim == 1:
        max_tensor = max_tensor.view(-1, 1, 1)

    tensor_rev = (tensor * (max_tensor - min_tensor)) + min_tensor
    tensor_rev = torch.nan_to_num(tensor_rev, nan=0.0)

    return tensor_rev


def scale_data(
    data: torch.Tensor,
    scaler: str | None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    min_value: Sequence[float] | None = None,
    max_value: Sequence[float] | None = None,
    inplace: bool = True,
):
    """Scale *data* according to the *scaler* defined.

    Args:
        data: torch.Tensor to scale.
        scaler: one of "std", "minmax", "local_mean_std".
        mean: (Optional) mean value - only used with std scaler.
        std: (Optional) std value  - only used with std, local_mean_std scalers.
        min_value: (Optional) min value - only used with minmax scaler.
        max_value: (Optional) max value - only used with minmax scaler.
        inplace: whether "std" and "local_mean_std" scaling operation should be done inplace.
    """
    if scaler == "std":
        if mean is None or std is None:
            raise ValueError("Std scaler selected but no mean/std provided.")
        return TF.normalize(data, mean=mean, std=std, inplace=inplace)
    elif scaler == "minmax":
        if min_value is None or max_value is None:
            raise ValueError("Minmax scaler selected but no min/max provided.")
        return minmax_op(tensor=data, min_value=min_value, max_value=max_value)
    elif scaler == "local_mean_std":
        if std is None:
            raise ValueError("local_mean_std scaler selected but no std provided.")
        return TF.normalize(
            data, mean=torch.mean(data, dim=(-1, -2)), std=std, inplace=inplace
        )
    elif scaler is None:
        return data
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")


def unscale_data(
    data: torch.Tensor,
    scaler: str | None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    min_value: Sequence[float] | None = None,
    max_value: Sequence[float] | None = None,
    inplace: bool = False,
):
    """Revert scaling done in *data* according to the *scaler* defined.

    Args:
        data: torch.Tensor to scale.
        scaler: one of "std", "minmax".
        mean: (Optional) mean value - only used with std scaler.
        std: (Optional) std value - only used with std scaler.
        min_value: (Optional) mean value - only used with minmax scaler.
        max_value: (Optional) std value - only used with minmax scaler.
        inplace: whether "std" and "local_mean_std" scaling operation should be done inplace.
    """
    if scaler in ["std", "local_mean_std"]:
        if mean is None or std is None:
            raise ValueError("Std scaler selected but no mean/std provided.")
        return TF.normalize(
            data.clone(),
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std],
            inplace=inplace,
        )
    elif scaler == "minmax":
        if min_value is None or max_value is None:
            raise ValueError("Minmax scaler selected but no min/max provided.")
        return minmax_op_reverse(data.clone(), min_value=min_value, max_value=max_value)
    elif scaler is None:
        return data
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")


def domain_unscale(
    data: torch.Tensor,
    domain: str,
    scaler_dict: dict[str, str | None] | None,
    stats: dict,
) -> torch.Tensor:
    """Helper to unscale domain data based on selected scalers."""
    if domain in [
        "vis",
        "vis_604",
        "uv",
        "dtm",
        "slope",
        "aspect",
        "nac",
        "dtm_3m",
        "slope_3m",
        "aspect_3m",
        "psr",
        "wac_mosaic",
    ]:
        x_unscaled = unscale_data(
            data,
            scaler=scaler_dict[domain] if scaler_dict is not None else None,
            mean=stats[domain]["mean"],
            std=stats[domain]["std"],
            min_value=stats[domain]["min"],
            max_value=stats[domain]["max"],
        )
    else:
        print(f"Domain {domain} not found. No unscaling performed.")
        x_unscaled = data

    return x_unscaled


def one_hot_encoder(sample: torch.Tensor, num_classes: int):
    """Convert sample with *num_classes* to one-hot encoding.

    Args:
        sample: torch.Tensor with shape (1, H, W)
        num_classes: number of classes to consider in the on-hot encodign scheme.

    Returns:
        one-hot encoded sample with shape (num_classes, H, W)
    """
    sample = sample.long()
    sample = torch.squeeze(sample)  # Remove the band dimension -> (H, W)
    one_hot = F.one_hot(sample, num_classes=num_classes).float()  # (H, W, num_classes)
    one_hot = rearrange(one_hot, "y x c -> c y x")

    return one_hot


def fill_nan_nearest_neighbor(img: np.ndarray) -> np.ndarray:
    """Fill NaNs using nearest neighbors (channel-first, pixel-wise).

    Args:
        img: (C, H, W) float array with NaNs

    Returns:
        filled array
    """
    if img.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got {img.shape}")

    # valid mask over spatial dimensions
    valid = np.isfinite(img).all(axis=0)  # (H, W)

    if valid.all():
        return img

    invalid_mask = ~valid

    idx = ndimage.distance_transform_edt(
        invalid_mask, return_indices=True, return_distances=False
    )
    rr, cc = idx

    filled = img.copy()
    filled[:, invalid_mask] = img[:, rr[invalid_mask], cc[invalid_mask]]

    return filled


class UnifiedDataTransform(object):
    def __init__(
        self,
        transforms_dict,
        image_augmenter,
        add_sizes: bool = False,
    ):
        """Unified data transform - preprocess + augmentation + postprocess (if available).

        Args:
            transforms_dict (dict): Dict of transforms for each modality
            image_augmenter (AbstractImageAugmenter): Image augmenter
            add_sizes (bool, optional): Whether to add crop coordinates and original size to the output dict
        """

        self.transforms_dict = transforms_dict
        self.image_augmenter = image_augmenter
        self.add_sizes = add_sizes

    def unified_image_augment(self, mod_dict, crop_settings):
        """Apply the image augmenter to all modalities where it is applicable.

        Args:
            mod_dict (dict): Dict of modalities
            crop_settings (dict): Crop settings

        Returns:
            dict: Transformed dict of modalities
        """

        crop_coords, flip, orig_size, target_size, rand_aug_idx = self.image_augmenter(
            mod_dict, crop_settings
        )

        mod_dict = {
            k: get_transform(k, self.transforms_dict).image_augment(
                v,
                crop_coords=crop_coords,
                flip=flip,
                orig_size=orig_size,
                target_size=target_size,
                rand_aug_idx=rand_aug_idx,
            )
            for k, v in mod_dict.items()
        }

        if self.add_sizes:
            mod_dict["crop_coords"] = torch.tensor(crop_coords)
            mod_dict["orig_size"] = torch.tensor(orig_size)

        return mod_dict

    def __call__(self, mod_dict):
        """Apply the augmentation to a dict of modalities (both image based and sequence based modalities).

        Args:
            mod_dict (dict): Dict of modalities

        Returns:
            dict: Transformed dict of modalities
        """
        crop_settings = mod_dict.pop("crop_settings", None)

        mod_dict = {
            k: get_transform(k, self.transforms_dict).preprocess(v)
            for k, v in mod_dict.items()
        }

        if self.image_augmenter is not None:
            mod_dict = self.unified_image_augment(mod_dict, crop_settings)

        mod_dict = {
            k: get_transform(k, self.transforms_dict).postprocess(v)
            for k, v in mod_dict.items()
        }

        return mod_dict

    def __repr__(self):
        return "(UnifiedDataTransform,\n)"


class AbstractTransform(ABC):
    """Base Transform class."""

    @abstractmethod
    def load(self, path: str) -> np.ndarray | str:
        pass

    @abstractmethod
    def preprocess(self, sample: np.ndarray | str) -> torch.Tensor:
        pass

    @abstractmethod
    def image_augment(
        self,
        sample: torch.Tensor,
        crop_coords: tuple,
        flip: bool,
        orig_size: tuple,
        target_size: tuple,
        rand_aug_idx: int | None,
        resample_mode: str | None = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, sample: torch.Tensor) -> torch.Tensor:
        pass


class ImageTransform(AbstractTransform):
    """Image Transform class."""

    @staticmethod
    def numpy_loader(path: str) -> np.ndarray:
        img = np.load(path)
        return img

    @staticmethod
    def netcdf_loader(path: str, channels: Sequence[str] | None = None) -> np.ndarray:
        data = load_netcdf(path=path, channels=channels)

        if np.isnan(data).any():
            print(f"NaN detected in data from file '{path}'")
        return data

    @staticmethod
    def image_hflip(img: torch.Tensor, flip: bool):
        """Crop and resize an image.

        Args:
            img: Image to crop and resize
            flip: Whether to flip the image

        Returns:
            Flipped image (if flip = True)
        """
        if flip:
            img = TF.hflip(img)
        return img

    @staticmethod
    def image_crop_and_resize(
        img: torch.Tensor,
        crop_coords: tuple,
        target_size: tuple,
        resample_mode: str = "bilinear",
    ):
        """Crop and resize an image.

        Args:
            img: Image to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)
            target_size: Coordinates of the resize (height, width)
            resample_mode: resample mode string

        Returns:
            Cropped and resized image
        """

        top, left, h, w = crop_coords
        resize_height, resize_width = target_size
        img = TF.crop(img, top, left, h, w)
        mode = get_resample_mode(resample_mode)
        img = TF.resize(img, size=[resize_height, resize_width], interpolation=mode)
        return img

    @staticmethod
    def image_resize(
        img: torch.Tensor, target_size: Sequence[int], resample_mode: str = "bilinear"
    ):
        """Resize an image.

        Args:
            img: Image to crop and resize
            target_size: Coordinates of the resize (height, width)
            resample_mode: resample mode string

        Returns:
            Resized image
        """
        resize_height, resize_width = target_size
        mode = get_resample_mode(resample_mode)
        img = TF.resize(img, size=[resize_height, resize_width], interpolation=mode)
        return img

    @staticmethod
    def image_crop(img: torch.Tensor, crop_coords: tuple, **kwargs):
        """Crop an image.

        Args:
            img: Image to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)

        Returns:
            Cropped image
        """

        top, left, h, w = crop_coords
        img = TF.crop(img, top, left, h, w)
        return img


class LunarTransform(ImageTransform):
    """Lunar data Transform class."""

    def __init__(
        self,
        mean: list,
        std: list,
        max: list,
        min: list,
        channels: list,
        scaler: str | None,
        pre_resize: int | None = None,
        resample_mode: str = "bilinear",  # One out of ["bilinear", "bicubic", "nearest"]
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.channels = channels
        self.scaler = scaler
        self.pre_resize = pre_resize
        self.resample_mode = resample_mode

    def load(self, path) -> np.ndarray:
        sample = self.netcdf_loader(path, self.channels)
        return sample

    def preprocess(self, sample: np.ndarray):
        img = torch.Tensor(sample)
        img = torch.nan_to_num(img, nan=0.0)  # safety replacement to avoid issues
        if self.pre_resize is not None:
            img = self.image_resize(
                img, [self.pre_resize] * 2, resample_mode=self.resample_mode
            )
        img = scale_data(
            img,
            mean=self.mean,
            std=self.std,
            min_value=self.min,
            max_value=self.max,
            scaler=self.scaler,
        )

        return img

    def image_augment(
        self,
        img: torch.Tensor,
        crop_coords: tuple,
        flip: bool,
        target_size: tuple,
        **kwargs,
    ):
        img = self.image_crop_and_resize(
            img, crop_coords, target_size, resample_mode=self.resample_mode
        )
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample: torch.Tensor):
        return sample


class GeomapTransform(ImageTransform):
    """Lunar Geomap data Transform class."""

    def __init__(
        self,
        channels: list,
        one_hot_encoding: int | None,
        pre_resize: int | None = None,
        class_mapping: dict[int, int] | None = None,
        **kwargs,
    ):
        self.channels = channels
        self.one_hot_encoding = one_hot_encoding
        self.pre_resize = pre_resize
        self.class_mapping = class_mapping

        # Pre-compute lookup tensor for class reduction if mapping is provided
        self.lookup = None
        if class_mapping is not None:
            max_class = max(class_mapping.keys()) + 1
            self.lookup = torch.full((max_class,), -1, dtype=torch.long, device="cpu")
            for orig_idx, reduced_idx in class_mapping.items():
                self.lookup[orig_idx] = reduced_idx

    def load(self, path):
        sample = self.netcdf_loader(path, self.channels)
        return sample

    def preprocess(self, sample: np.ndarray):
        sample = fill_nan_nearest_neighbor(img=sample)
        processed = torch.Tensor(sample)
        if self.pre_resize is not None:
            processed = self.image_resize(
                processed, [self.pre_resize] * 2, resample_mode="nearest"
            )

        # Apply class reduction if required
        if self.class_mapping is not None:
            processed = self._apply_class_reduction(processed)

        if self.one_hot_encoding is not None:
            processed = one_hot_encoder(processed, num_classes=self.one_hot_encoding)

        return processed

    def _apply_class_reduction(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply class reduction mapping from geomap DN values (0-49) to 16 classes.

        Args:
            sample: Tensor with shape (1, H, W) containing DN values 0-49 (0=background, 1-49=geological units)

        Returns:
            Tensor with shape (1, H, W) containing reduced class indices 0-15 (0-14=geological, 15=background)
        """
        if self.lookup is None:
            raise ValueError(
                "Class mapping lookup tensor not initialized. Provide class_mapping in __init__."
            )

        lookup = self.lookup.to(sample.device)

        # Apply mapping
        sample_long = sample.long()
        reduced = lookup[sample_long]

        # Handle any unmapped classes - set to background class
        reduced = torch.where(
            reduced == -1,
            torch.tensor(self.class_mapping[0], device=sample.device),
            reduced,
        )

        return reduced.float()

    def image_augment(
        self,
        img,
        crop_coords: tuple,
        flip: bool,
        target_size: tuple,
        **kwargs,
    ):
        img = self.image_crop_and_resize(
            img, crop_coords, target_size, resample_mode="nearest"
        )
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample: torch.Tensor):
        return sample


class CraterMasksTransform(ImageTransform):
    """Lunar Crater Masks data Transform class."""

    def __init__(
        self,
        channels: list,
        one_hot_encoding: int | None,
        pre_resize: int | None = None,
        **kwargs,
    ):
        self.channels = channels
        self.one_hot_encoding = one_hot_encoding
        self.pre_resize = pre_resize

    def load(self, path):
        ext = os.path.splitext(path)[-1]
        if ext == ".nc":
            sample = self.netcdf_loader(path, self.channels)
        elif ext == ".npy":
            sample = self.numpy_loader(path)

        if sample.ndim == 2:
            sample = sample[np.newaxis, ...]

        return sample

    def preprocess(self, sample: np.ndarray):
        sample = fill_nan_nearest_neighbor(img=sample)
        processed = torch.Tensor(sample)
        if self.pre_resize is not None:
            processed = self.image_resize(
                processed, [self.pre_resize] * 2, resample_mode="nearest"
            )

        if self.one_hot_encoding is not None:
            processed = one_hot_encoder(processed, num_classes=self.one_hot_encoding)

        return processed

    def image_augment(
        self,
        img,
        crop_coords: tuple,
        flip: bool,
        target_size: tuple,
        **kwargs,
    ):
        img = self.image_crop_and_resize(
            img, crop_coords, target_size, resample_mode="nearest"
        )
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample: torch.Tensor):
        return sample


class UntokLunarTransform(LunarTransform):
    """Untokenized Lunar Transform class."""

    def image_augment(self, img, crop_coords: tuple, flip: bool, **kwargs):
        img = self.image_crop(img, crop_coords)
        img = self.image_hflip(img, flip)
        return img


class UntokGeomapTransform(GeomapTransform):
    """Untokenized Geomap Transform class."""

    def image_augment(self, img, crop_coords: tuple, flip: bool, **kwargs):
        img = self.image_crop(img, crop_coords)
        img = self.image_hflip(img, flip)
        return img


class UntokCraterMasksTransform(CraterMasksTransform):
    """Untokenized Geomap Transform class."""

    def image_augment(self, img, crop_coords: tuple, flip: bool, **kwargs):
        img = self.image_crop(img, crop_coords)
        img = self.image_hflip(img, flip)
        return img


class MaskTransform(ImageTransform):
    """Mask Transform class."""

    def __init__(self, mask_pool_size=1):
        assert isinstance(mask_pool_size, int)
        self.mask_pool_size = mask_pool_size  # Use to expand masks

    def mask_to_tensor(self, img):
        mask = TF.to_tensor(img)
        if self.mask_pool_size > 1:
            mask = reduce(
                mask,
                "c (h1 h2) (w1 w2) -> c h1 w1",
                "min",
                h2=self.mask_pool_size,
                w2=self.mask_pool_size,
            )
            mask = repeat(
                mask,
                "c h1 w1 -> c (h1 h2) (w1 w2)",
                h2=self.mask_pool_size,
                w2=self.mask_pool_size,
            )
        return mask == 1.0

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(
        self, img, crop_coords: tuple, flip: bool, target_size: tuple, **kwargs
    ):
        # Override resampling mode to 'nearest' for masks
        img = self.image_crop_and_resize(
            img, crop_coords, target_size, resample_mode="nearest"
        )
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.mask_to_tensor(sample)
        return sample


class TokTransform(AbstractTransform):
    """Tokenized base Transform class."""

    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample: np.ndarray):
        # Squeeze last dimension if it's 1 -> handles 15k tokenizers with num_codebooks=1
        if self.num_codebooks == 1:
            sample = sample.squeeze(-1)
        return sample

    def image_augment(self, v, rand_aug_idx: int | None, **kwargs):
        if rand_aug_idx is None:
            raise ValueError(
                "Crop settings/augmentation index are missing but a pre-tokenized modality is being used"
            )
        v = torch.tensor(v[rand_aug_idx])
        return v

    def postprocess(self, sample):
        return sample


class SingleValueTransform(AbstractTransform):
    """Base class for transforms that handle single-value parameters with binning."""

    def __init__(self, return_raw: bool, shuffle: bool, aggregation_method: str):
        self.return_raw = return_raw
        self.shuffle = shuffle
        self.aggregation_method = aggregation_method
        self.bin_sizes = None

    def load(self, path):
        return Path(path).read_text()

    def _load_binning_config(self, csv_path: str) -> dict[str, float]:
        """Load binning configuration from CSV file with format: parameter,min,max,step,bins_n."""

        if not os.path.exists(csv_path):
            raise ValueError(f"Binning config file not found: {csv_path}")

        bin_sizes = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_name = row["parameter"]
                step = float(row["step"])
                bin_sizes[param_name] = step
        return bin_sizes

    def _parse_items(self, sample: str) -> list[list[str]]:
        """Parse str sample into list of [key, value_str] lists."""

        items = []
        for raw_line in sample.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"Invalid line: {raw_line!r}")
            key, value = line.split("=", 1)
            items.append([key.strip(), value.strip()])

        return items

    @staticmethod
    def _bin_value(
        field_name: str,
        value: float,
        bin_sizes: dict[str, float],
        decimals: int | None = None,
    ) -> str:
        """Bin a value into the tokenizer's range-token format.

        Args:
            field_name: Field name
            value: Raw numeric value
            bin_sizes: Dictionary mapping field names to bin sizes

        Returns:
            Binned token string in format "FIELD=start-->end"
        """
        if field_name not in bin_sizes:
            raise KeyError(f"No bin size configured for field: {field_name}")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Value must be finite, got {value!r}")

        bin_size = bin_sizes[field_name]
        lower = math.floor(numeric_value / bin_size) * bin_size
        upper = lower + bin_size

        # Format with appropriate decimal places based on bin size
        if decimals is None:
            if bin_size >= 1.0:
                decimals = 0
            elif bin_size >= 0.1:
                decimals = 1
            else:
                decimals = 2

        return f"{field_name}={lower:.{decimals}f}-->{upper:.{decimals}f}"

    def preprocess(self, sample: str):
        return sample

    def image_augment(self, sample, **kwargs):
        return sample

    def _parse_value(self, value_str: str) -> float | list[float] | None:
        """Parse a value string, handling multiple values and ERROR cases."""
        value_str = value_str.strip()

        if value_str.upper() in ("ERROR", "MISSING", "NAN", ""):
            return None

        # Check if multiple values (comma-separated)
        if "," in value_str:
            values = []
            for v in value_str.split(","):
                v = v.strip()
                if v.lower() == "nan" or not v:
                    continue
                try:
                    values.append(float(v))
                except ValueError:
                    continue

            if len(values) == 0:
                return None

            return values
        else:
            return float(value_str)

    def parse_sample(self, sample: str, var_list: list | None = None) -> list[list]:
        """Parse and filter keys to specified vars only."""
        items = self._parse_items(sample)

        processed_items = []
        for key, value_str in items:
            if var_list is not None and key not in var_list:
                continue

            parsed_value = self._parse_value(value_str)
            processed_items.append([key, parsed_value])

        return processed_items

    def _aggregate_values(self, values: list[float]) -> float | list[float]:
        """Aggregate multiple values using the specified method."""
        if len(values) == 0:
            raise ValueError("Cannot aggregate empty list of values")

        if self.aggregation_method == "min":
            return min(values)
        elif self.aggregation_method == "max":
            return max(values)
        elif self.aggregation_method == "mean":
            return sum(values) / len(values)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def sample_to_str(self, items: list[list], decimals: int | None) -> list[str]:
        """Format items to binned text strings or MISSING markers."""
        result = []
        for key, value in items:
            if value is None:
                result.append(f"{key}=MISSING")
            elif isinstance(value, list):
                binned = self._bin_value(
                    key, self._aggregate_values(value), self.bin_sizes, decimals
                )
                result.append(binned)
            else:
                result.append(self._bin_value(key, value, self.bin_sizes, decimals))

        return result


class MetadataTransform(SingleValueTransform):
    """Metadata transform for observation parameters."""

    METADATA_VARS = (
        "SS_GROUND_AZIMUTH",
        "SS_LAT",
        "SS_LON",
        "PHASE_ANG",
        "INC_ANG",
        "EM_ANG",
        "UL_LON",
        "UL_LAT",
        "LR_LON",
        "LR_LAT",
    )

    def __init__(
        self,
        binning_config_path: str = DEFAULT_BINNING_FILE,
        return_raw: bool = False,
        shuffle: bool = True,
    ):
        """Initialize MetadataTransform.

        Args:
            binning_config_path: Path to CSV file with binning configuration.
            return_raw: whether to return raw list of lists or binned text strings.
            shuffle: whether to shuffle the order of metadata entries.
        """
        super().__init__(
            return_raw, shuffle, aggregation_method="mean"
        )  # aggregation method not currently used
        all_bin_sizes = self._load_binning_config(binning_config_path)
        assert set(self.METADATA_VARS).issubset(set(all_bin_sizes.keys()))

        self.bin_sizes = {v: all_bin_sizes[v] for v in self.METADATA_VARS}
        self.decimals = 2  # Metadata vocabulary has 2 decimals
        self.selected_vars = list(self.METADATA_VARS)

    def postprocess(self, sample: str):

        items = self.parse_sample(sample, var_list=self.selected_vars)

        if self.return_raw:
            return items

        if self.shuffle:
            random.shuffle(items)

        return self.sample_to_str(items, decimals=self.decimals)


class StaticMapsTransform(SingleValueTransform):
    """Transform for static map variables."""

    def __init__(
        self,
        selected_vars: list[str] | None = None,
        aggregation_method: str = "mean",
        binning_config_path: str = DEFAULT_BINNING_FILE,
        return_raw: bool = False,
        shuffle: bool = True,
    ):
        """Initialize StaticMapsTransform.

        Args:
            selected_vars: List of variable names to tokenize. If None, uses all variables except 10 metadata variables
            aggregation_method: Method to aggregate multiple values: "min", "mean", or "max"
            binning_config_path: Path to CSV file with binning configuration.
            return_raw: whether to return raw list of lists or binned text strings.
            shuffle: whether to shuffle the order of entries.
        """
        super().__init__(return_raw, shuffle, aggregation_method)

        all_bin_sizes = self._load_binning_config(binning_config_path)
        exclude = MetadataTransform.METADATA_VARS  # Exclude metadata variables

        if selected_vars is None:
            self.selected_vars = [v for v in all_bin_sizes if v not in exclude]
            self.bin_sizes = {
                v: step for v, step in all_bin_sizes.items() if v not in exclude
            }
        else:
            self.selected_vars = selected_vars
            self.bin_sizes = {
                v: all_bin_sizes[v] for v in self.selected_vars if v in all_bin_sizes
            }

    def postprocess(self, sample: str):

        items = self.parse_sample(sample, var_list=self.selected_vars)

        if self.return_raw:
            return items

        if self.shuffle:
            random.shuffle(items)

        return self.sample_to_str(items, decimals=None)


class BBoxTransform(AbstractTransform):
    """BBox Transform class for crater bounding boxes.

    Args:
        input_size: input_size used for the model (size images will be cropped to).
        original_size: the size of the images bboxes were collected from.
        pre_resize: pre-resize applied to *vis* or *tok_vis* modalities
        return_raw: whether to return parsed bboxes as numpy array.
    """

    def __init__(
        self,
        original_size: int,
        pre_resize: int | None,
        bbox_order: str = "dist_to_orig",
        decimals: int = 3,
        return_raw: bool = False,
    ):
        self.original_size = original_size
        self.pre_resize = pre_resize if pre_resize is not None else original_size
        self.format = "xywh"
        self.decimals = decimals
        self.return_raw = return_raw

        if bbox_order == "area":
            self.bbox_order = self.order_bboxes_by_area
        elif bbox_order == "random":
            self.bbox_order = self.shuffle_bboxes
        else:
            self.bbox_order = self.order_bboxes_by_dist_to_orig

    @staticmethod
    def order_bboxes_by_area(bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return bboxes
        areas = bboxes[:, 2] * bboxes[:, 3]
        order = np.argsort(areas)[::-1]  # descending order
        return bboxes[order]

    @staticmethod
    def order_bboxes_by_dist_to_orig(bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return bboxes
        dist = bboxes[:, 0] ** 2 + bboxes[:, 1] ** 2
        order = np.argsort(dist)
        return bboxes[order]

    @staticmethod
    def shuffle_bboxes(bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return bboxes
        order = np.random.permutation(len(bboxes))
        return bboxes[order]

    def load(self, path):
        return Path(path).read_text()

    @staticmethod
    def _parse_bbox_coords(text: str) -> np.ndarray:
        """Parse bbox coordinates from text. Each line must contain: [xmin ymin width height]."""
        bboxes = []
        for line in text.strip().split("\n"):
            line_clean = line.strip()
            if not line_clean:
                continue
            # Remove brackets if present
            line_clean = line_clean.replace("[", "").replace("]", "")
            values = [float(v) for v in line_clean.split()]
            if len(values) == 4:
                bboxes.append(values)
        return np.asarray(bboxes)

    @staticmethod
    def crop_bboxes(bboxes: np.ndarray, cx, cy, cw, ch) -> np.ndarray:
        """Crop bboxes to crop_origin.

        Args:
            bboxes: bounding boxes in (xmin, ymin, w, h) format
            cx, cy: left and top coordinates of the crop
            cw, ch: width and height of the crop
        """
        if len(bboxes) == 0:
            return bboxes

        xmin, ymin, w, h = bboxes.T  # xywh to xyxy

        # Compute intersections
        ix1 = np.maximum(xmin, cx)
        iy1 = np.maximum(ymin, cy)
        ix2 = np.minimum(xmin + w, cx + cw)
        iy2 = np.minimum(ymin + h, cy + ch)

        # Filter valid intersections
        valid = (ix1 < ix2) & (iy1 < iy2)

        # back to xywh (relative to crop)
        cropped = np.stack(
            [
                ix1[valid] - cx,
                iy1[valid] - cy,
                ix2[valid] - ix1[valid],
                iy2[valid] - iy1[valid],
            ],
            axis=1,
        )

        return cropped

    @staticmethod
    def normalize_bbox_coords(
        coords: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Convert bbox coordinates to normalized [0, 1] range.

        Args:
            coords: Bbox pixel coords (N, 4) in format (xmin, ymin, w, h)
            height: height of the image the coords are based on
            width: width of the image the coords are based on

        Returns:
            Normalized coordinates (N, 4) in format (xmin, ymin, w, h)
        """
        if len(coords) == 0:
            return coords
        factors = np.asarray(
            [width, height, width, height]
        )  # (xmin, w) / width, (ymin, h) / height
        normalized = coords / factors
        return np.clip(normalized, 0.0, 1.0)

    def preprocess(self, sample: str) -> np.ndarray:
        """Parse bbox coordinates and return dict with text and bbox_coords."""
        bbox_coords = self._parse_bbox_coords(sample)

        # Scale bboxes if pre-resize was applied
        if self.pre_resize is not None and self.pre_resize != self.original_size:
            scale_factor = self.pre_resize / self.original_size
            bbox_coords = bbox_coords * scale_factor

        return bbox_coords

    def image_augment(
        self, bbox_coords: np.ndarray, crop_coords: tuple, flip: bool, **kwargs
    ):
        """Handle spatial augmentations for bboxes.

        Args:
            bbox_coords: bounding boxes in (xmin, ymin, w, h) format
            crop_coords: Coordinates of the crop (top, left, h, w)
            flip: Whether to flip horizontally.
        """
        if len(bbox_coords) > 0:
            if crop_coords is not None:
                bbox_coords = self.crop_bboxes(
                    bboxes=bbox_coords,
                    cx=crop_coords[1],
                    cy=crop_coords[0],  # swap x,y
                    cw=crop_coords[3],
                    ch=crop_coords[2],
                )  # swap h,w
            if flip and len(bbox_coords) > 0:
                crop_w = crop_coords[3]  # Width from crop_coords (top, left, h, w)
                bbox_coords[:, 0] = crop_w - bbox_coords[:, 0] - bbox_coords[:, 2]

            # Normalize after cropping
            if crop_coords is not None:
                bbox_coords = self.normalize_bbox_coords(
                    bbox_coords, height=crop_coords[2], width=crop_coords[3]
                )
            else:
                bbox_coords = self.normalize_bbox_coords(
                    bbox_coords, height=self.pre_resize, width=self.pre_resize
                )

            bbox_coords = self.bbox_order(bbox_coords)

        return bbox_coords

    def _format_coord(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"

    def format_bbox_text(self, coords: np.ndarray, separator: str = " ") -> str:
        """Format bbox coordinates as tokenizable text with normalized float values.

        Args:
            coords: Normalized coordinates (N, 4) in format (xmin, ymin, w, h) or (xmin, ymin, xmax, ymax)
            separator: Separator between coordinate tokens (default: " ", use "" for no separator)

        Returns:
            Formatted text string with coordinates as normalized floats

        Example:
            >>> coords = np.array([[0.1, 0.2, 0.3, 0.4]])
            >>> text = format_bbox_text(coords, format="xywh", decimals=3, separator="")
            >>> # Result: "xmin=0.100ymin=0.200xmax=0.400ymax=0.600"
        """
        if coords.size == 0:
            return ""

        tokens = []
        for bbox in coords:
            if self.format == "xyxy":
                # Expect (xmin, ymin, xmax, ymax)
                tokens.extend(
                    [
                        f"xmin={self._format_coord(bbox[0].item())}",
                        f"ymin={self._format_coord(bbox[1].item())}",
                        f"xmax={self._format_coord(bbox[2].item())}",
                        f"ymax={self._format_coord(bbox[3].item())}",
                    ]
                )
            elif self.format == "xywh":
                # Expect (xmin, ymin, w, h), convert to xyxy
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                tokens.extend(
                    [
                        f"xmin={self._format_coord(xmin.item())}",
                        f"ymin={self._format_coord(ymin.item())}",
                        f"xmax={self._format_coord(xmax.item())}",
                        f"ymax={self._format_coord(ymax.item())}",
                    ]
                )
            else:
                raise ValueError(f"Unknown format: {self.format}")

        return separator.join(tokens)

    @staticmethod
    def reverse_format_bbox_text(
        bbox_text: str, bbox_format: str = "xywh", separator: str = " "
    ) -> np.ndarray:
        """Reverse of format_bbox_text: parse text back to numpy array.

        Args:
            bbox_text: String formatted as "xmin=0.100 ymin=0.200 xmax=0.400 ymax=0.600 ..."
            bbox_format: Output format - "xywh" or "xyxy"
                - "xywh": Returns (xmin, ymin, w, h)
                - "xyxy": Returns (xmin, ymin, xmax, ymax)
            separator: Separator used in the text (default: " ", can be "" for no separator)

        Returns:
            Numpy array of shape (N, 4) with normalized coordinates [0, 1]
        """
        if (not bbox_text) or (not bbox_text.strip()):
            return np.empty((0, 4))

        # Pattern matches: xmin=X<sep>ymin=Y<sep>xmax=Z<sep>ymax=W
        sep_pattern = re.escape(separator) + r"\s*" if separator else r"\s*"

        pattern = (
            r"xmin=([\d.]+)"
            + sep_pattern
            + r"ymin=([\d.]+)"
            + sep_pattern
            + r"xmax=([\d.]+)"
            + sep_pattern
            + r"ymax=([\d.]+)"
        )

        matches = re.findall(pattern, bbox_text)

        if not matches:
            return np.empty((0, 4))

        # Convert to numpy array: each match is (xmin, ymin, xmax, ymax)
        bboxes_xyxy = np.array([[float(x) for x in match] for match in matches])

        if bbox_format == "xywh":
            # Convert from (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
            xmin = bboxes_xyxy[:, 0]
            ymin = bboxes_xyxy[:, 1]
            xmax = bboxes_xyxy[:, 2]
            ymax = bboxes_xyxy[:, 3]
            w = xmax - xmin
            h = ymax - ymin
            return np.stack([xmin, ymin, w, h], axis=1)
        elif bbox_format == "xyxy":
            return bboxes_xyxy
        else:
            raise ValueError(f"Unknown format: {bbox_format}")

    def postprocess(self, sample: np.ndarray):
        if self.return_raw:
            return sample
        text = self.format_bbox_text(sample)
        return text


class IdentityTransform(AbstractTransform):
    """Identity Transform class."""

    def load(self, path):
        raise NotImplementedError("IdentityTransform does not support loading")

    def preprocess(self, sample, **kwargs):
        return sample

    def image_augment(self, sample, **kwargs):
        return sample

    def postprocess(self, sample, **kwargs):
        return sample
