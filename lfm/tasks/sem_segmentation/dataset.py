"""
dataset.py
Dataset class for DINO LoRA fine-tuning on lunar crater detection.
Supports images with any number of channels (grayscale, RGB, multispectral, etc.).
"""

import os
from pathlib import Path
from glob import glob
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import rasterio


class LunarCraterDataset(Dataset):
    """
    Dataset for multi-channel images and segmentation masks.
    Supports images with any number of channels.

    Args:
        image_dir: Directory with .npy images (H, W, C) or (C, H, W)
        label_dir: Directory with .npy label masks (H, W)
        mean: Mean values per channel for normalization (shape: [num_channels])
        std: Standard deviation per channel for normalization
            (shape: [num_channels])
        target_size: Target size for model input. Default: (304, 304)
        max_samples: Max samples to use. None uses all samples.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        target_size: Tuple[int, int] = (304, 304),
        max_samples: Optional[int] = None,
        input_file_type: str = ".npy",
        label_file_type: str = ".npy",
        band_indices_filter: List[int] = None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # Store mean and std as float32 for consistency
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.num_channels = len(mean)

        # Glob all images and labels
        if input_file_type not in [".npy", ".npz", ".tif"]:
            raise ValueError(
                "Inputs are expected to be of type .npy, .npz, or .tif"
            )
        if label_file_type not in [".npy", ".npz"]:
            raise ValueError("Inputs are expected to be of type .npy or .npz")
        self.input_file_type = input_file_type
        self.label_file_type = label_file_type
        self.image_paths = sorted(
            glob(os.path.join(image_dir, f"*{self.input_file_type}"))
        )
        label_paths = sorted(
            glob(os.path.join(label_dir, f"*{self.label_file_type}"))
        )

        # Load example to validate input band number
        example_band_number = self._load_example_input(self.image_paths[0])
        if band_indices_filter is not None:
            if example_band_number < len(band_indices_filter):
                raise ValueError(
                    f"Incompatible input band filter specified.\n"
                    f"Number of bands found: {example_band_number}\n"
                    f"Band filter: {band_indices_filter}, "
                    f"len={len(band_indices_filter)}"
                )
            print(f"Filtered to channels: {band_indices_filter}")
            self.band_indices_filter = band_indices_filter
        else:
            self.band_indices_filter = list(range(example_band_number))

        # Extract basenames for matching
        self.image_basenames = [
            "_".join(Path(filename).stem.split("_")[:-1])
            for filename in self.image_paths
        ]
        label_basenames = [
            "_".join(Path(filename).stem.split("_")[:-1])
            for filename in label_paths
        ]

        # Create lookup dictionary for labels
        self.label_lookup = {
            basename: path
            for basename, path in zip(label_basenames, label_paths)
        }

        # Filter to only images that have matching labels
        self.valid_indices = []
        self.valid_image_paths = []
        self.valid_label_paths = []

        for idx, (img_path, basename) in enumerate(
            zip(self.image_paths, self.image_basenames)
        ):
            if basename in self.label_lookup:
                self.valid_indices.append(idx)
                self.valid_image_paths.append(img_path)
                self.valid_label_paths.append(self.label_lookup[basename])

        # Limit to max_samples if specified
        if max_samples is not None and max_samples < len(
            self.valid_image_paths
        ):
            self.valid_image_paths = self.valid_image_paths[:max_samples]
            self.valid_label_paths = self.valid_label_paths[:max_samples]
            print(f"Limited to {max_samples} samples")

        print(f"Found {len(self.valid_image_paths)} matched image-label pairs")
        print(
            f"Dataset configured for {len(self.band_indices_filter)} channel(s)"
        )

        if len(self.valid_image_paths) == 0:
            raise ValueError(
                "No matching image-label pairs found! "
                "Check that basenames match between image_dir and label_dir"
            )

    def _load_example_input(self, img_path: str):
        if self.input_file_type in [".npy", ".npz"]:
            image = np.load(img_path).astype(
                np.float32
            )  # (H, W, C) or (C, H, W)
        else:  # .tif
            image = rasterio.open(img_path).read()

        if image.ndim == 3:
            # Determine format: if first dimension is smallest, assume (C, H, W)
            if image.shape[0] < min(image.shape[1], image.shape[2]):
                image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
            # Otherwise, assume already in (H, W, C) format

        elif image.ndim == 2:
            # Handle grayscale images without explicit channel dimension
            image = image[:, :, np.newaxis]  # Shape: (H, W, 1)
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image.shape} for {img_path}"
            )
        return image.shape[2]  # Return channel, which is final idx

    @staticmethod
    def _min_max_scale_bands(img: np.array):
        """
        Min-max scale each band to [0, 1]

        Args:
            img: Image array with shape (H, W, C)

        Returns:
            scaled: Scaled image with same shape
        """
        scaled = np.zeros_like(img, dtype=np.float32)

        # Iterate over channels (last dimension for H, W, C format)
        for i in range(img.shape[2]):  # Changed from shape[0] to shape[2]
            band = img[:, :, i]  # Get channel i
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                scaled[:, :, i] = (band - band_min) / (band_max - band_min)
            else:
                scaled[:, :, i] = band

        return scaled

    def __len__(self) -> int:
        return len(self.valid_image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image (torch.Tensor): Normalized and resized image (C, H, W)
            label (torch.Tensor): Resized label mask (H, W)
        """
        # Load image and label
        img_path = self.valid_image_paths[idx]
        label_path = self.valid_label_paths[idx]

        if self.input_file_type in [".npy", ".npz"]:
            image = np.load(img_path).astype(
                np.float32
            )  # (H, W, C) or (C, H, W)
        else:  # .tif
            image = rasterio.open(img_path).read()

        label = np.load(label_path).astype(np.int64)  # (H, W)

        # Handle different image shapes: (H, W, C), (C, H, W), or (H, W)
        if image.ndim == 3:
            # Determine format: if first dimension is smallest, assume (C, H, W)
            if image.shape[0] < min(image.shape[1], image.shape[2]):
                image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
            # Otherwise, assume already in (H, W, C) format

        elif image.ndim == 2:
            # Handle grayscale images without explicit channel dimension
            image = image[:, :, np.newaxis]  # Shape: (H, W, 1)
        else:
            raise ValueError(
                f"Unexpected image dimensions: {image.shape} for {img_path}"
            )

        # Verify channel count matches expected
        if image.shape[2] != self.num_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.num_channels}, "
                f"got {image.shape[2]} for {img_path}"
            )

        # Filter down to desired bands
        image = image[:, :, self.band_indices_filter]

        # .tif inputs have been saved as raw values; needs min/max
        if self.input_file_type == ".tif":
            image = LunarCraterDataset._min_max_scale_bands(image)

        # Normalize image with dataset statistics
        # Filter mean/std to use our band indices filter
        mean_filtered = (
            self.mean[self.band_indices_filter]
            if self.band_indices_filter is not None
            else self.mean
        )
        std_filtered = (
            self.std[self.band_indices_filter]
            if self.band_indices_filter is not None
            else self.std
        )

        # Reshape mean and std to (1, 1, C) for broadcasting with (H, W, C)
        mean_reshaped = mean_filtered.reshape(1, 1, -1)
        std_reshaped = std_filtered.reshape(1, 1, -1)
        image = (image - mean_reshaped) / std_reshaped

        # Convert to torch tensors in (C, H, W)/(H, W) for inputs, labels resp.
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label)

        # Resize image to target size for model, (C, target_H, target_W)
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Resize label using nearest neighbor to preserve class indices
        # Labels are NOT transformed, only resized to (target_H, target_W)
        label = (
            F.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(),
                size=self.target_size,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        return image, label


def calculate_dataset_statistics(
    image_dir: str, input_file_type: str, debug: bool = False
):
    """
    Calculate mean and standard deviation for a dataset. Used in normalization.
    """
    if input_file_type not in [".npy", ".npz", ".tif"]:
        raise ValueError(
            "Calculating dataset statistics expects .npy, .npz, or .tif "
            "filetypes."
        )

    input_paths = glob(os.path.join(image_dir, f"*{input_file_type}"))

    if len(input_paths) == 0:
        raise ValueError(f"No files found in {image_dir}")

    pixel_sum = None
    pixel_sq_sum = None
    pixel_count = 0
    valid_images = 0
    num_channels = None

    for image_path in tqdm(input_paths, desc="Computing dataset statistics"):
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            continue

        try:
            if input_file_type in [".npy", ".npz"]:
                img = np.load(image_path).astype(np.float32)
            else:  # .tif
                img = rasterio.open(image_path).read().astype(np.float32)

            # Handle different shapes: (H, W, C), (C, H, W), or (H, W)
            if img.ndim == 3:
                if img.shape[0] < min(img.shape[1], img.shape[2]):
                    img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
            elif img.ndim == 2:
                img = img[:, :, np.newaxis]
            else:
                print(
                    f"Warning: Image has {img.ndim} dimensions, skipping {image_path}"
                )
                continue

            # Initialize on first valid image
            if valid_images == 0:
                num_channels = img.shape[2]
                pixel_sum = np.zeros(num_channels)
                pixel_sq_sum = np.zeros(num_channels)
                print(f"First image shape: {img.shape}, dtype: {img.dtype}")
                print(f"Detected {num_channels} channel(s)")
                print(
                    f"Value range BEFORE preprocessing: min={img.min()}, max={img.max()}"
                )

            # Verify consistent number of channels
            if img.shape[2] != num_channels:
                print(
                    f"Warning: Inconsistent channels. Expected {num_channels}, "
                    f"got {img.shape[2]} for {image_path}. Skipping."
                )
                continue

            # For .tif files, apply min-max scaling per band
            if input_file_type == ".tif":
                for band_idx in range(img.shape[2]):
                    band = img[:, :, band_idx]
                    band_min, band_max = band.min(), band.max()
                    if band_max > band_min:
                        img[:, :, band_idx] = (band - band_min) / (
                            band_max - band_min
                        )
                    else:
                        img[:, :, band_idx] = band

                if valid_images == 0:
                    print(
                        f"Value range AFTER min-max scaling: min={img.min()}, max={img.max()}"
                    )

            # Accumulate statistics
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img**2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]
            valid_images += 1

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            continue

    if valid_images == 0:
        raise ValueError("No valid images were processed.")

    # Calculate mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

    print(
        f"\nProcessed {valid_images} valid images out of {len(input_paths)} total"
    )
    print(f"Mean per channel: {mean}")
    print(f"Std per channel: {std}")

    # Sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS:")
    print("=" * 70)
    if input_file_type == ".tif":
        print("✓ Min-max scaling was applied (expected for .tif)")
        print("✓ Mean should be ~0.3-0.7 (middle of [0,1] range)")
        print("✓ Std should be ~0.1-0.3")
    else:
        print("✓ No min-max scaling applied (for .npy/.npz)")

    for i, (m, s) in enumerate(zip(mean, std)):
        status = "✓" if 0 <= m <= 1 and s > 0 else "⚠️"
        print(f"{status} Band {i}: mean={m:.4f}, std={s:.4f}")

    return mean, std


def get_dataloaders(
    image_dir: str,
    label_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (304, 304),
    max_samples: Optional[int] = None,
    seed: int = 42,
    stats_save_dir: Optional[str] = None,
    input_file_type: str = ".npy",
    label_file_type: str = ".npy",
    debug: bool = False,
    band_indices_filter: List[int] = None,
):
    """
    Create train/val dataloaders with automatic statistics calculation.
    Statistics are used to z-score normalize inputs upon dataloader creation.
    Automatically handles images with any number of channels.

    Args:
        image_dir: Directory with multi-channel images
        label_dir: Directory with label masks
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        num_workers: Workers for data loading
        target_size: Target image size
        max_samples: Max samples to use. None uses all.
        seed: Random seed for reproducibility
        stats_save_dir: Directory to save/load stats. None skips saving.

    Returns:
        Tuple of (train_loader, val_loader, mean, std)
    """
    # Check if statistics already exist
    mean_path = None
    std_path = None
    mean = None
    std = None

    if stats_save_dir is not None:
        os.makedirs(stats_save_dir, exist_ok=True)
        mean_path = os.path.join(stats_save_dir, "dataset_mean.npy")
        std_path = os.path.join(stats_save_dir, "dataset_std.npy")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            print("Loading existing dataset statistics...")
            mean = np.load(mean_path)
            std = np.load(std_path)
            print(f"Mean per channel: {mean}")
            print(f"Std per channel: {std}")

    # Calculate statistics if not loaded, or if we are doing a new filtering
    if mean is None or std is None:
        print("Computing dataset statistics...")
        mean, std = calculate_dataset_statistics(
            image_dir, input_file_type, debug
        )

        # Save statistics if directory provided
        if stats_save_dir is not None:
            np.save(mean_path, mean)
            np.save(std_path, std)
            print(f"✓ Saved statistics to {stats_save_dir}")

    # Create full dataset, normalize using mean/std of loaded data
    full_dataset = LunarCraterDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        mean=mean,
        std=std,
        target_size=target_size,
        max_samples=max_samples,
        input_file_type=input_file_type,
        label_file_type=label_file_type,
        band_indices_filter=band_indices_filter,
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create val dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader, mean, std
