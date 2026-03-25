"""
dataset.py
Dataset class for DINO LoRA fine-tuning on lunar crater detection.
Supports images with any number of channels (grayscale, RGB, multispectral, etc.).
"""

import os
from pathlib import Path
from glob import glob
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import rasterio


class LunarCraterDatasetMask2Former(Dataset):
    """
    Dataset for multi-channel images and segmentation masks.
    Supports images with any number of channels (1=grayscale, 3=RGB, 4=RGBA, etc.).

    Args:
        image_dir: Directory with .npy images (H, W, C) or (C, H, W)
        label_dir: Directory with .npy label masks (H, W)
        mean: Mean values per channel for normalization (shape: [num_channels])
        std: Standard deviation per channel for normalization (shape: [num_channels])
        target_size: Target size for model input. Default: (304, 304)
        max_samples: Max samples to use. None uses all samples.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        image_processor,
        target_size: Tuple[int, int] = (304, 304),
        max_samples: Optional[int] = None,
        input_file_type: str = ".npy",
        label_file_type: str = ".npz",
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # Used for m2f
        self.image_processor = image_processor

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
        print(f"Dataset configured for {self.num_channels} channel(s)")

        if len(self.valid_image_paths) == 0:
            raise ValueError(
                "No matching image-label pairs found! "
                "Check that basenames match between image_dir and label_dir"
            )

    @staticmethod
    def _min_max_scale_bands(img: np.array):
        """Min-max scale each band to [0, 1]"""
        scaled = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[0]):
            band = img[i]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                scaled[i] = (band - band_min) / (band_max - band_min)
            else:
                scaled[i] = band
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

        seg_data = np.load(label_path)
        instance_mask = seg_data["mask"].astype(
            np.int8
        )  # (H, W) with instance IDs

        # Used for postprocessing in m2f
        original_height, original_width = instance_mask.shape

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
            raise ValueError(f"Unexpected image dimensions: {image.shape}")

        # Verify channel count matches expected
        if image.shape[2] != self.num_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.num_channels}, "
                f"got {image.shape[2]} for {img_path}"
            )

        # .tif inputs have been saved as raw values; needs min/max
        if self.input_file_type == ".tif":
            image = LunarCraterDatasetMask2Former._min_max_scale_bands(image)

        # Normalize image with dataset statistics
        # Reshape mean and std to (1, 1, C) for broadcasting with (H, W, C)
        mean_reshaped = self.mean.reshape(1, 1, self.num_channels)
        std_reshaped = self.std.reshape(1, 1, self.num_channels)
        image = (image - mean_reshaped) / std_reshaped

        # Build instance_id_to_semantic_id mapping
        # Get unique instance IDs (excluding background=0)
        unique_instances = np.unique(instance_mask)
        instance_to_semantic = {
            int(inst_id): (0 if inst_id == 0 else 1)
            for inst_id in unique_instances
        }

        # Apply image processor (handles resizing and format conversion)
        inputs = self.image_processor(
            images=[image],
            segmentation_maps=[instance_mask],
            instance_id_to_semantic_id=instance_to_semantic,
            return_tensors="pt",
            input_data_format="channels_last",  # ADD THIS: your image is (H, W, C)
            data_format="channels_first",  # ADD THIS: output should be (C, H, W)
        )

        # Return in Mask2Former format
        return {
            "pixel_values": inputs.pixel_values[0],  # (C, H, W)
            "mask_labels": inputs.mask_labels[0],  # (num_instances, H, W)
            "class_labels": inputs.class_labels[0],  # (num_instances,)
            "original_size": (original_height, original_width),  # Tuple
        }


def collate_fn(batch):
    """
    Custom collate function for Mask2Former.

    Handles variable number of instances per image by keeping
    mask_labels and class_labels as lists instead of stacking.
    """
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],  # List!
        "class_labels": [item["class_labels"] for item in batch],  # List!
        "original_size": [item["original_size"] for item in batch],  # List!
    }


def calculate_dataset_statistics(
    image_dir: str, input_file_type: str, debug: bool = False
):
    """
    Calculate mean and standard deviation for a dataset of .npy images.
    Automatically detects the number of channels from the first valid image.

    Args:
        image_dir (str): Directory containing .npy image files
        input_file_type (str): File extension (.npy, .npz, or .tif)
        debug (bool): If True, print detailed debugging information

    Returns:
        mean (np.ndarray): Mean per channel (shape: [num_channels])
        std (np.ndarray): Standard deviation per channel (shape: [num_channels])
    """
    if input_file_type not in [".npy", ".npz", ".tif"]:
        raise ValueError(
            "Calculating dataset statistics expects .npy, .npz, or .tif "
            "filetypes."
        )

    input_paths = glob(os.path.join(image_dir, f"*{input_file_type}"))

    if len(input_paths) == 0:
        raise ValueError(f"No .npy files found in {image_dir}")

    pixel_sum = None
    pixel_sq_sum = None
    pixel_count = 0
    valid_images = 0
    num_channels = None

    # Debug tracking
    if debug:
        band_inf_counts = None
        band_nan_counts = None
        band_neg_inf_counts = None
        band_min_vals = None
        band_max_vals = None

    for image_path in tqdm(input_paths, desc="Computing dataset statistics"):
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            continue

        try:
            if input_file_type in [".npy", ".npz"]:
                img = np.load(image_path).astype(np.float64)
            else:  # .tif
                img = rasterio.open(image_path).read()

            # Handle different shapes: (H, W, C), (C, H, W), or (H, W)
            if img.ndim == 3:
                # Determine format: if first dimension is smallest, assume (C, H, W)
                if img.shape[0] < min(img.shape[1], img.shape[2]):
                    img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
                # Otherwise, assume already in (H, W, C) format

            elif img.ndim == 2:
                # Handle grayscale images without explicit channel dimension
                img = img[:, :, np.newaxis]  # Shape: (H, W, 1)

            else:
                print(
                    f"Warning: Image has {img.ndim} dimensions, "
                    f"expected 2 or 3. Skipping {image_path}"
                )
                continue

            # Initialize statistics arrays on first valid image
            if valid_images == 0:
                num_channels = img.shape[2]
                pixel_sum = np.zeros(num_channels)
                pixel_sq_sum = np.zeros(num_channels)
                print(f"First image shape: {img.shape}, dtype: {img.dtype}")
                print(f"Detected {num_channels} channel(s)")
                print(f"Value range: min={img.min()}, max={img.max()}")

                if debug:
                    band_inf_counts = np.zeros(num_channels)
                    band_nan_counts = np.zeros(num_channels)
                    band_neg_inf_counts = np.zeros(num_channels)
                    band_min_vals = np.full(num_channels, np.inf)
                    band_max_vals = np.full(num_channels, -np.inf)

            # Verify consistent number of channels
            if img.shape[2] != num_channels:
                print(
                    f"Warning: Inconsistent channels. Expected {num_channels}, "
                    f"got {img.shape[2]} for {image_path}. Skipping."
                )
                continue

            # Debug: Check for problematic values per band
            if debug:
                for band_idx in range(num_channels):
                    band_data = img[:, :, band_idx]

                    # Count special values
                    n_inf = np.isinf(band_data) & (band_data > 0)
                    n_neg_inf = np.isinf(band_data) & (band_data < 0)
                    n_nan = np.isnan(band_data)

                    band_inf_counts[band_idx] += n_inf.sum()
                    band_neg_inf_counts[band_idx] += n_neg_inf.sum()
                    band_nan_counts[band_idx] += n_nan.sum()

                    # Track min/max (excluding inf/nan)
                    finite_mask = np.isfinite(band_data)
                    if finite_mask.any():
                        band_min_vals[band_idx] = min(
                            band_min_vals[band_idx],
                            band_data[finite_mask].min(),
                        )
                        band_max_vals[band_idx] = max(
                            band_max_vals[band_idx],
                            band_data[finite_mask].max(),
                        )

                    # Report issues immediately if found
                    if (
                        n_inf.sum() > 0
                        or n_neg_inf.sum() > 0
                        or n_nan.sum() > 0
                    ):
                        print(f"\n⚠️  File: {os.path.basename(image_path)}")
                        print(
                            f"    Band {band_idx}: +inf={n_inf.sum()}, "
                            f"-inf={n_neg_inf.sum()}, nan={n_nan.sum()}"
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
        raise ValueError(
            "No valid images were processed. Check your .npy files."
        )

    # Calculate mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

    print(
        f"\nProcessed {valid_images} valid images out of {len(input_paths)} total"
    )
    print(f"Mean per channel: {mean}")
    print(f"Std per channel: {std}")

    # Debug summary
    if debug:
        print("\n" + "=" * 70)
        print("DEBUG SUMMARY - Per Band Statistics:")
        print("=" * 70)
        for band_idx in range(num_channels):
            print(f"\nBand {band_idx}:")
            print(f"  Total +inf values: {int(band_inf_counts[band_idx])}")
            print(f"  Total -inf values: {int(band_neg_inf_counts[band_idx])}")
            print(f"  Total nan values:  {int(band_nan_counts[band_idx])}")
            print(f"  Min (finite):      {band_min_vals[band_idx]:.6f}")
            print(f"  Max (finite):      {band_max_vals[band_idx]:.6f}")
            print(f"  Calculated mean:   {mean[band_idx]:.6f}")
            print(f"  Calculated std:    {std[band_idx]:.6f}")
            print(f"  pixel_sum:         {pixel_sum[band_idx]:.6f}")
            print(f"  pixel_sq_sum:      {pixel_sq_sum[band_idx]:.6f}")

            # Diagnose issues
            if np.isinf(mean[band_idx]) or np.isnan(mean[band_idx]):
                print(f"  ⚠️  ISSUE DETECTED!")
                if band_neg_inf_counts[band_idx] > 0:
                    print(
                        f"     → Found {int(band_neg_inf_counts[band_idx])} -inf values"
                    )
                if band_inf_counts[band_idx] > 0:
                    print(
                        f"     → Found {int(band_inf_counts[band_idx])} +inf values"
                    )
                if np.isinf(pixel_sum[band_idx]):
                    print(f"     → pixel_sum is infinite!")

            if np.isnan(std[band_idx]):
                print(f"  ⚠️  STD IS NAN!")
                variance = (pixel_sq_sum[band_idx] / pixel_count) - (
                    mean[band_idx] ** 2
                )
                print(f"     → Variance = {variance:.6f}")
                if variance < 0:
                    print(
                        f"     → Negative variance (numerical precision issue)"
                    )
        print("=" * 70)

    return mean, std


def get_dataloaders(
    image_dir: str,
    label_dir: str,
    image_processor,
    batch_size: int = 8,
    train_split: float = 0.8,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (304, 304),
    max_samples: Optional[int] = None,
    seed: int = 42,
    stats_save_dir: Optional[str] = None,
    input_file_type: str = ".npy",
    label_file_type: str = ".npy",
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

    # Calculate statistics if not loaded
    if mean is None or std is None:
        print("Computing dataset statistics...")
        mean, std = calculate_dataset_statistics(image_dir, input_file_type)

        # Save statistics if directory provided
        if stats_save_dir is not None:
            np.save(mean_path, mean)
            np.save(std_path, std)
            print(f"✓ Saved statistics to {stats_save_dir}")

    # Create full dataset, normalize using mean/std of loaded data
    full_dataset = LunarCraterDatasetMask2Former(
        image_dir=image_dir,
        label_dir=label_dir,
        mean=mean,
        std=std,
        image_processor=image_processor,
        target_size=target_size,
        max_samples=max_samples,
        input_file_type=input_file_type,
        label_file_type=label_file_type,
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

    sample = train_dataset[0]
    print(f"\nDataset Sample Check:")
    print(
        f"  pixel_values shape: {sample['pixel_values'].shape}"
    )  # Should be [5, H, W]
    print(
        f"  mask_labels shape: {sample['mask_labels'].shape}"
    )  # Should be [num_instances, H, W]
    print(
        f"  class_labels shape: {sample['class_labels'].shape}"
    )  # Should be [num_instances]
    print(
        f"  Number of channels: {sample['pixel_values'].shape[0]}"
    )  # Should be 5

    # Create train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create val dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader, mean, std
