"""
dataset.py
Dataset class for DINO LoRA fine-tuning on lunar crater detection.
Supports images with any number of channels (grayscale, RGB, multispectral, etc.).
"""

import os
from pathlib import Path
from glob import glob
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm


def calculate_dataset_statistics(image_dir: str):
    """
    Calculate mean and standard deviation for a dataset of .npy images.
    Automatically detects the number of channels from the first valid image.

    Args:
        image_dir (str): Directory containing .npy image files

    Returns:
        mean (np.ndarray): Mean per channel (shape: [num_channels])
        std (np.ndarray): Standard deviation per channel (shape: [num_channels])
    """
    npy_paths = glob(os.path.join(image_dir, "*.npy"))

    if len(npy_paths) == 0:
        raise ValueError(f"No .npy files found in {image_dir}")

    pixel_sum = None
    pixel_sq_sum = None
    pixel_count = 0
    valid_images = 0
    num_channels = None

    for npy_path in tqdm(npy_paths, desc="Computing dataset statistics"):
        if not os.path.exists(npy_path):
            print(f"Warning: File not found: {npy_path}")
            continue

        try:
            img = np.load(npy_path).astype(np.float64)

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
                    f"expected 2 or 3. Skipping {npy_path}"
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

            # Verify consistent number of channels
            if img.shape[2] != num_channels:
                print(
                    f"Warning: Inconsistent channels. Expected {num_channels}, "
                    f"got {img.shape[2]} for {npy_path}. Skipping."
                )
                continue

            # Accumulate statistics
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img**2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]
            valid_images += 1

        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue

    if valid_images == 0:
        raise ValueError(
            "No valid images were processed. Check your .npy files."
        )

    # Calculate mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

    print(
        f"Processed {valid_images} valid images out of {len(npy_paths)} total"
    )
    print(f"Mean per channel: {mean}")
    print(f"Std per channel: {std}")

    return mean, std


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
        norm_to_one: bool = False,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # If we want to normalize instances to 1
        self.norm_to_one = norm_to_one

        # Used for m2f
        self.image_processor = image_processor

        # Store mean and std as float32 for consistency
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.num_channels = len(mean)

        # Glob all images and labels
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))
        label_paths = sorted(glob(os.path.join(label_dir, "*.npz")))

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

        image = np.load(img_path).astype(np.float32)  # (H, W, C) or (C, H, W)
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

        # Normalize image with dataset statistics
        # Reshape mean and std to (1, 1, C) for broadcasting with (H, W, C)
        mean_reshaped = self.mean.reshape(1, 1, self.num_channels)
        std_reshaped = self.std.reshape(1, 1, self.num_channels)
        image = (image - mean_reshaped) / std_reshaped

        # Build instance_id_to_semantic_id mapping
        # Get unique instance IDs (excluding background=0)
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]

        # Map each instance to its semantic class
        # For crater detection: all instances are class 1 (crater)
        instance_to_semantic = {}
        for inst_id in unique_instances:
            instance_to_semantic[int(inst_id)] = 1  # All craters are class 1

        # Apply image processor (handles resizing and format conversion)
        inputs = self.image_processor(
            images=[image],
            segmentation_maps=[instance_mask],
            instance_id_to_semantic_id=instance_to_semantic,
            return_tensors="pt",
        )

        # Return in Mask2Former format
        return {
            "pixel_values": inputs.pixel_values[0],  # (C, H, W)
            "mask_labels": inputs.mask_labels[0],  # (num_instances, H, W)
            "class_labels": inputs.class_labels[0],  # (num_instances,)
            "original_size": (original_height, original_width),  # Tuple
        }

        # (C, H, W) where C = num_channels
        # image = torch.from_numpy(image).permute(2, 0, 1)
        # (H, W)
        # instance_label = torch.from_numpy(instance_label)

        # Resize image to target size for model
        # image = F.interpolate(
        #     image.unsqueeze(0),
        #     size=self.target_size,
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze(
        # 0
        # )  # (C, target_H, target_W)

        # Resize label using nearest neighbor to preserve class indices
        # Labels are NOT transformed, only resized
        # label = (
        #     F.interpolate(
        #         label.unsqueeze(0).unsqueeze(0).float(),
        #         size=self.target_size,
        #         mode="nearest",
        #     )
        #     .squeeze(0)
        #     .squeeze(0)
        #     .long()
        # )  # (target_H, target_W)

        # Used in mask2former architecture
        # if self.norm_to_one:
        #     label[label > 0] = 1

        # return image, label


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
    norm_to_one: bool = False,
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
        mean, std = calculate_dataset_statistics(image_dir)

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
        norm_to_one=norm_to_one,
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
