"""
dataset.py
Dataset class for DINO LoRA fine-tuning on lunar crater detection.
"""

import os
from pathlib import Path
from glob import glob
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm


def calculate_dataset_statistics(image_dir: str):
    """
    Calculate mean and standard deviation for a dataset of .npy images.

    Args:
        image_dir (str): Directory containing .npy image files

    Returns:
        mean (np.ndarray): Mean per channel (RGB)
        std (np.ndarray): Standard deviation per channel (RGB)
    """
    npy_paths = glob(os.path.join(image_dir, "*.npy"))

    if len(npy_paths) == 0:
        raise ValueError(f"No .npy files found in {image_dir}")

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0
    valid_images = 0

    for npy_path in tqdm(npy_paths, desc="Computing dataset statistics"):
        if not os.path.exists(npy_path):
            print(f"Warning: File not found: {npy_path}")
            continue

        try:
            img = np.load(npy_path).astype(np.float64)

            # Debug: Check the first image
            if valid_images == 0:
                print(f"First image shape: {img.shape}, dtype: {img.dtype}")
                print(f"Value range: min={img.min()}, max={img.max()}")

            # Handle different shapes: (H, W, C) or (C, H, W)
            if img.ndim == 3:
                if img.shape[0] == 3:  # (C, H, W) format
                    img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
                elif img.shape[2] != 3:
                    print(
                        f"Warning: Unexpected shape {img.shape} for {npy_path}"
                    )
                    continue
            else:
                print(
                    f"Warning: Image has {img.ndim} dimensions, "
                    f"expected 3. Skipping {npy_path}"
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
    print(f"Mean (RGB): {mean}")
    print(f"Std (RGB): {std}")

    return mean, std


class LunarCraterDataset(Dataset):
    """
    Dataset for RGB images and segmentation masks.

    Args:
        image_dir: Directory with .npy RGB images (300, 300, 3)
        label_dir: Directory with .npy label masks (300, 300)
        mean: Mean values for RGB normalization
        std: Standard deviation for RGB normalization
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
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # Store mean and std as float32 for consistency
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        # Glob all images and labels
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))
        label_paths = sorted(glob(os.path.join(label_dir, "*.npy")))

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

        image = np.load(img_path).astype(
            np.float32
        )  # (300, 300, 3) or (3, 300, 300)
        label = np.load(label_path).astype(np.int64)  # (300, 300)

        # Handle different image shapes: (H, W, C) or (C, H, W)
        if image.shape[0] == 3:  # (C, H, W) format
            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)

        # Now image is guaranteed to be (H, W, C) format
        # Normalize image with dataset statistics
        # Reshape mean and std to (1, 1, 3) for broadcasting with (H, W, 3)
        mean_reshaped = self.mean.reshape(1, 1, 3)
        std_reshaped = self.std.reshape(1, 1, 3)
        image = (image - mean_reshaped) / std_reshaped

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, 300, 300)
        label = torch.from_numpy(label)  # (300, 300)

        # Resize image to target size for model
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # (3, 304, 304)

        # Resize label using nearest neighbor to preserve class indices
        # Labels are NOT transformed, only resized
        label = (
            F.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(),
                size=self.target_size,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )  # (304, 304)

        return image, label


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
):
    """
    Create train/val dataloaders with automatic statistics calculation.
    Statistics are used to z-score normalize inputs upon dataloader creation.

    Args:
        image_dir: Directory with RGB images
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
            print(f"Mean (RGB): {mean}")
            print(f"Std (RGB): {std}")

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
    full_dataset = LunarCraterDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        mean=mean,
        std=std,
        target_size=target_size,
        max_samples=max_samples,
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
