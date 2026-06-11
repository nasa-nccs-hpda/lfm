# Standard library imports
from datetime import datetime
from glob import glob
from pathlib import Path
import math
import re

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import rasterio
from tiler import Tiler, Merger
import torch
from transformers import AutoImageProcessor
import xarray as xr
from tqdm import tqdm


import xarray as xr
import rioxarray as rxr
import rasterio
import re
from pathlib import Path
import os
from glob import glob


def verify_band_selection(filepath, band_indices, pattern, verbose=True):
    """
    Verify that selected bands actually match the pattern.

    Args:
        filepath: Path to raster file
        band_indices: List of 1-based band indices (from rasterio)
        pattern: Regex pattern that was used for matching
        verbose: Print verification details

    Returns:
        tuple: (all_match: bool, band_names: list, mismatches: list)
    """
    regex = re.compile(pattern)
    band_names = []
    mismatches = []

    with rasterio.open(filepath) as src:
        if verbose:
            print(f"\n  Verifying bands for {Path(filepath).name}:")
            print(f"  Pattern: '{pattern}'")
            print(f"  Selected band indices (1-based): {band_indices}")
            print(f"  Band verification:")

        for band_idx in band_indices:
            tags = src.tags(band_idx)
            band_name = tags.get('Name', '')
            band_names.append(band_name)

            matches = bool(regex.search(band_name))
            status = "✓" if matches else "✗"

            if verbose:
                print(f"    Band {band_idx}: {band_name} {status}")

            if not matches:
                mismatches.append((band_idx, band_name))

    all_match = len(mismatches) == 0

    if verbose:
        if all_match:
            print(f"  ✓ All {len(band_indices)} bands match pattern")
        else:
            print(f"  ✗ WARNING: {len(mismatches)} bands don't match pattern!")

    return all_match, band_names, mismatches


def check_bands_exist(filepath: Path, pattern: str) -> tuple:
    """
    Check if bands matching regex exist.

    Args:
        filepath: Path to raster file
        pattern: Regex pattern to match against band names

    Returns:
        (exists: bool, band_numbers: list, error_message: str or None)
    """
    band_indices = []
    try:
        regex = re.compile(pattern)
        with rasterio.open(filepath) as src:
            for band_idx in range(1, src.count + 1):
                tags = src.tags(band_idx)
                band_name = tags.get('Name', '')

                if regex.search(band_name):
                    band_indices.append(band_idx)

            if not band_indices:
                return False, None, f"No bands matching pattern '{pattern}' found"

            return True, band_indices, None

    except re.error as e:
        return False, None, f"Invalid regex pattern: {str(e)}"
    except Exception as e:
        return False, None, f"Error reading file: {str(e)}"


def filter_static_bands(static_cubes, verbose=True, verify=True):
    """
    Filter static cube bands to only include those matching 'lola_kaguya' pattern.

    Args:
        static_cubes: List of file paths to static cube GeoTIFFs
        verbose: Print filtering information
        verify: Verify band selection is correct

    Returns:
        List of xarray DataArrays with filtered bands
    """
    static_datasets = []
    STATIC_BAND_REGEX = r"lola_kaguya"

    for cube in static_cubes:
        bands_exist, band_indices, error_message = check_bands_exist(
            cube, STATIC_BAND_REGEX
        )

        if not bands_exist:
            if verbose:
                print(f"  ⚠ Skipping {Path(cube).name}: {error_message}")
            continue

        # Verify the band selection if requested
        if verify:
            all_match, band_names, mismatches = verify_band_selection(
                cube, band_indices, STATIC_BAND_REGEX, verbose=verbose
            )
            if not all_match:
                raise ValueError(
                    f"Band verification failed for {cube}. "
                    f"Mismatches: {mismatches}"
                )

        # Convert from 1-based (rasterio) to 0-based (xarray) indexing
        band_indices_0based = [idx - 1 for idx in band_indices]

        if verbose:
            print(f"  Converting indices: {band_indices} (rasterio) -> {band_indices_0based} (xarray)")

        # Open with rioxarray and select matching bands
        ds = rxr.open_rasterio(cube)

        if verbose:
            print(f"  Total bands in file: {ds.sizes['band']}")
            print(f"  Selecting bands at 0-based indices: {band_indices_0based}")

        ds_filtered = ds.isel(band=band_indices_0based)

        if verbose:
            print(f"  ✓ Result: {ds_filtered.sizes['band']} bands selected\n")

        static_datasets.append(ds_filtered)

    return static_datasets


def group_cubes_by_tile(datacubes: list) -> tuple:
    """
    Group datacubes by tile ID.

    Args:
        datacubes: List of datacube file paths

    Returns:
        (cubes_by_tile: dict, ltm_dict: dict)
    """
    import re

    # Extract cubes by modality
    wac_datacubes = [cube for cube in datacubes if "Static" not in cube]
    static_datacubes = [cube for cube in datacubes if "Static" in cube]

    # Get LTM zones
    ltm_pattern = r"Cube-LTM[0-9]+[NS]"
    ltm_matches = [
        re.search(ltm_pattern, cube).group() for cube in datacubes
        if re.search(ltm_pattern, cube)
    ]
    ltm_unique = set(ltm_matches)
    ltm_dict = {
        "all_zones": ltm_matches,
        "unique": ltm_unique,
    }

    # Get tile indices
    tile_pattern = r"_Tile-[0-9]+-[0-9]+"
    tile_matches = [
        re.search(tile_pattern, cube).group() for cube in datacubes
        if re.search(tile_pattern, cube)
    ]

    # Get unique tile indices
    unique_tiles = set(tile_matches)
    tile_ids = sorted([match.replace("_Tile-", "") for match in unique_tiles])

    # Group cubes by tile
    cubes_by_tile = {
        tile_id: {
            "wac": [f for f in wac_datacubes if tile_id in f][0],
            "static": [f for f in static_datacubes if tile_id in f][0]
        }
        for tile_id in tile_ids
    }

    return cubes_by_tile, ltm_dict


def get_datacube_data(
    input_paths,
    band_filter=None,
    max_images=None,
    verbose=True,
    verify_bands=True,
):
    """
    Extract and stack vis and static GeoTIFF data.

    Args:
        input_paths: Path(s) to directory or file(s)
        band_filter: Not currently used
        max_images: Maximum number of image pairs to process
        verbose: Print progress information
        verify_bands: Verify that band filtering worked correctly

    Returns:
        tuple: (stacked_data, file_pairs)
            - stacked_data: numpy array of shape (N, bands, H, W)
            - file_pairs: List of (vis_file, static_file) tuples
    """
    # Convert input to list of file paths
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    file_paths = []
    for path in input_paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() == ".tif":
            file_paths.append(str(path))
        elif path.is_dir():
            pattern = str(path / "**/*.tif")
            if verbose:
                print(f"Searching with pattern: {pattern}")
            found = glob(pattern, recursive=True)
            if verbose:
                print(f"Found {len(found)} files")
            file_paths.extend(found)

    if not file_paths:
        raise ValueError(f"No .tif files found in {input_paths}")

    file_paths = sorted(file_paths)

    cubes_by_tile, _ = group_cubes_by_tile(file_paths)
    wac_cubes = [data_dict['wac'] for tile_id, data_dict in cubes_by_tile.items()]
    static_cubes = [data_dict['static'] for tile_id, data_dict in cubes_by_tile.items()]
    # Create file pairs
    file_pairs = [
        (cubes_by_tile[tid]['wac'], cubes_by_tile[tid]['static'])
        for tid in cubes_by_tile
    ]

    if verbose:
        print(
            f"\nFound {len(file_paths)} total .tif files. "
            f"Wac: {len(wac_cubes)}, Static: {len(static_cubes)}"
        )

    # Process each pair
    all_datasets = []

    # We have already extracted only a single WAC/STATIC file per tile ID
    for tile_id, dataset_dict in cubes_by_tile.items():
        wac_file = dataset_dict['wac']
        static_file = dataset_dict['static']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing tile_id: {tile_id}")
            print(f"  Wac: {wac_file}")
            print(f"  Static: {static_file}")

        # Load wac data, reorder to training order
        wac_ds = rxr.open_rasterio(wac_file)
        vis_uv_order = [2, 3, 4, 5, 6, 0, 1]
        wac_ds = wac_ds.isel(band=vis_uv_order)
        if verbose:
            print(f"  Wac bands: {wac_ds.sizes['band']}")

        # Load and filter static data
        static_datasets = filter_static_bands(
            [static_file],
            verbose=verbose,
            verify=verify_bands
        )

        if not static_datasets:
            if verbose:
                print(f"  ⚠ Skipping pair - no static bands matched")
            continue

        static_ds = static_datasets[0]

        # Combine wac and static
        combined_ds = xr.concat([wac_ds, static_ds], dim='band')
        # clipped_combined_ds = np.clip(combined_ds.values, 0, 1)

        if verbose:
            print(f"  Combined shape: {combined_ds.shape}")
            print(
                f"  Combined bands: wac({wac_ds.sizes['band']}) "
                f"+ static({static_ds.sizes['band']}) = {combined_ds.sizes['band']}"
            )
            print(f"  Combined min/max: {np.min(combined_ds.values), np.max(combined_ds.values)}")

        # Convert to numpy and append
        if not np.any(combined_ds.values == combined_ds.rio.nodata):
            all_datasets.append(combined_ds)
        elif verbose:
            print(f"  ⚠ Skipping - contains nodata (value: {combined_ds.rio.nodata})")

    # Apply max_images limit AFTER processing (not before)
    if max_images is not None:
        all_datasets = all_datasets[:max_images]
        file_pairs = file_pairs[:max_images]

    if verbose:
        print(f"\n{'='*60}")
        if all_datasets:
            # FIX: Use numpy array properties instead of xarray
            total_bands = sum(ds.shape[0] for ds in all_datasets)  # Changed from .sizes['band']
            print(
                f"✓ Extraction complete: {len(all_datasets)} combined datacubes, "
                f"{total_bands} total bands across all cubes"
            )
            print(f"  Example shape: {all_datasets[0].shape}")
        else:
            print("⚠️ No datasets were extracted")

    # Convert to numpy array with shape (N, bands, H, W)
    if all_datasets:
        return np.array(all_datasets), file_pairs
    else:
        return np.array([]), file_pairs


def min_max_scale_bands(bands):
    """Min-max scale each band to [0, 1]"""
    print(bands.shape)
    scaled = np.zeros_like(bands, dtype=np.float32)
    for i in range(bands.shape[-1]):
        band = bands[:, :, i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            scaled[:, :, i] = (band - band_min) / (band_max - band_min)
        else:
            scaled[:, :, i] = band
    return scaled


def create_binary_colormap(instance_mask):
    """
    Create a simple binary colormap: 0 -> black, any other value -> red.

    Args:
        instance_mask: (H, W) array with instance IDs (0 = background)

    Returns:
        colored: (H, W, 3) RGB array with values in [0, 1]
    """
    h, w = instance_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    # Set all non-zero pixels to red
    mask = instance_mask != 0
    colored[mask] = [1.0, 0.0, 0.0]  # Pure red

    return colored


def sliding_window_inference(
    images_npy, model, target_size=304, device="cuda", threshold=0.5, n_channels=12
):
    """
    Perform sliding window inference on large images.
    Merges probabilities then thresholds for smoother results.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    model.eval()

    # Handle single image
    single_image = False
    if images_npy.ndim == 3:
        images_npy = images_npy[np.newaxis, ...]
        single_image = True

    all_predictions = []
    all_probabilities = []  # Optional: keep probabilities too

    # Process each image
    for idx in range(images_npy.shape[0]):
        print(f"\nProcessing image {idx + 1}/{images_npy.shape[0]}")
        image = images_npy[idx]  # Shape: (512, 512, n_channels)

        # Create INPUT tiler
        input_tiler = Tiler(
            data_shape=image.shape,  # (512, 512, n_channels)
            tile_shape=(target_size[0], target_size[1], n_channels),
            channel_dimension=-1,
            mode="reflect",
        )

        # Create OUTPUT tiler for SINGLE channel (crater probabilities only)
        output_tiler = Tiler(
            data_shape=(image.shape[0], image.shape[1], 1),  # (512, 512, 1)
            tile_shape=(target_size[0], target_size[1], 1),  # (304, 304, 1)
            channel_dimension=-1,
            mode="reflect",
        )

        # Create merger based on OUTPUT tiler
        output_merger = Merger(tiler=output_tiler, window="triang")

        print(f"Number of tiles: {len(input_tiler)}")

        # Iterate through tiles
        with torch.no_grad():
            for tile_id, tile_batch in input_tiler(
                image, batch_size=1, progress_bar=True
            ):
                # tile_batch shape: (1, 304, 304, n_channels)
                tile = tile_batch[0]  # (304, 304, n_channels)

                # Convert to torch tensor and change to channels-first
                tile_tensor = torch.from_numpy(tile).permute(
                    2, 0, 1
                )  # (3, 304, 304)
                tile_tensor = (
                    tile_tensor.unsqueeze(0).float().to(device)
                )  # (1, n_channels, 304, 304)

                # Run inference
                logits = model(tile_tensor)  # Shape: (1, 2, 304, 304)

                # Convert to probabilities (same as training eval)
                probs = torch.sigmoid(logits)  # (1, 2, 304, 304)

                # Get crater class only (class 1)
                if probs.shape[1] == 2:
                    probs = probs[:, 1:2]  # (1, 1, 304, 304) - crater class

                # Convert to numpy and channels-last
                probs_np = probs.cpu().numpy()
                probs_np = np.transpose(
                    probs_np[0], (1, 2, 0)
                )  # (304, 304, 1)

                # Add PROBABILITIES to merger (not thresholded yet!)
                actual_tile_id = tile_id * 1  # Since batch_size=1
                output_merger.add(actual_tile_id, probs_np)

        # Merge all tiles - this gives smooth probability map
        merged_probs = output_merger.merge(unpad=True)  # (512, 512, 1)
        print(f"Merged probabilities shape: {merged_probs.shape}")

        # Threshold to get binary predictions (same as training eval)
        merged_preds = (merged_probs > threshold).astype(
            np.float32
        )  # (512, 512, 1)
        merged_preds = merged_preds.squeeze(
            -1
        )  # (512, 512) - squeeze channel dim

        all_predictions.append(merged_preds)
        all_probabilities.append(
            merged_probs.squeeze(-1)
        )  # Optional: keep probs

        print(f"Final prediction shape: {merged_preds.shape}")
        print(
            f"Prediction range: [{merged_preds.min():.2f}, {merged_preds.max():.2f}]"
        )

    predictions = np.stack(all_predictions, axis=0)
    probabilities = np.stack(all_probabilities, axis=0)

    if single_image:
        predictions = predictions[0]
        probabilities = probabilities[0]

    return predictions, probabilities  # Return both


def calculate_datacube_statistics(
    datacube_dir,
    max_samples=None,
    verbose=True
):
    """
    Calculate mean/std from datacube .tif files.
    Applies same preprocessing as inference: min-max scale then reorder.

    Returns mean/std in TRAINING order [vis, UV, static] with shape (12,)
    """
    print("Calculating statistics from datacube files...")

    # Load all datacubes
    images_raw, _ = get_datacube_data(
        input_paths=datacube_dir,
        max_images=max_samples,
        verbose=verbose,
    )

    print(f"Loaded {len(images_raw)} datacubes")
    print(f"Raw shape: {images_raw.shape}")  # (N, 12, H, W)

    # Transpose to (N, H, W, 12)
    images_transposed = np.transpose(images_raw, (0, 2, 3, 1))

    # Min-max scale each image
    print("Applying min-max scaling...")
    images_scaled = np.zeros_like(images_transposed, dtype=np.float32)
    for i in tqdm(range(len(images_transposed)), desc="Scaling"):
        images_scaled[i] = min_max_scale_bands(images_transposed[i])

    print(f"After scaling shape: {images_scaled.shape}")  # (N, H, W, 12)

    # Reorder bands: datacube [UV, vis, static] -> training [vis, UV, static]
    print("Reordering bands to training order...")
    datacube_to_training = [
        2, 3, 4, 5, 6,  # vis
        0, 1,            # UV
        7, 8, 9, 10, 11  # static
    ]
    images_reordered = images_scaled[:, :, :, datacube_to_training]
    print(f"After reordering shape: {images_reordered.shape}")  # (N, H, W, 12)

    # Calculate statistics using the same method as calculate_dataset_statistics
    print("Calculating mean and std per channel...")

    num_channels = 12
    pixel_sum = np.zeros(num_channels, dtype=np.float64)
    pixel_sq_sum = np.zeros(num_channels, dtype=np.float64)
    pixel_count = 0

    for img in tqdm(images_reordered, desc="Computing statistics"):
        # img shape: (H, W, 12)
        pixel_sum += img.sum(axis=(0, 1))  # Sum over H, W -> shape (12,)
        pixel_sq_sum += (img.astype(np.float64)**2).sum(axis=(0, 1))  # shape (12,)
        pixel_count += img.shape[0] * img.shape[1]  # H * W

    # Calculate mean and std
    mean = pixel_sum / pixel_count  # shape (12,)
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))  # shape (12,)

    print(f"\nMean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    print("\n" + "="*60)
    print("DATACUBE STATISTICS (in training order)")
    print("="*60)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("\nPer-band breakdown:")
    for i in range(num_channels):
        band_type = (
            "vis" if i < 5 else
            "UV" if i < 7 else
            "static"
        )
        print(f"  Band {i:2d} ({band_type:6s}): mean={mean[i]:.4f}, std={std[i]:.4f}")
    print("="*60)

    # Sanity checks
    print("\nSANITY CHECKS:")
    if not (0 <= mean.min() and mean.max() <= 1):
        print(f"⚠️  WARNING: Mean values outside [0,1] range! min={mean.min():.4f}, max={mean.max():.4f}")
    else:
        print(f"✓ Mean values in valid range: [{mean.min():.4f}, {mean.max():.4f}]")

    if (std < 0.01).any():
        print("⚠️  WARNING: Some bands have very low std (nearly constant)!")
        for i, s in enumerate(std):
            if s < 0.01:
                print(f"    Band {i}: std={s:.6f}")
    else:
        print(f"✓ Std values look reasonable: [{std.min():.4f}, {std.max():.4f}]")

    # Compare to provided mean/std if available
    print("\n" + "="*60)

    return mean, std


def run_datacube_inference(
    model,
    device,
    input_dir,
    mean,
    std,
    output_path="inference_test.png",
    n_images=20,
    model_native_size=304,
    tile_overlap=0.25,
    threshold=0.75,
    save_inputs_dir=None,
):
    """
    Run inference on 12-band datacubes (WAC + static).

    Processing pipeline:
    1. Load raw .tif datacubes (N, 12, H, W)
    2. Transpose to (N, H, W, 12)
    3. Apply min-max scaling per band
    4. **Reorder bands from datacube order to training order**
    5. Normalize with mean/std (in training order)
    6. Run inference
    """
    model.eval()

    print(f"Loading up to {n_images} datacubes from TIFF files...")

    # Load raw datacubes: (N, 12, H, W) - channels first, unnormalized
    images_raw, file_paths = get_datacube_data(
        input_paths=input_dir,
        max_images=n_images,
        verbose=False,
    )

    print(f"Raw datacubes shape: {images_raw.shape}")  # (N, 12, H, W)

    # Transpose to (N, H, W, 12) for band-wise processing
    images_transposed = np.transpose(images_raw, (0, 2, 3, 1))
    print(f"Transposed to: {images_transposed.shape}")  # (N, 512, 512, 12)
    return images_transposed, None

    # ============================================
    # STEP 1: Min-max scale each band to [0, 1]
    # ============================================
    print("\nApplying min-max scaling per band...")
    images_scaled = np.zeros_like(images_transposed, dtype=np.float32)
    for i in range(len(images_transposed)):
        images_scaled[i] = min_max_scale_bands(images_transposed[i])

    print(f"After scaling: min={images_scaled.min():.3f}, max={images_scaled.max():.3f}")

    # ============================================
    # STEP 3: Normalize with training mean/std
    # ============================================
    n_channels = 12

    if mean.shape[0] != n_channels or std.shape[0] != n_channels:
        raise ValueError(
            f"Mean/std must have {n_channels} values (one per band). "
            f"Got mean: {mean.shape[0]}, std: {std.shape[0]}"
        )

    print(f"Applying normalization with training statistics:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Mean (first 3 bands): {mean[:3]}")
    print(f"  Std (first 3 bands): {std[:3]}")

    mean_reshaped = mean.reshape(1, 1, 1, n_channels)
    std_reshaped = std.reshape(1, 1, 1, n_channels)

    images_npy = (images_scaled - mean_reshaped) / (std_reshaped + 1e-8)

    print(f"\nAfter normalization:")
    print(f"  Shape: {images_npy.shape}")
    print(f"  Range: [{images_npy.min():.3f}, {images_npy.max():.3f}]")
    print(f"  Mean per channel: {images_npy.mean(axis=(0,1,2))}")
    print(f"  Std per channel: {images_npy.std(axis=(0,1,2))}")

    # ============================================
    # STEP 4: Run sliding window inference
    # ============================================
    print(f"\n{'='*60}")
    print(f"Running inference:")
    print(f"  Input images: {images_npy.shape[1]}x{images_npy.shape[2]}")
    print(f"  Channels: {n_channels} (reordered to training order)")
    print(f"  Model native resolution: {model_native_size}x{model_native_size}")
    print(f"  Tile overlap: {tile_overlap}")
    print(f"  Threshold: {threshold}")
    print(f"{'='*60}\n")

    preds_list, probabilities = sliding_window_inference(
        images_npy,
        model,
        target_size=model_native_size,
        device=device,
        threshold=threshold,
    )

    print(f"\nGot {len(preds_list)} predictions")
    print(f"Output mask shapes: {[p.shape for p in preds_list[:3]]}")

    # ============================================
    # STEP 5: Create visualization
    # ============================================
    print("\nCreating visualization...")

    batch_size = min(len(images_npy), 10)
    fig, axes = plt.subplots(2, batch_size, figsize=(5 * batch_size, 10))

    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    # Extract filenames from tuples
    display_filenames = [
        Path(wac_file).stem for wac_file, static_file in file_paths[:batch_size]
    ]

    for i in range(batch_size):
        # Use reordered scaled images for visualization (not normalized)
        img = images_npy[i]  # (H, W, 12) in [0, 1], training order
        pred_mask = preds_list[i]  # (H, W)

        # ============================================
        # Create RGB composite for visualization
        # Use first 3 vis bands (now at indices 0, 1, 2 after reordering)
        # ============================================
        img_vis = img[:, :, 0]  # First 3 vis bands as RGB

        # Row 0: Original image
        axes[0, i].imshow(img_vis)
        axes[0, i].set_title(
            f"{display_filenames[i]}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0, i].axis("off")

        # Row 1: Prediction
        pred_colored = create_binary_colormap(pred_mask)
        axes[1, i].imshow(pred_colored, vmin=0, vmax=1)
        axes[1, i].set_title(f"Inference", fontsize=11)
        axes[1, i].axis("off")

    fig.suptitle(
        f"Inference on {images_npy.shape[1]}×{images_npy.shape[2]} Datacubes\n"
        f"12 channels (reordered: vis+UV+static) | "
        f"Model: {model_native_size}×{model_native_size} | "
        f"Threshold: {threshold}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")

    model.train()

    return images_npy, preds_list

