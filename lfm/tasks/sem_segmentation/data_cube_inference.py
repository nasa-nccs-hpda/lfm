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

# ============================================================
# DATA LOADING
# ============================================================

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

    cubes_by_tile = {}

    for tile_id in tile_ids:
        wac_cubes = [f for f in wac_datacubes if tile_id in f]
        static_cubes = [f for f in static_datacubes if tile_id in f]
        if len(wac_cubes) > 0 and len(static_cubes) > 0:
            wac_cube = wac_cubes[0]
            static_cube = static_cubes[0]
            cubes_by_tile[tile_id] = {"wac": wac_cube, "static": static_cube}
        # otherwise we leave the dict without an element for this tile
        #  this way rxr doesn't try to open an invalid filename

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

# ============================================================
# DATA PREPROCESSING (mimic training)
# ============================================================

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

# ============================================================
# SLIDING WINDOW
# ============================================================

def calculate_tile_position(tile_id, data_shape, tile_shape, overlap, mode="reflect"):
    """
    Calculate the position of a tile in the image based on tile_id.

    Args:
        tile_id: Index of the tile
        data_shape: Shape of full image (H, W, C)
        tile_shape: Shape of tiles (tile_H, tile_W, C)
        overlap: Overlap fraction or pixels
        mode: Tiling mode

    Returns:
        (y_start, y_end, x_start, x_end)
    """
    img_h, img_w, channels = data_shape
    tile_h, tile_w, _ = tile_shape

    # Calculate step size (distance between tile starts)
    if isinstance(overlap, float):
        step_h = int(tile_h * (1 - overlap))
        step_w = int(tile_w * (1 - overlap))
    else:
        step_h = tile_h - overlap
        step_w = tile_w - overlap

    # Calculate number of tiles in each dimension
    n_tiles_h = int(np.ceil((img_h - tile_h) / step_h)) + 1
    n_tiles_w = int(np.ceil((img_w - tile_w) / step_w)) + 1

    # Calculate row and column from tile_id
    row = tile_id // n_tiles_w
    col = tile_id % n_tiles_w

    # Calculate position
    y_start = row * step_h
    x_start = col * step_w

    # Adjust end positions (may extend beyond image if using reflect mode)
    y_end = min(y_start + tile_h, img_h)
    x_end = min(x_start + tile_w, img_w)

    return y_start, y_end, x_start, x_end


def sliding_window_inference(
    images_scaled,
    model,
    target_size=304,
    device="cuda",
    threshold=0.5,
    n_channels=12,
    overlap=0.25,
    debug=False,
    window='triang',
    return_tiles=False
):
    """
    Perform sliding window inference on large images.
    Merges probabilities then thresholds for smoother results.

    Args:
        images_scaled: Input images (N, H, W, C) or (H, W, C)
        model: PyTorch model for inference
        target_size: Size of tiles for model input
        device: Device to run inference on
        threshold: Threshold for binary predictions
        n_channels: Number of input channels
        overlap: Tile overlap as fraction (0-1) or pixels (int)
        debug: If True, print detailed debugging info
        window: Blending window type
        return_tiles: If True, also return individual tile predictions

    Returns:
        predictions: Binary predictions
        probabilities: Probability maps
        tile_info: (Optional) Dict with 'tile_predictions' and 'tile_positions'
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    model.eval()

    # Handle single image
    single_image = False
    if images_scaled.ndim == 3:
        images_scaled = images_scaled[np.newaxis, ...]
        single_image = True

    all_predictions = []
    all_probabilities = []
    all_tile_info = []

    # Process each image
    for idx in range(images_scaled.shape[0]):
        print(f"\nProcessing image {idx + 1}/{images_scaled.shape[0]}")
        image = images_scaled[idx]  # Shape: (H, W, n_channels)

        img_h, img_w = image.shape[0], image.shape[1]

        if debug:
            print(f"Image shape: {image.shape}")
            print(f"Target tile size: {target_size}")

        # Create INPUT tiler with overlap
        input_tiler = Tiler(
            data_shape=image.shape,
            tile_shape=(target_size[0], target_size[1], n_channels),
            channel_dimension=-1,
            overlap=overlap,
            mode="reflect",
        )

        # Create OUTPUT tiler
        output_tiler = Tiler(
            data_shape=(img_h, img_w, 1),
            tile_shape=(target_size[0], target_size[1], 1),
            channel_dimension=-1,
            overlap=overlap,
            mode="reflect",
        )

        # Verify tilers match
        if len(input_tiler) != len(output_tiler):
            raise ValueError(
                f"Tile count mismatch! Input: {len(input_tiler)}, "
                f"Output: {len(output_tiler)}"
            )

        # Create merger
        output_merger = Merger(tiler=output_tiler, window=window)

        print(f"Number of tiles: {len(input_tiler)}")

        # Calculate and display effective overlap
        if isinstance(overlap, float):
            overlap_pixels = int(target_size[0] * overlap)
            print(f"Tile overlap: {overlap*100:.0f}% ({overlap_pixels} pixels)")
        else:
            print(f"Tile overlap: {overlap} pixels")

        # Storage for tile info
        tile_predictions = []
        tile_positions = []

        # Iterate through tiles
        tile_count = 0
        with torch.no_grad():
            for tile_id, tile_batch in input_tiler(
                image, batch_size=1, progress_bar=True
            ):
                tile_count += 1
                tile = tile_batch[0]

                # Convert to torch tensor
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1)
                tile_tensor = tile_tensor.unsqueeze(0).float().to(device)

                # Run inference
                logits = model(tile_tensor)
                probs = torch.sigmoid(logits)

                # Get crater class
                if probs.shape[1] == 2:
                    probs = probs[:, 1:2]

                # Convert to numpy
                probs_np = probs.cpu().numpy()
                probs_np = probs_np[0]
                probs_np = np.transpose(probs_np, (1, 2, 0))

                # Store tile prediction and position if requested
                if return_tiles:
                    # Threshold this tile
                    tile_pred = (probs_np.squeeze() > threshold).astype(np.float32)

                    # Calculate tile position manually
                    y_start, y_end, x_start, x_end = calculate_tile_position(
                        tile_id=tile_id,
                        data_shape=image.shape,
                        tile_shape=(target_size[0], target_size[1], n_channels),
                        overlap=overlap,
                        mode="reflect"
                    )

                    tile_predictions.append((tile_pred, tile_id))
                    tile_positions.append((y_start, y_end, x_start, x_end))

                # Add to merger
                output_merger.add(tile_id, probs_np)

        if debug:
            print(f"\nProcessed {tile_count} tiles total")

        # Merge all tiles
        merged_probs = output_merger.merge(unpad=True)

        print(f"Merged probabilities shape: {merged_probs.shape}")
        print(f"Merged prob range: [{merged_probs.min():.4f}, {merged_probs.max():.4f}]")

        # Check if merging actually blended values
        unique_vals = np.unique(merged_probs)
        if debug:
            print(f"Number of unique probability values: {len(unique_vals)}")
            if len(unique_vals) < 100:
                print(f"Warning: Very few unique values, blending might not be working!")

        # Threshold merged predictions
        merged_preds = (merged_probs > threshold).astype(np.float32)
        merged_preds = merged_preds.squeeze(-1)

        all_predictions.append(merged_preds)
        all_probabilities.append(merged_probs.squeeze(-1))

        # Store tile info
        if return_tiles:
            all_tile_info.append({
                'tile_predictions': tile_predictions,
                'tile_positions': tile_positions,
                'img_shape': (img_h, img_w),
                'target_size': target_size
            })

        print(f"Final prediction shape: {merged_preds.shape}")
        print(f"Prediction range: [{merged_preds.min():.2f}, {merged_preds.max():.2f}]")
        print(f"Positive pixels: {(merged_preds > 0).sum()} / {merged_preds.size}")

    predictions = np.stack(all_predictions, axis=0)
    probabilities = np.stack(all_probabilities, axis=0)

    if single_image:
        predictions = predictions[0]
        probabilities = probabilities[0]
        if return_tiles:
            all_tile_info = all_tile_info[0]

    if return_tiles:
        return predictions, probabilities, all_tile_info
    else:
        return predictions, probabilities

# ============================================================
# VIZ FCNS
# ============================================================

def create_binary_colormap(instance_mask):
    """
    Create a simple binary colormap: 0 -> black, any other value -> red.

    Args:
        instance_mask: (H, W) or (H, W, 1) array with instance IDs (0 = background)

    Returns:
        colored: (H, W, 3) RGB array with values in [0, 1]
    """
    # Ensure 2D by squeezing any extra dimensions
    if instance_mask.ndim > 2:
        instance_mask = instance_mask.squeeze()

    # If still not 2D after squeeze, take first channel
    if instance_mask.ndim > 2:
        instance_mask = instance_mask[:, :, 0]

    h, w = instance_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    # Set all non-zero pixels to red
    mask = instance_mask != 0
    colored[mask] = [1.0, 0.0, 0.0]  # Pure red

    return colored


def get_tile_color(tile_idx, n_colors=20):
    """
    Get consistent color for a tile index.

    Args:
        tile_idx: Index of the tile
        n_colors: Number of colors in colormap

    Returns:
        RGB tuple (R, G, B)
    """
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab20
    color_idx = tile_idx % n_colors
    return np.array(cmap(color_idx)[:3])  # RGB only


def visualize_tile_grid(
    img_shape,
    tile_positions,
    target_size=(304, 304),
    alpha=0.15,
    border_width=3
):
    """
    Show tiles with transparent fills and solid colored borders.
    Overlaps will show multiple borders.
    """
    # Start with black background
    canvas = np.zeros((*img_shape, 3), dtype=np.float32)

    for tile_idx, (y_start, y_end, x_start, x_end) in enumerate(tile_positions):
        color_rgb = get_tile_color(tile_idx)

        # Calculate TRUE tile extent (304x304)
        true_y_end = y_start + target_size[0]
        true_x_end = x_start + target_size[1]

        # Clip to visible region
        vis_y_end = min(true_y_end, img_shape[0])
        vis_x_end = min(true_x_end, img_shape[1])

        # Fill with transparent color
        canvas[y_start:vis_y_end, x_start:vis_x_end] += color_rgb * alpha

        # Draw solid borders (these will show overlaps clearly)
        # Top border
        if y_start < img_shape[0]:
            canvas[y_start:min(y_start+border_width, vis_y_end), x_start:vis_x_end] = color_rgb

        # Bottom border
        if vis_y_end > border_width:
            canvas[max(vis_y_end-border_width, y_start):vis_y_end, x_start:vis_x_end] = color_rgb

        # Left border
        if x_start < img_shape[1]:
            canvas[y_start:vis_y_end, x_start:min(x_start+border_width, vis_x_end)] = color_rgb

        # Right border
        if vis_x_end > border_width:
            canvas[y_start:vis_y_end, max(vis_x_end-border_width, x_start):vis_x_end] = color_rgb

    # Clip values
    canvas = np.clip(canvas, 0, 1)
    return canvas


def visualize_tiles_with_colors(
    img_shape,
    tile_predictions,
    tile_positions,
    merged_prediction,
    target_size,
    alpha=0.25
):
    """
    Create visualization showing which tiles contributed to the FINAL merged predictions.
    Only colors pixels that are actually predicted in the merged result.
    Uses same colors as visualize_tile_grid for consistency.

    Args:
        img_shape: Shape of full image (H, W)
        tile_predictions: List of (tile_pred, tile_id) tuples
        tile_positions: List of (y_start, y_end, x_start, x_end) for each tile
        merged_prediction: Final merged prediction mask (H, W) - boolean or 0/1
        target_size: Size of tiles (H, W)
        alpha: Not used (kept for compatibility)

    Returns:
        RGB image with colored predictions (H, W, 3)
    """
    # Create output canvas (RGB)
    canvas = np.zeros((*img_shape, 3), dtype=np.float32)

    # Track which tiles contribute to each pixel
    tile_contributions = np.zeros((*img_shape, len(tile_predictions)), dtype=np.float32)

    # Record which tiles predicted positive at each pixel
    for tile_idx, ((tile_pred, tile_id), (y_start, y_end, x_start, x_end)) in enumerate(
        zip(tile_predictions, tile_positions)
    ):
        # Get tile region
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Resize tile pred if needed
        if tile_pred.shape != (tile_h, tile_w):
            from scipy.ndimage import zoom
            zoom_h = tile_h / tile_pred.shape[0]
            zoom_w = tile_w / tile_pred.shape[1]
            tile_pred_resized = zoom(tile_pred, (zoom_h, zoom_w), order=0)
        else:
            tile_pred_resized = tile_pred

        # Mark where this tile predicted positive
        tile_contributions[y_start:y_end, x_start:x_end, tile_idx] = tile_pred_resized > 0

    # Now, ONLY color pixels that are in the final merged prediction
    merged_mask = merged_prediction > 0

    # For each pixel in the merged prediction, color it based on contributing tiles
    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            if merged_mask[y, x]:
                # Find which tiles contributed to this pixel
                contributing_tile_indices = np.where(tile_contributions[y, x, :] > 0)[0]

                if len(contributing_tile_indices) > 0:
                    # Use the maximum (last) contributing tile
                    max_tile_idx = contributing_tile_indices.max()
                    color = get_tile_color(max_tile_idx)
                    canvas[y, x, :] = color

    return canvas


def visualize_tiles_with_boundaries(
    img_shape,
    tile_predictions,
    tile_positions,
    target_size,
    border_width=2
):
    """
    Create visualization of prediction tiles with red boundaries.

    Args:
        img_shape: Shape of full image (H, W)
        tile_predictions: List of (tile_pred, tile_id) tuples
        tile_positions: List of (y_start, y_end, x_start, x_end) for each tile
        target_size: Size of tiles (H, W)
        border_width: Width of red border in pixels

    Returns:
        RGB image with tiles and boundaries (H, W, 3)
    """
    # Create output canvas (black background)
    canvas = np.zeros((*img_shape, 3), dtype=np.float32)

    # Place each tile prediction on canvas
    for (tile_pred, tile_id), (y_start, y_end, x_start, x_end) in zip(
        tile_predictions, tile_positions
    ):
        # Get tile region
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Resize tile pred if needed
        if tile_pred.shape != (tile_h, tile_w):
            from scipy.ndimage import zoom
            zoom_h = tile_h / tile_pred.shape[0]
            zoom_w = tile_w / tile_pred.shape[1]
            tile_pred_resized = zoom(tile_pred, (zoom_h, zoom_w), order=0)
        else:
            tile_pred_resized = tile_pred

        # Place tile (white for predictions)
        canvas[y_start:y_end, x_start:x_end, :] = np.stack([
            tile_pred_resized, tile_pred_resized, tile_pred_resized
        ], axis=-1)

    # Draw red borders around tiles
    for y_start, y_end, x_start, x_end in tile_positions:
        # Top border
        canvas[
            max(0, y_start):min(img_shape[0], y_start + border_width),
            x_start:x_end,
            0
        ] = 1.0  # Red channel

        # Bottom border
        canvas[
            max(0, y_end - border_width):min(img_shape[0], y_end),
            x_start:x_end,
            0
        ] = 1.0

        # Left border
        canvas[
            y_start:y_end,
            max(0, x_start):min(img_shape[1], x_start + border_width),
            0
        ] = 1.0

        # Right border
        canvas[
            y_start:y_end,
            max(0, x_end - border_width):min(img_shape[1], x_end),
            0
        ] = 1.0

    return canvas

# ============================================================
# DRIVER DATACUBE INFERENCE
# ============================================================

def run_datacube_inference(
    model,
    device,
    input_dir,
    mean,
    std,
    output_dir="outputs/cube_inference",
    n_images=20,
    model_native_size=304,
    tile_overlap=0.75,
    threshold=0.3,
    save_inputs_dir=None,
    debug=False,
    tile_window='triang',
    visualize_tiles=True
):
    """
    Run inference on 12-band datacubes with detailed tile visualization.
    """
    model.eval()

    print(f"Loading up to {n_images} datacubes from TIFF files...")

    # Load and preprocess data
    images_raw, file_paths = get_datacube_data(
        input_paths=input_dir,
        max_images=n_images,
        verbose=False,
    )

    print(f"Raw datacubes shape: {images_raw.shape}")

    # Transpose and scale
    images_transposed = np.transpose(images_raw, (0, 2, 3, 1))
    print(f"Transposed to: {images_transposed.shape}")

    print("\nApplying min-max scaling per band...")
    images_scaled = np.zeros_like(images_transposed, dtype=np.float32)
    for i in range(len(images_transposed)):
        images_scaled[i] = min_max_scale_bands(images_transposed[i])

    print(f"After scaling: min={images_scaled.min():.3f}, max={images_scaled.max():.3f}")

    # Run inference
    n_channels = images_scaled.shape[-1]
    print(f"\n{'='*60}")
    print(f"Running inference with tile visualization: {visualize_tiles}")
    print(f"{'='*60}\n")

    # Get predictions with tile info if visualizing
    if visualize_tiles:
        preds_list, probabilities, tile_info = sliding_window_inference(
            images_scaled,
            model,
            target_size=model_native_size,
            device=device,
            threshold=threshold,
            n_channels=n_channels,
            overlap=tile_overlap,
            debug=debug,
            window=tile_window,
            return_tiles=True
        )
    else:
        preds_list, probabilities = sliding_window_inference(
            images_scaled,
            model,
            target_size=model_native_size,
            device=device,
            threshold=threshold,
            n_channels=n_channels,
            overlap=tile_overlap,
            debug=debug,
            window=tile_window,
            return_tiles=False
        )
        tile_info = None

    print(f"\nGot predictions")

    # ============================================
    # Create visualization
    # ============================================
    print("\nCreating visualization...")

    batch_size = min(len(images_scaled), 10)

    # 5 rows if visualizing tiles, 2 rows otherwise
    n_rows = 5 if visualize_tiles else 2
    fig, axes = plt.subplots(n_rows, batch_size, figsize=(5 * batch_size, 5 * n_rows))

    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    # Extract filenames
    display_filenames = [
        Path(wac_file).stem for wac_file, static_file in file_paths[:batch_size]
    ]

    for i in range(batch_size):
        img = images_scaled[i]

        # Handle both single image and batch cases
        if isinstance(preds_list, np.ndarray):
            if preds_list.ndim == 3:  # Batch: (N, H, W)
                pred_mask = preds_list[i]
            elif preds_list.ndim == 2:  # Single image: (H, W)
                pred_mask = preds_list
            else:  # Unexpected shape, try to extract
                pred_mask = np.squeeze(preds_list)
                if pred_mask.ndim == 3:
                    pred_mask = pred_mask[i]
        else:
            pred_mask = preds_list[i]

        # Ensure pred_mask is 2D
        if pred_mask.ndim > 2:
            pred_mask = pred_mask.squeeze()

        # Get single channel for visualization
        img_vis = img[:, :, 0]

        # Row 0: Original image
        axes[0, i].imshow(img_vis, cmap='gray')
        axes[0, i].set_title(
            f"{display_filenames[i]}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0, i].axis("off")

        if visualize_tiles:
            # Get tile info for this image
            if isinstance(tile_info, list):
                img_tile_info = tile_info[i]
            else:
                img_tile_info = tile_info

            # DEBUG: Print actual tile information
            print(f"\nDEBUG for image {i}:")
            print(f"  Number of tile_predictions: {len(img_tile_info['tile_predictions'])}")
            print(f"  Number of tile_positions: {len(img_tile_info['tile_positions'])}")
            print(f"  Tile positions:")
            for idx, (y_start, y_end, x_start, x_end) in enumerate(img_tile_info['tile_positions']):
                print(f"    Tile {idx}: ({y_start}, {y_end}, {x_start}, {x_end})")

            # Row 1: Tile grid only (colored rectangles on black)
            tile_grid = visualize_tile_grid(
                img_shape=img_tile_info['img_shape'],
                tile_positions=img_tile_info['tile_positions'],
                alpha=0.25
            )
            axes[1, i].imshow(tile_grid)
            axes[1, i].set_title(
                f"Tile Grid (n={len(img_tile_info['tile_predictions'])})",
                fontsize=11
            )
            axes[1, i].axis("off")

            # Row 2: Tile predictions with red boundaries
            tile_boundaries = visualize_tiles_with_boundaries(
                img_shape=img_tile_info['img_shape'],
                tile_predictions=img_tile_info['tile_predictions'],
                tile_positions=img_tile_info['tile_positions'],
                target_size=img_tile_info['target_size'],
                border_width=3
            )
            axes[2, i].imshow(tile_boundaries)
            axes[2, i].set_title(
                f"Tile Predictions",
                fontsize=11
            )
            axes[2, i].axis("off")

            # Row 3: Colored prediction overlay of final pred
            tile_colors = visualize_tiles_with_colors(
                img_shape=img_tile_info['img_shape'],
                tile_predictions=img_tile_info['tile_predictions'],
                tile_positions=img_tile_info['tile_positions'],
                merged_prediction=pred_mask,  # NEW: pass in final merged prediction
                target_size=img_tile_info['target_size'],
                alpha=0.25
            )
            axes[3, i].imshow(img_vis, cmap='gray')
            axes[3, i].imshow(tile_colors, alpha=0.5)  # Can increase alpha since it matches now
            axes[3, i].set_title(
                f"Merged (Colored by Tile)",
                fontsize=11
            )
            axes[3, i].axis("off")

        # Row 4 (or 1 if no tiles): Merged prediction
        row_idx = 4 if visualize_tiles else 1
        pred_colored = create_binary_colormap(pred_mask)
        axes[row_idx, i].imshow(pred_colored, vmin=0, vmax=1)
        axes[row_idx, i].set_title(f"Merged Prediction", fontsize=11)
        axes[row_idx, i].axis("off")

    fig.suptitle(
        f"Inference on {images_scaled.shape[1]}×{images_scaled.shape[2]} Datacubes\n"
        f"Model: {model_native_size}×{model_native_size} | "
        f"Overlap: {tile_overlap} | Window: {tile_window} | "
        f"Threshold: {threshold}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.subplots_adjust(hspace=0.4)  # Adjust value as needed (0.3-0.5 typically works)
    plt.tight_layout()
    output_path = f"{output_dir}/inference_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")

    model.train()

    return images_scaled, preds_list