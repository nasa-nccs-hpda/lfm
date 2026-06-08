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


import xarray as xr
import rioxarray as rxr
import rasterio
import re
from pathlib import Path
import os
from glob import glob


def get_product_id(filepath):
    """
    Extract product ID from filename for pairing vis/static files.

    Example:
        'M115502559CE_r1950_c150_input' -> 'M115502559CE_r1950_c150'
        'M115502559CE_r1950_c150_Static_input' -> 'M115502559CE_r1950_c150'
    """
    basename = Path(filepath).parent.name
    # Remove '_input' and '_Static_input' suffixes
    product_id = basename.replace('_Static_input', '').replace('_input', '')
    return product_id


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
            - stacked_data: List of xarray DataArrays, one per location
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

    if verbose:
        print(
            f"\nFound {len(file_paths)} total .tif files. "
            f"Wac: {len(wac_cubes)}, Static: {len(static_cubes)}"
        )

    # Apply max_images limit
    if max_images is not None:
        file_pairs = file_pairs[:max_images]
        if verbose:
            print(f"Limiting to {max_images} pairs")

    # Process each pair
    all_datasets = []

    # We have already extracted only a single WAC/STATIC file per tile ID
    for tile_id, dataset_dict in cubes_by_tile.items():
        wac_file = dataset_dict['wac']  # Assume there's only a single file
        static_file = dataset_dict['static']  # Assume there's only a single file
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing tile_id: {tile_id}")
            print(f"  Wac: {wac_file}")
            print(f"  Static: {static_file}")

        # Load wac data
        wac_ds = rxr.open_rasterio(wac_file)
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

        if verbose:
            print(f"  Combined shape: {combined_ds.shape}")
            print(
                f"  Combined bands: wac({wac_ds.sizes['band']}) "
                f"+ static({static_ds.sizes['band']}) = {combined_ds.sizes['band']}"
            )

        all_datasets.append(combined_ds)

    if verbose:
        print(f"\n{'='*60}")
        total_bands = sum(ds.sizes['band'] for ds in all_datasets)
        print(
            f"✓ Extraction complete: {len(all_datasets)} combined datacubes, "
            f"{total_bands} total bands across all cubes"
        )
        if all_datasets:
            print(f"  Example shape: {all_datasets[0].shape}")

    file_pairs = [
        (cubes_by_tile[tid]['wac'], cubes_by_tile[tid]['static'])
        for tid in cubes_by_tile
    ]

    return all_datasets, file_pairs


def print_band_info(data_array, label="DataArray"):
    """
    Print information about bands in an xarray DataArray.
    Useful for debugging/verification.
    """
    print(f"\n{label} band information:")
    print(f"  Shape: {data_array.shape}")
    print(f"  Number of bands: {data_array.sizes['band']}")

    # Try to get band names if they exist
    if 'long_name' in data_array.attrs:
        print(f"  Bands: {data_array.attrs['long_name']}")

    # Print first few band values to check
    print(f"  Band dimension coordinates: {data_array.band.values}")



def plot_data_cubes(
    input_paths,
    mode="rgb",
    mean=None,
    std=None,
    max_images=None,
    band_filter=None,
    bands_per_slice=7,
    figsize=None,
    titles=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    suptitle=None,
    colorbar=False,
    normalize_per_band=True,
    apply_normalization=True,
    output_path="output.png",
    verbose=True,
):
    """
    Extract and plot data cubes from .tif files as either RGB composites or individual bands.

    Parameters:
    -----------
    input_paths : str, Path, or list
        Path(s) to search for .tif files. Can be:
        - Single directory path (will glob recursively)
        - Single .tif file path
        - List of paths
    mode : str, default='rgb'
        'rgb' for RGB composite or 'bands' for individual band plots
    mean : np.ndarray or None
        Mean values for normalization. Shape should be (n_bands,) or broadcastable.
        If None and apply_normalization=True, computed from data.
    std : np.ndarray or None
        Std values for normalization. Shape should be (n_bands,) or broadcastable.
        If None and apply_normalization=True, computed from data.
    max_images : int, optional
        Maximum number of image slices to extract and plot
    band_filter : list of int, optional
        Which bands to extract from each slice.
        For RGB mode: e.g., [5, 3, 2] extracts bands 5, 3, 2 as R, G, B
        For bands mode: e.g., [0, 1, 2, 3, 4] extracts first 5 bands
        If None, uses all bands in slice (determined by bands_per_slice)
    bands_per_slice : int, default=7
        Total number of bands per complete slice in the TIFF file
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    titles : list of str, optional
        Titles for each row (image). If None, uses filenames.
    cmap : str, default='viridis'
        Colormap for individual bands (only used in 'bands' mode)
    vmin, vmax : float, optional
        Color scale limits for individual bands. If None, auto-scales.
    suptitle : str, optional
        Overall figure title
    colorbar : bool, default=False
        Add colorbar to band plots (only in 'bands' mode)
    normalize_per_band : bool, default=True
        If True, normalize each band independently for display.
        Only applies to color scaling, not mean/std normalization.
    apply_normalization : bool, default=True
        If True, applies (data - mean) / std normalization
    output_path : str, default='output.png'
        Path to save the figure. If None, doesn't save.
    verbose : bool, default=True
        Print extraction progress

    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    data : np.ndarray
        Extracted raw data array (N, bands, H, W)
    data_normalized : np.ndarray
        Normalized data array (N, H, W, bands)
    file_paths : list
        List of file paths that were loaded

    Examples:
    ---------
    # RGB mode - extracts bands 5, 3, 2 as RGB
    fig, axes, data, data_norm, paths = plot_data_cubes(
        '/path/to/tifs/',
        mode='rgb',
        band_filter=[5, 3, 2],
        mean=my_mean,
        std=my_std,
        max_images=10
    )

    # Individual bands mode - extracts all 7 bands
    fig, axes, data, data_norm, paths = plot_data_cubes(
        '/path/to/tifs/',
        mode='bands',
        bands_per_slice=7,
        cmap='viridis',
        max_images=5
    )
    """

    # Determine band configuration based on mode
    if mode == "rgb":
        if band_filter is None:
            band_filter = [3, 1, 0]  # Default RGB bands
        if len(band_filter) != 3:
            raise ValueError(
                f"RGB mode requires exactly 3 bands in band_filter, got {len(band_filter)}"
            )
    elif mode == "bands":
        if band_filter is None:
            band_filter = list(range(bands_per_slice))  # Extract all bands
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'rgb' or 'bands'.")

    if verbose:
        print(f"Extracting data in '{mode}' mode...")
        print(f"Band filter: {band_filter}")

    # Extract images using extract_images
    data, file_paths = extract_images(
        input_paths=input_paths,
        band_filter=band_filter,
        bands_per_slice=bands_per_slice,
        max_images=max_images,
        verbose=verbose,
    )

    # data shape: (N, bands, H, W) - channels first, raw positive values
    n_images, n_bands, height, width = data.shape

    if verbose:
        print(f"Extracted {n_images} images with {n_bands} bands each")
        print(f"Image dimensions: {height}×{width}")

    # Transpose to (N, H, W, bands) for processing
    data_transposed = np.transpose(data, (0, 2, 3, 1))  # (N, H, W, bands)

    # Apply min-max scaling to [0, 1]
    data_scaled = np.zeros_like(data_transposed, dtype=np.float32)
    for i in range(n_images):
        for b in range(n_bands):
            band = data_transposed[i, :, :, b]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                data_scaled[i, :, :, b] = (band - band_min) / (
                    band_max - band_min
                )
            else:
                data_scaled[i, :, :, b] = band

    if verbose:
        print(
            f"After min-max scaling: min={data_scaled.min():.3f}, max={data_scaled.max():.3f}"
        )

    # Apply mean/std normalization if requested
    if apply_normalization:
        if mean is None:
            mean = data_scaled.mean(axis=(0, 1, 2))  # (n_bands,)
            if verbose:
                print(f"Computed mean from data: {mean}")
        else:
            mean = np.array(mean)
            if mean.ndim == 1:
                mean = mean.reshape(1, 1, 1, -1)  # (1, 1, 1, n_bands)

        if std is None:
            std = data_scaled.std(axis=(0, 1, 2))  # (n_bands,)
            if verbose:
                print(f"Computed std from data: {std}")
        else:
            std = np.array(std)
            if std.ndim == 1:
                std = std.reshape(1, 1, 1, -1)  # (1, 1, 1, n_bands)

        data_normalized = (data_scaled - mean) / (std + 1e-8)

        if verbose:
            print(
                f"After normalization: min={data_normalized.min():.3f}, max={data_normalized.max():.3f}"
            )
    else:
        data_normalized = data_scaled.copy()
        if verbose:
            print("Skipping normalization")

    # Generate titles from filenames if not provided
    if titles is None:
        titles = [Path(fp).stem for fp in file_paths]

    # Create visualization
    if mode == "rgb":
        # ==================== RGB Composite Mode ====================
        if figsize is None:
            figsize = (5 * min(n_images, 5), 5 * math.ceil(n_images / 5))

        # Calculate grid layout
        n_cols = min(n_images, 5)
        n_rows = math.ceil(n_images / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False
        )
        axes = axes.flatten()

        for i in range(n_images):
            # Use scaled data (pre-normalization) for visualization
            # This keeps values in [0, 1] range which looks good
            rgb = data_scaled[i, :, :, :]  # (H, W, 3)

            # Display RGB image
            axes[i].imshow(rgb)
            axes[i].axis("off")

            if i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
            else:
                axes[i].set_title(f"Image {i}", fontsize=10)

        # Turn off extra subplots
        for i in range(n_images, len(axes)):
            axes[i].axis("off")

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.99)
        else:
            fig.suptitle(
                f"RGB Data Cubes ({n_images} images)", fontsize=14, y=0.99
            )

    elif mode == "bands":
        # ==================== Individual Bands Mode ====================
        if figsize is None:
            figsize = (2.5 * n_bands, 2.5 * n_images)

        fig, axes = plt.subplots(
            n_images, n_bands, figsize=figsize, squeeze=False
        )

        # Calculate global vmin/vmax if not provided and not normalizing per band
        if not normalize_per_band:
            if vmin is None:
                vmin = data_scaled.min()  # Use scaled data for display
            if vmax is None:
                vmax = data_scaled.max()

        for i in range(n_images):
            for j in range(n_bands):
                # Use scaled data (pre-normalization) for visualization
                band_data = data_scaled[i, :, :, j]

                # Determine color limits
                if normalize_per_band:
                    band_vmin = band_data.min() if vmin is None else vmin
                    band_vmax = band_data.max() if vmax is None else vmax
                else:
                    band_vmin = vmin
                    band_vmax = vmax

                im = axes[i, j].imshow(
                    band_data, cmap=cmap, vmin=band_vmin, vmax=band_vmax
                )
                axes[i, j].axis("off")

                # Column titles (band index and value range) on each row
                if band_filter is not None:
                    original_band = band_filter[j]
                else:
                    original_band = j

                band_min = band_data.min()
                band_max = band_data.max()
                axes[i, j].set_title(
                    f"Band {original_band}\n[{band_min:.2f}, {band_max:.2f}]",
                    fontsize=9,
                )

                # Add colorbar if requested
                if colorbar:
                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    divider = make_axes_locatable(axes[i, j])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

            # Row titles (image name) on left
            if i < len(titles):
                axes[i, 0].set_ylabel(
                    titles[i],
                    rotation=0,
                    labelpad=80,
                    fontsize=10,
                    va="center",
                    ha="right",
                )
            else:
                axes[i, 0].set_ylabel(
                    f"Image {i}",
                    rotation=0,
                    labelpad=80,
                    fontsize=10,
                    va="center",
                    ha="right",
                )

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.99)
        else:
            fig.suptitle(
                f"Data Cube Bands ({n_images} images × {n_bands} bands)",
                fontsize=14,
                y=0.99,
            )

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"✓ Saved figure to: {output_path}")

    if verbose:
        print(f"✓ Plotted {n_images} images with {n_bands} bands each")

    return fig, axes, data, data_normalized, file_paths


def min_max_scale_bands(bands):
    """Min-max scale each band to [0, 1]"""
    scaled = np.zeros_like(bands, dtype=np.float32)
    for i in range(bands.shape[0]):
        band = bands[i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            scaled[i] = (band - band_min) / (band_max - band_min)
        else:
            scaled[i] = band
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


def save_image_to_tif(i, image, n_bands=7, save_path="image.tif"):
    """Writes 7-band image to save_path as .tif"""
    driver = gdal.GetDriverByName("GTiff")

    # Create output file with 7 bands
    output_dataset = driver.Create(
        save_path,
        image.shape[2],  # width (columns)
        image.shape[1],  # height (rows)
        n_bands,  # number of bands
        gdal.GDT_Float32,
    )

    # Write each band
    for band_num in range(7):
        output_band = output_dataset.GetRasterBand(
            band_num + 1
        )  # Band numbering starts at 1
        output_band.WriteArray(image[band_num, :, :])
        output_band.FlushCache()

    print(f"Saved all bands to {save_path}")

    # Close the dataset
    output_dataset = None
    return


def get_tilers_mergers(images_npy, TARGET_SIZE):
    """Gets tilers and mergers used for sliding window inference."""
    print("Creating tilers and mergers...")
    input_tiler = Tiler(
        data_shape=images_npy.shape,
        tile_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        channel_dimension=-1,
        mode="reflect",
    )
    # Model output should be (304, 304, 1), aka target size
    output_tiler = Tiler(
        data_shape=images_npy.shape,
        tile_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1),
        channel_dimension=-1,
        mode="reflect",
    )
    output_merger = Merger(tiler=output_tiler, window="triang")
    return input_tiler, output_tiler, output_merger


def sliding_window_inference(
    images_npy, model, target_size=304, device="cuda", threshold=0.5
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
        image = images_npy[idx]  # Shape: (512, 512, 3)

        # Create INPUT tiler
        input_tiler = Tiler(
            data_shape=image.shape,  # (512, 512, 3)
            tile_shape=(target_size[0], target_size[1], 3),
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
                # tile_batch shape: (1, 304, 304, 3)
                tile = tile_batch[0]  # (304, 304, 3)

                # Convert to torch tensor and change to channels-first
                tile_tensor = torch.from_numpy(tile).permute(
                    2, 0, 1
                )  # (3, 304, 304)
                tile_tensor = (
                    tile_tensor.unsqueeze(0).float().to(device)
                )  # (1, 3, 304, 304)

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
    normalize=True,
    save_inputs_dir=None,
    use_sliding=True,
    band_filter=None,
):
    model.eval()

    # Load numpy array of 3-band images at full resolution
    print(f"Loading up to {n_images} images from TIFF files...")

    # NEW: extract_images now returns (data, file_paths)
    # data shape: (N, bands, H, W) - channels first, unnormalized
    images_raw, file_paths = extract_images(
        input_paths=input_dir,
        band_filter=band_filter,  # e.g., [3, 1, 0]
        bands_per_slice=7,  # How many bands per slice in the original file
        max_images=n_images,
        verbose=True,
    )

    print(
        f"Extracted raw images shape: {images_raw.shape}"
    )  # (N, 3, 512, 512)

    # NEW: Transpose from (N, bands, H, W) to (N, H, W, bands) for processing
    images_transposed = np.transpose(
        images_raw, (0, 2, 3, 1)
    )  # (N, 512, 512, 3)
    print(f"Transposed to: {images_transposed.shape}")

    # NEW: Apply min-max scaling to [0, 1]
    images_scaled = np.zeros_like(images_transposed, dtype=np.float32)
    for i in range(len(images_transposed)):
        images_scaled[i] = min_max_scale_bands(images_transposed[i])

    print(
        f"After min-max scaling: min={images_scaled.min():.3f}, max={images_scaled.max():.3f}"
    )

    # NEW: Apply mean/std normalization
    if mean is not None and std is not None:
        # Reshape mean and std for broadcasting
        mean_reshaped = mean.reshape(1, 1, 1, 3)  # (1, 1, 1, 3)
        std_reshaped = std.reshape(1, 1, 1, 3)  # (1, 1, 1, 3)

        images_npy = (images_scaled - mean_reshaped) / (std_reshaped + 1e-8)
        print(
            f"After normalization: min={images_npy.min():.3f}, max={images_npy.max():.3f}"
        )
    else:
        images_npy = images_scaled
        print(
            "Warning: No mean/std provided, using scaled data without normalization"
        )

    print(
        f"Final images shape ready for inference: {images_npy.shape}"
    )  # (N, 512, 512, 3)

    # Save inputs if specified
    if save_inputs_dir:
        save_inputs_dir = Path(save_inputs_dir)
        save_inputs_dir.mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(images_npy):
            save_path = save_inputs_dir / f"image_{i}.tif"
            image_transposed = np.transpose(image, (2, 0, 1))
            save_image_to_tif(
                i, image_transposed, n_bands=3, save_path=str(save_path)
            )

    # Rest of inference code remains the same...
    print(f"\n{'='*60}")
    print(f"Running inference:")
    print(f"  Input images: {images_npy.shape[1]}×{images_npy.shape[2]}")
    print(
        f"  Model native resolution: {model_native_size}×{model_native_size}"
    )
    print(f"  Processing with sliding window (overlap={tile_overlap})")
    print(f"{'='*60}\n")

    # Run sliding window inference
    if use_sliding:
        target_size = (model_native_size, model_native_size)
        preds_list, probabilities = sliding_window_inference(
            images_npy,
            model,
            target_size=model_native_size,
            device=device,
            threshold=threshold,
        )
    else:  # add support for resizing of inputs then regular pred
        preds_list = None

    print(f"\nGot {len(preds_list)} predictions")
    print(
        f"Output mask shapes: {[p.shape for p in preds_list[:3]]}"
    )  # Should all be (512, 512)

    # Create visualization
    print("\nCreating visualization...")

    batch_size = min(len(images_npy), 10)  # Limit visualization to 10 images
    fig, axes = plt.subplots(2, batch_size, figsize=(5 * batch_size, 10))

    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    # Use actual filenames from extract_images
    display_filenames = [Path(fp).name for fp in file_paths[:batch_size]]

    for i in range(batch_size):
        # Use original scaled images (before normalization) for visualization
        img = images_scaled[i]  # (512, 512, 3) in [0, 1]
        pred_mask = preds_list[i]  # (512, 512)

        # Row 0: Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(
            f"{display_filenames[i]}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0, i].axis("off")

        # Row 1: Inference with binary colormap
        pred_colored = create_binary_colormap(pred_mask)
        axes[1, i].imshow(pred_colored, vmin=0, vmax=1)
        axes[1, i].set_title(
            f"Inference",
            fontsize=11,
        )
        axes[1, i].axis("off")

    fig.suptitle(
        f"Inference on {images_npy.shape[1]}×{images_npy.shape[2]} Images "
        f"(processed with {model_native_size}×{model_native_size} tiles)\n"
        f"Threshold={threshold}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")

    model.train()

    return images_npy, preds_list
