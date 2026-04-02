# Standard library imports
from datetime import datetime
from glob import glob
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import xarray as xr
import rasterio

from transformers import AutoImageProcessor
from tiler import Tiler, Merger


import rasterio
from glob import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


def plot_data_cubes(input_dir, n_samples, n_bands, output_dir):
    tiffs = glob(f"{input_dir}/*.tif")
    samples = tiffs[:n_samples]
    images = []
    filenames = []

    now = datetime.now()
    time_str = now.strftime("%y%m%d_%H%M")
    output_basename = f"{time_str}"

    print(f"Loading {len(samples)} TIFF files...")
    for tiff_path in samples:
        with rasterio.open(tiff_path) as src:
            ds_band_count = src.count
            base_tiff_path = Path(tiff_path).name

            print(f"Opened {base_tiff_path} with {ds_band_count} total bands")
            print(
                f"Searching for {n_bands} bands with all valid (positive) data..."
            )

            # Find bands with all valid (positive) data
            valid_band_indices = []
            for band_idx in range(1, ds_band_count + 1):
                band_data = src.read(band_idx)

                # Check if all values are positive
                if np.all(band_data > 0):
                    valid_band_indices.append(band_idx)
                    print(f"  Band {band_idx}: VALID (all positive values)")

                    # Stop once we have enough valid bands
                    if len(valid_band_indices) >= n_bands:
                        break
                else:
                    num_negative = np.sum(band_data <= 0)
                    print(
                        f"  Band {band_idx}: INVALID ({num_negative} non-positive values)"
                    )

            # Check if we found enough valid bands
            if len(valid_band_indices) < n_bands:
                print(
                    f"WARNING: Only found {len(valid_band_indices)} valid bands out of {n_bands} requested"
                )
                print(f"Proceeding with {len(valid_band_indices)} bands")
            else:
                print(
                    f"Found {n_bands} valid bands: {valid_band_indices[:n_bands]}"
                )

            # Use only the valid bands we found
            valid_band_indices = valid_band_indices[:n_bands]

            if len(valid_band_indices) == 0:
                print(f"No valid bands found in {base_tiff_path}. Skipping...")
                continue

            # Read only the valid bands
            img_array = src.read(valid_band_indices)
            actual_n_bands = len(valid_band_indices)

            print("\n" + "=" * 60)
            print("PLOTTING VALID BANDS")
            print("=" * 60)

            for i in range(0, actual_n_bands, 25):
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                bands_subset = img_array[i : i + 25]
                band_idx = 0

                for row in range(5):
                    for col in range(5):
                        current_band_num = band_idx + i

                        if (
                            band_idx < len(bands_subset)
                            and current_band_num < actual_n_bands
                        ):
                            band_to_plot = bands_subset[band_idx]
                            band_min, band_max = np.min(band_to_plot), np.max(
                                band_to_plot
                            )
                            print(
                                f"Band {valid_band_indices[current_band_num]} min, max:",
                                band_min,
                                band_max,
                            )

                            # Since we've pre-filtered for positive values,
                            # we can optionally still mask if needed, or skip masking
                            # Here I'm keeping minimal masking for safety
                            masked_band = band_to_plot
                            valid_values = band_to_plot.flatten()

                            if valid_values.size > 0:
                                min_val = valid_values.min()
                                max_val = valid_values.max()
                                print(
                                    f"Range of values: [{min_val}, {max_val}]"
                                )
                                unique_vals, counts = np.unique(
                                    valid_values, return_counts=True
                                )
                                print(
                                    f"Number of unique values: {len(unique_vals)}"
                                )

                            # Get vmin/vmax of plot based on data range
                            vmin = np.percentile(valid_values, 2)
                            vmax = np.percentile(valid_values, 98)

                            if vmin == vmax:
                                vmin = valid_values.min()
                                vmax = valid_values.max()
                                if vmin == vmax:
                                    vmax = vmin + 1

                            cmap = plt.cm.viridis.copy()

                            im = axes[row, col].imshow(
                                masked_band, cmap=cmap, vmin=vmin, vmax=vmax
                            )
                            axes[row, col].set_title(
                                f"Band {valid_band_indices[current_band_num]}",
                                fontsize=8,
                            )
                            axes[row, col].axis("off")
                            fig.colorbar(
                                im, ax=axes[row, col], fraction=0.046, pad=0.04
                            )
                        else:
                            axes[row, col].axis("off")

                        band_idx += 1

                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/{base_tiff_path}_{output_basename}{i}_{i+25}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                print(
                    f"Saved figure to path: {output_dir}/{base_tiff_path}_{output_basename}{i}_{i+25}.png"
                )


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


def extract_images(
    tiff_paths,
    num_images_to_extract,
    bands_per_slice=7,
    bands_per_image=3,
    band_order=[2, 0, 1],
    mean=None,
    std=None,
):
    """
    Extract unique spatial images by processing complete band slices.

    Logic:
    - Each 7-band slice represents one spatial image
    - Extract only the first `bands_per_image` bands from each slice
    - Check if those bands have all positive values
    - If yes, keep them; if no, skip the entire slice
    - This prevents spatial duplication

    Args:
        tiff_paths: List of TIFF file paths
        num_images_to_extract: Number of unique spatial images to extract
        bands_per_slice: Total bands per complete image (default: 7)
        bands_per_image: Number of bands to extract from each slice (default: 3)
        band_order: Order to arrange the extracted bands (e.g., [2, 0, 1] for RGB)

    Returns:
        Array of shape (N, H, W, bands_per_image)
    """
    if mean is None or std is None:
        raise ValueError("Mean/std are required to preprocess inputs.")
    else:
        print(f"Extracting images with mean={mean}, std={std}")

    extracted = []

    for tiff_path in tiff_paths:
        if len(extracted) >= num_images_to_extract:
            break

        with xr.open_dataset(tiff_path, engine="rasterio") as ds:
            total_bands = ds.sizes["band"]
            num_slices = total_bands // bands_per_slice

            print(
                f"Processing {tiff_path}: {total_bands} bands, {num_slices} complete {bands_per_slice}-band slices"
            )

            for slice_idx in range(num_slices):
                if len(extracted) >= num_images_to_extract:
                    break

                # Calculate starting band for this 7-band slice
                start_band = slice_idx * bands_per_slice

                # Extract first `bands_per_image` bands from this slice
                slice_data = ds.isel(
                    band=slice(start_band, start_band + bands_per_image)
                )
                data_var = list(slice_data.data_vars)[0]
                image_data = slice_data[
                    data_var
                ].values  # Shape: (bands_per_image, H, W)

                # Check if ALL values in these bands are positive
                NODATA = -3.4028227e38
                nodata_count = np.sum(image_data == NODATA)
                nodata_percentage = (nodata_count / image_data.size) * 100
                if nodata_count == 0:  # zero tolerance
                    # Reorder bands (e.g., [2, 0, 1] for RGB order)
                    image_data_reordered = image_data[band_order, :, :]

                    # Transpose to (H, W, C) for channels-last format
                    image_data_reordered = np.transpose(
                        image_data_reordered, (1, 2, 0)
                    )

                    # Scale inputs to 0, 1
                    image_data_scaled = min_max_scale_bands(
                        image_data_reordered
                    )

                    # Normalize to work with shape
                    mean_reshaped = mean.reshape(1, 1, 3)  # 1 value per band
                    std_reshaped = std.reshape(1, 1, 3)  # 1 value per band
                    print(
                        f"Image shape before norm, after resize: {image_data_reordered.shape}"
                    )
                    print(
                        f"Mean, std shapes before norm: {mean_reshaped.shape, std_reshaped.shape}"
                    )
                    image_data_norm = (
                        image_data_scaled - mean_reshaped
                    ) / std_reshaped

                    extracted.append(image_data_norm)
                    print(
                        f"  ✓ Slice {slice_idx}: bands {start_band}-{start_band + bands_per_image - 1} → valid"
                    )
                else:
                    print(
                        f"  ✗ Slice {slice_idx}: bands {start_band}-{start_band + bands_per_image - 1} → skipped (non-positive values)"
                    )
                    min_val = np.min(image_data)
                    print(
                        f"     Slice min value: {min_val}, count and percent: {nodata_count}, {nodata_percentage}"
                    )

    if len(extracted) < num_images_to_extract:
        raise ValueError(
            f"Requested {num_images_to_extract} valid images but only found "
            f"{len(extracted)} with all positive values across all TIFFs"
        )

    result = np.array(extracted)
    print(
        f"\nFinal extracted array shape: {result.shape}"
    )  # Should be (N, H, W, 3)

    # Show first few for verification
    for i, elem in enumerate(result[:5]):
        print(
            f"Image {i} shape: {elem.shape}, min: {elem.min():.2f}, max: {elem.max():.2f}"
        )

    return result


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
    tiff_dirs,
    mean,  # No longer needed, but kept for compatibility
    std,  # No longer needed, but kept for compatibility
    output_path="inference_test.png",
    n_images=20,
    model_native_size=304,
    tile_overlap=0.25,
    threshold=0.75,
    normalize=True,  # Ignore this parameter
    save_inputs_dir=None,
    use_sliding=True,
):
    model.eval()

    # Load TIFF files
    all_tiff_files = []
    for dir in tiff_dirs:
        globbed = glob(f"{tiff_dir}/*.tif")
        all_tiff_files = all_tiff_files + globbed
    print(f"Found {len(tiffs)} TIFF files")

    if len(tiffs) == 0:
        raise ValueError(f"No TIFF files found in {tiff_dir}")

    # Load numpy array of 3-band images at full resolution
    print(f"Loading {n_images} images from TIFF files...")
    images_npy = extract_images(
        tiff_paths=tiffs,
        num_images_to_extract=n_images,
        bands_per_slice=5,
        bands_per_image=3,
        band_order=[2, 0, 1],
        mean=mean,
        std=std,
    )
    print(f"Extracted images shape: {images_npy.shape}")  # (N, 512, 512, 3)
    return

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

    # Pass raw images, image_processor handles normalization
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
            device="cuda",
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

    filenames = [f"image_{i}" for i in range(batch_size)]

    for i in range(batch_size):
        img = images_npy[i]  # Original unnormalized image (512, 512, 3)
        pred_mask = preds_list[i]  # (512, 512)

        # Prepare image for display
        img_vis = img.copy()
        # Normalize to [0, 1] for display
        img_vis = (img_vis - img_vis.min()) / (
            img_vis.max() - img_vis.min() + 1e-8
        )

        # Row 0: Original image
        axes[0, i].imshow(img_vis)
        axes[0, i].set_title(
            f"{filenames[i]}",
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
