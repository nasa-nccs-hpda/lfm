import json
import logging
import os
import sys
import time
import shutil
from glob import glob
from pathlib import Path
from contextlib import redirect_stdout, nullcontext
from functools import partial
from collections import Counter
from io import StringIO
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, get_logger

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.crs import CRS
from tqdm import tqdm
from rioxarray.merge import merge_arrays

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

repo_dir = "lfm"
repo_path = Path(os.getcwd()) / repo_dir
sys.path.append(str(repo_path.parent))
from lfm.model.Pipeline import Pipeline

mp.set_start_method('spawn', force=True)

COMMON_NODATA = -3.40282265508890445e+38

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(level=logging.INFO):
    """Configure logging for main process - outputs to stdout for SLURM"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def get_worker_logger():
    """Get logger for worker processes"""
    if mp.current_process().name != 'MainProcess':
        # Worker process - use multiprocessing logger that writes to stderr
        return get_logger()
    else:
        # Main process - use standard logger
        return logging.getLogger(__name__)


# ============================================================================
# WORKER INITIALIZATION
# ============================================================================
def init_worker():
    """Initialize worker process with GDAL environment variables"""
    import os

    # Set GDAL environment variables to prevent memory corruption issues
    # These settings help GDAL work safely in multiprocessing environments

    # Limit GDAL's internal cache (default is 5% of RAM, can cause issues)
    os.environ['GDAL_CACHEMAX'] = '512'  # 512 MB per worker

    # Disable threaded reads (can conflict with multiprocessing)
    os.environ['GDAL_NUM_THREADS'] = '1'

    # Disable directory reading optimizations that can cause issues
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

    # Set VSI cache size (for virtual file system operations)
    os.environ['VSI_CACHE'] = 'FALSE'
    os.environ['VSI_CACHE_SIZE'] = '0'

    # Disable GDAL's internal locking (we're using spawn method already)
    os.environ['GDAL_DISABLE_CPLLOCKTYPE'] = 'YES'

    # Set CPL_TMPDIR if needed (optional, for worker-specific temp dirs)
    # os.environ['CPL_TMPDIR'] = f'/tmp/gdal_worker_{os.getpid()}'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_product_id(train_filename: str) -> str:
    """
    Extract product ID from training filename.

    Example:
        'M1308401680CE_r12750_c1500_input.tif' -> 'M1308401680CE'

    Args:
        train_filename: Training sample filename

    Returns:
        Product ID (everything before first underscore)
    """
    basename = os.path.basename(train_filename)
    product_id = basename.split('_')[0]
    return product_id


def get_memory_usage():
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in MB, or None if psutil not available
    """
    if not HAS_PSUTIL:
        return None

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
    except Exception:
        return None


# ============================================================================
# DATACUBE PROCESSING (PIPELINE)
# ============================================================================
def run_pipeline_for_sample(
    train_fn: str,
    product_id: str,
    geom_bounds: tuple,
    datacube_dir: Path,
    TILE_DB_PATH: Path,
    logger,
    zoom_level,
) -> tuple:
    """
    Run Pipeline for a single training sample.

    Args:
        train_fn: Training sample filename
        product_id: Product ID to filter datacubes
        geom_bounds: Tuple of (ulLon, lrLat, lrLon, ulLat)
        datacube_dir: Output directory for this sample's datacubes
        TILE_DB_PATH: Path to tile database shapefile
        rm: Resampling method
        logger: Logger instance

    Returns:
        (cube_files: list, status: str)
    """
    # Unpack geometry bounds
    ulLon, lrLat, lrLon, ulLat = geom_bounds

    # Create datacube output directory
    datacube_dir.mkdir(parents=True, exist_ok=True)

    cube_files = []
    status = "success"

    try:
        # Redirect all Pipeline stdout to stderr (Option A from earlier discussion)
        # This keeps Pipeline's print() output in the .err file
        with redirect_stdout(sys.stderr):
            # Create Pipeline instance with product ID filtering
            pipeline = Pipeline(
                TILE_DB_PATH,
                datacube_dir,
                debug=False,
                targetProductID=product_id  # Only process matching product ID
            )
            print(f'Created pipeline instance for PID: {product_id}')

            # Run pipeline for this sample's bbox
            cube_files = pipeline.run(
                ulLat,
                ulLon,
                lrLat,
                lrLon,
                zoom_level
            )

        # Check if any cubes were created
        if not cube_files:
            logger.warning(f"  No datacubes created for {train_fn}")
            status = "warning_no_cubes"

    except Exception as e:
        # Log error with context
        logger.error(f"  Pipeline failed for {train_fn}: {e}")
        logger.error(f"    Product ID: {product_id}")
        logger.error(f"    Bounds: {geom_bounds}")
        status = "error_pipeline"
        # Don't raise - continue processing other samples

    return cube_files, status


# ============================================================================
# POSTPROCESSING (MERGE/REPROJECT/CLIP)
# ============================================================================
def group_cubes_by_tile(datacubes: list, logger) -> tuple:
    """
    Group datacubes by tile ID.

    Args:
        datacubes: List of datacube file paths
        logger: Logger instance

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
            "wac": [f for f in wac_datacubes if tile_id in f],
            "static": [f for f in static_datacubes if tile_id in f]
        }
        for tile_id in tile_ids
    }

    return cubes_by_tile, ltm_dict


def get_band_names(filepath: Path, band_indices: list = None) -> list:
    """
    Extract band names from raster file.

    Args:
        filepath: Path to raster file
        band_indices: Optional list of specific band indices (1-indexed)

    Returns:
        List of band names from 'Name' tag
    """
    import rasterio

    band_names = []
    try:
        with rasterio.open(filepath) as src:
            if band_indices is None:
                band_indices = range(1, src.count + 1)

            for band_idx in band_indices:
                tags = src.tags(band_idx)
                band_name = tags.get('Name', f'band_{band_idx}')
                band_names.append(band_name)

        return band_names
    except Exception as e:
        # Return default names on error
        return [f'band_{idx}' for idx in (band_indices or [])]


def check_bands_exist(filepath: Path, static_band_list: list) -> tuple:
    """
    Check if bands matching the static band list exist.

    Args:
        filepath: Path to raster file
        static_band_list: List of exact band names to match

    Returns:
        (exists: bool, band_numbers: list, error_message: str or None)
    """
    try:
        with rasterio.open(filepath) as src:
            # Create a mapping of band names to indices
            band_name_to_idx = {}
            for band_idx in range(1, src.count + 1):
                tags = src.tags(band_idx)
                band_name = tags.get('Name', '')
                if band_name:
                    band_name_to_idx[band_name] = band_idx

            # Check all required bands exist and preserve order
            band_indices = []
            missing_bands = []
            for band_name in static_band_list:
                if band_name in band_name_to_idx:
                    band_indices.append(band_name_to_idx[band_name])
                else:
                    missing_bands.append(band_name)

            if missing_bands:
                return False, [], f"Missing bands: {missing_bands}"

            return True, band_indices, None

    except Exception as e:
        return False, [], f"Error reading file: {str(e)}"


def merge_and_reproject_datasets(
    wac_files: list,
    static_files: list,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
    static_band_list: list,
    logger
) -> tuple:
    """
    Merge tiles and reproject to target grid.

    Args:
        wac_files: List of WAC datacube files
        static_files: List of static datacube files
        target_crs: Target CRS
        target_transform: Target affine transform
        target_height: Target height in pixels
        target_width: Target width in pixels
        static_band_list: list of exact static band names to use
        logger: Logger instance

    Returns:
        (merged_wac: xr.DataArray, merged_static: xr.DataArray,
         wac_band_names: list, static_band_names: list, status: str)
    """

    try:
        # ====================================================================
        # Load and filter WAC datasets
        # ====================================================================
        wac_datasets = []
        for wac_file in wac_files:
            ds = rxr.open_rasterio(wac_file)
            wac_datasets.append(ds)

        # Extract WAC band names from first file
        wac_band_names_original = get_band_names(wac_files[0])

        # ====================================================================
        # Load and filter static datasets
        # ====================================================================
        static_datasets = []
        static_band_nums = None

        for static_file in static_files:
            # Check if required bands exist
            bands_exist, band_nums, error_msg = check_bands_exist(
                static_file, static_band_list
            )
            if not bands_exist:
                logger.error(f"  Static bands not found: {error_msg}")
                return None, None, None, None, "error_static_bands_missing"

            # Store band numbers from first file
            if static_band_nums is None:
                static_band_nums = band_nums

            # Verify all files have same bands
            if band_nums != static_band_nums:
                logger.error(f"  Static band mismatch across files")
                return None, None, None, None, "error_static_band_mismatch"

            # Load and select matching bands
            ds = rxr.open_rasterio(static_file)
            ds_filtered = ds.sel(band=band_nums, drop=False)
            static_datasets.append(ds_filtered)

        # Extract static band names
        static_band_names_original = get_band_names(static_files[0], band_indices=static_band_nums)

        # ====================================================================
        # Merge tiles
        # ====================================================================
        logger.info("  Merging WAC tiles...")
        merged_wac = merge_arrays(wac_datasets, method='last')

        logger.info("  Merging STATIC tiles...")
        merged_static = merge_arrays(static_datasets, method='last')

        # ====================================================================
        # Reproject to target grid
        # ====================================================================
        logger.info("  Reprojecting WAC to target CRS...")
        merged_wac_reproj = merged_wac.rio.reproject(
            dst_crs=target_crs,
            transform=target_transform,
            shape=(target_height, target_width),
            resampling=Resampling.cubic
        )

        logger.info("  Reprojecting STATIC to target CRS...")
        merged_static_reproj = merged_static.rio.reproject(
            dst_crs=target_crs,
            transform=target_transform,
            shape=(target_height, target_width),
            resampling=Resampling.cubic
        )

        # Close original datasets
        for ds in wac_datasets:
            ds.close()
        for ds in static_datasets:
            ds.close()
        merged_wac.close()
        merged_static.close()

        return (merged_wac_reproj, merged_static_reproj,
                wac_band_names_original, static_band_names_original, "success")

    except Exception as e:
        logger.error(f"  Error in merge/reproject: {e}")
        return None, None, None, None, f"error_merge_reproject: {e}"


def clip_and_combine_datasets(
    merged_wac,
    merged_static,
    wac_band_names: list,
    static_band_names: list,
    train_bbox: tuple,
    train_shape: tuple,
    logger
) -> tuple:
    """
    Clip to AOI and combine WAC+STATIC.

    Args:
        merged_wac: Merged and reprojected WAC data
        merged_static: Merged and reprojected STATIC data
        wac_band_names: List of WAC band names
        static_band_names: List of STATIC band names
        train_bbox: Bounding box (minx, miny, maxx, maxy)
        train_shape: Target shape (bands, height, width)
        logger: Logger instance

    Returns:
        (combined: xr.DataArray, all_band_names: list, status: str)
    """

    try:
        # ====================================================================
        # Clip to training sample bbox
        # ====================================================================
        logger.info("  Clipping datasets to training AOI...")
        clipped_wac = merged_wac.rio.clip_box(*train_bbox)
        clipped_static = merged_static.rio.clip_box(*train_bbox)

        # Reorder WAC bands: VIS bands first (indices 2-6), then UV bands (0-1)
        clipped_wac_band_order = [2, 3, 4, 5, 6, 0, 1]
        clipped_wac = clipped_wac.isel(band=clipped_wac_band_order)
        wac_band_names = [wac_band_names[i] for i in clipped_wac_band_order]

        # ====================================================================
        # Validate shapes
        # ====================================================================
        expected_height, expected_width = train_shape[-2:]

        if clipped_wac.shape[-2:] != (expected_height, expected_width):
            logger.error(f"  WAC shape mismatch: {clipped_wac.shape[-2:]} != {(expected_height, expected_width)}")
            return None, None, "error_wac_shape_mismatch"

        if clipped_static.shape[-2:] != (expected_height, expected_width):
            logger.error(f"  STATIC shape mismatch: {clipped_static.shape[-2:]} != {(expected_height, expected_width)}")
            return None, None, "error_static_shape_mismatch"

        # ====================================================================
        # Normalize nodata and combine
        # ====================================================================
        logger.info("  Combining WAC and STATIC datasets...")

        # Normalize nodata values
        clipped_wac_normalized = clipped_wac.copy()
        clipped_static_normalized = clipped_static.copy()

        # Replace existing nodata with common value
        if clipped_wac.rio.nodata is not None:
            clipped_wac_normalized = clipped_wac_normalized.where(
                clipped_wac_normalized != clipped_wac.rio.nodata,
                COMMON_NODATA
            )
        if clipped_static.rio.nodata is not None:
            clipped_static_normalized = clipped_static_normalized.where(
                clipped_static_normalized != clipped_static.rio.nodata,
                COMMON_NODATA
            )

        # Set nodata where NaN values are found
        clipped_wac_normalized = clipped_wac_normalized.where(
            ~np.isnan(clipped_wac_normalized),
            COMMON_NODATA
        )
        clipped_static_normalized = clipped_static_normalized.where(
            ~np.isnan(clipped_static_normalized),
            COMMON_NODATA
        )

        # Set common nodata
        clipped_wac_normalized.rio.write_nodata(COMMON_NODATA, inplace=True)
        clipped_static_normalized.rio.write_nodata(COMMON_NODATA, inplace=True)

        # Concatenate along band dimension
        combined = xr.concat([clipped_wac_normalized, clipped_static_normalized], dim='band')
        combined.rio.write_nodata(COMMON_NODATA, inplace=True)

        # Combine band names
        all_band_names = wac_band_names + static_band_names

        logger.info(f"  Combined shape: {combined.shape}, bands: {len(all_band_names)}")

        return combined, all_band_names, "success"

    except Exception as e:
        logger.error(f"  Error in clip/combine: {e}")
        return None, None, f"error_clip_combine: {e}"


def write_chip_to_tif(
    combined,
    output_filename: Path,
    all_band_names: list,
    logger,
    nodata_value=COMMON_NODATA
) -> str:
    """
    Write combined chip to GeoTIFF with band descriptions.

    Args:
        combined: Combined xarray DataArray
        output_filename: Output file path
        all_band_names: List of band names
        logger: Logger instance
        nodata_value: Nodata value to use

    Returns:
        Status string
    """
    try:
        # Replace NaN with nodata value before writing
        combined_array = combined.values.copy()
        combined_array[np.isnan(combined_array)] = nodata_value

        with rasterio.open(
            output_filename,
            'w',
            driver='GTiff',
            height=combined.shape[1],
            width=combined.shape[2],
            count=combined.shape[0],
            dtype=combined.dtype,
            crs=combined.rio.crs,
            transform=combined.rio.transform(),
            nodata=nodata_value,
            compress='LZW'
        ) as dst:
            # Write data
            dst.write(combined_array)

            # Set band descriptions
            for i, band_name in enumerate(all_band_names, start=1):
                dst.set_band_description(i, band_name)

        return "success"

    except Exception as e:
        logger.error(f"  Error writing chip: {e}")
        return f"error_write: {e}"


# ============================================================================
# MAIN WORKER FUNCTION
# ============================================================================
def process_train_sample(
    entry: dict,
    TILE_DB_PATH: Path,
    CHIP_DIR: Path,
    datacube_base_dir: Path,
    chip_output_dir: Path,
    band_regex: str,
    expected_static: int,
    zoom_level: int,
    verbose: bool = False,
) -> tuple:
    """
    Process a single training sample: Pipeline → filter → postprocess → save chip.

    Args:
        entry: Dict with 'location' and 'geometry' from GeoDataFrame row
        TILE_DB_PATH: Path to tile database shapefile
        CHIP_DIR: Path to existing training chips (for reference grid)
        datacube_base_dir: Base directory for datacubes (will create subdir per sample)
        chip_output_dir: Output directory for final chips
        band_regex: Regex pattern for static band filtering
        expected_static: Expected number of static bands
        verbose: If True, log detailed progress

    Returns:
        (output_files: list, status: str, timing: dict, memory_peak_mb: float)
    """
    logger = get_worker_logger()

    # Initialize timing and memory tracking
    timing = {}
    mem_start = get_memory_usage()
    mem_peak = mem_start if mem_start else 0

    overall_start = time.time()

    # Extract info from entry
    train_fn = os.path.basename(entry['location'])
    geom = entry['geometry']
    geom_bounds = geom.bounds  # (ulLon, lrLat, lrLon, ulLat)

    if verbose:
        logger.info(f"Processing: {train_fn}")

    try:
        # ====================================================================
        # Step 1: Extract product ID
        # ====================================================================
        product_id = extract_product_id(train_fn)

        if verbose:
            logger.info(f"  Product ID: {product_id}")

        # ====================================================================
        # Step 2: Create datacube subdirectory
        # ====================================================================
        train_fn_no_ext = train_fn.replace('.tif', '')
        datacube_dir = datacube_base_dir / train_fn_no_ext

        # ====================================================================
        # Step 3: Run Pipeline
        # ====================================================================
        if verbose:
            logger.info(f"  Running Pipeline...")

        pipeline_start = time.time()

        cube_files, pipeline_status = run_pipeline_for_sample(
            train_fn=train_fn,
            product_id=product_id,
            geom_bounds=geom_bounds,
            datacube_dir=datacube_dir,
            TILE_DB_PATH=TILE_DB_PATH,
            logger=logger,
            zoom_level=zoom_level,
        )

        timing['pipeline'] = time.time() - pipeline_start

        # Update peak memory
        mem_current = get_memory_usage()
        if mem_current:
            mem_peak = max(mem_peak, mem_current)

        # Check pipeline status
        if pipeline_status != "success":
            if verbose or pipeline_status.startswith("error"):
                logger.warning(f"  Pipeline status: {pipeline_status}")
            return [], pipeline_status, timing, mem_peak

        if not cube_files:
            if verbose:
                logger.warning(f"  No datacubes created")
            return [], "warning_no_cubes", timing, mem_peak

        if verbose:
            logger.info(f"  Pipeline created {len(cube_files)} datacube files")

        # ====================================================================
        # Step 4: Group datacubes by tile
        # ====================================================================
        datacubes = glob(str(datacube_dir / "*.tif"))
        cubes_by_tile, ltm_dict = group_cubes_by_tile(datacubes, logger)

        # Check for multiple LTM zones
        if len(ltm_dict['unique']) > 1:
            if verbose:
                logger.warning(f"  Multiple LTM zones detected: {ltm_dict['unique']}")
            return [], "skipped_multiple_ltm_zones", timing, mem_peak

        # Check tile count
        if len(cubes_by_tile) not in [1, 2, 4]:
            if verbose:
                logger.warning(f"  Unexpected tile count: {len(cubes_by_tile)}")
            return [], "skipped_unexpected_tile_count", timing, mem_peak

        # ====================================================================
        # Step 5: Match WAC and STATIC files
        # ====================================================================
        wac_files = []
        static_files = []

        for tile_id, elems in cubes_by_tile.items():
            wac_match = [f for f in elems['wac'] if f"ProdId-{product_id}" in f]
            static_match = [f for f in elems['static']]

            if not wac_match or len(wac_match) > 1:
                if verbose:
                    logger.warning(f"  WAC file mismatch for tile {tile_id}")
                return [], "skipped_wac_mismatch", timing, mem_peak

            if not static_match or len(static_match) > 1:
                if verbose:
                    logger.warning(f"  STATIC file mismatch for tile {tile_id}")
                return [], "skipped_static_mismatch", timing, mem_peak

            wac_files.append(wac_match[0])
            static_files.append(static_match[0])

        if verbose:
            logger.info(f"  Matched {len(wac_files)} WAC and {len(static_files)} STATIC files")

        # ====================================================================
        # Step 6: Load reference training chip for target grid
        # ====================================================================
        train_path = CHIP_DIR / train_fn
        train_ds = rxr.open_rasterio(train_path)

        target_crs = train_ds.rio.crs
        target_transform = train_ds.rio.transform()
        target_height = train_ds.shape[-2]
        target_width = train_ds.shape[-1]
        train_bbox = train_ds.rio.bounds()
        train_shape = train_ds.shape

        # ====================================================================
        # Step 7: Merge and reproject
        # ====================================================================
        if verbose:
            logger.info(f"  Merging and reprojecting datacubes...")

        postprocess_start = time.time()

        merged_wac, merged_static, wac_band_names, static_band_names, merge_status = \
            merge_and_reproject_datasets(
                wac_files=wac_files,
                static_files=static_files,
                target_crs=target_crs,
                target_transform=target_transform,
                target_height=target_height,
                target_width=target_width,
                band_regex=band_regex,
                logger=logger
            )

        if merge_status != "success":
            if verbose or merge_status.startswith("error"):
                logger.error(f"  Merge/reproject failed: {merge_status}")
            train_ds.close()
            return [], merge_status, timing, mem_peak

        # Update peak memory
        mem_current = get_memory_usage()
        if mem_current:
            mem_peak = max(mem_peak, mem_current)

        # ====================================================================
        # Step 8: Clip and combine
        # ====================================================================
        if verbose:
            logger.info(f"  Clipping and combining datasets...")

        combined, all_band_names, combine_status = clip_and_combine_datasets(
            merged_wac=merged_wac,
            merged_static=merged_static,
            wac_band_names=wac_band_names,
            static_band_names=static_band_names,
            train_bbox=train_bbox,
            train_shape=train_shape,
            logger=logger
        )

        if combine_status != "success":
            if verbose or combine_status.startswith("error"):
                logger.error(f"  Clip/combine failed: {combine_status}")
            merged_wac.close()
            merged_static.close()
            train_ds.close()
            return [], combine_status, timing, mem_peak

        # ====================================================================
        # Step 9: Write chip to disk
        # ====================================================================
        output_filename = chip_output_dir / f"{train_fn_no_ext}_wac_static_chip.tif"

        if verbose:
            logger.info(f"  Writing chip to {output_filename.name}...")

        write_status = write_chip_to_tif(
            combined=combined,
            output_filename=output_filename,
            all_band_names=all_band_names,
            logger=logger,
            nodata_value=COMMON_NODATA
        )

        timing['postprocess'] = time.time() - postprocess_start

        # Update peak memory
        mem_current = get_memory_usage()
        if mem_current:
            mem_peak = max(mem_peak, mem_current)

        # Clean up
        combined.close()
        merged_wac.close()
        merged_static.close()
        train_ds.close()

        # ====================================================================
        # Finalize
        # ====================================================================
        timing['total'] = time.time() - overall_start

        if write_status == "success":
            return [output_filename], "success", timing, mem_peak
        else:
            return [], write_status, timing, mem_peak

    except Exception as e:
        # Catch-all error handler
        logger.error(f"  Unexpected error processing {train_fn}: {e}")
        timing['total'] = time.time() - overall_start
        return [], f"error_unexpected: {e}", timing, mem_peak


# ============================================================================
# MULTIPROCESSING ORCHESTRATION
# ============================================================================
def create_chips_multiprocessing(
    train_gdf: gpd.GeoDataFrame,
    TILE_DB_PATH: Path,
    CHIP_DIR: Path,
    datacube_base_dir: Path,
    chip_output_dir: Path,
    band_regex: str,
    expected_static: int,
    zoom_level: int,
    max_workers: int = None,
    max_entries: int = None,
    verbose: bool = False
) -> list:
    """
    Process all training samples with multiprocessing.

    Args:
        train_gdf: GeoDataFrame with training samples
        TILE_DB_PATH: Path to tile database
        CHIP_DIR: Path to existing training chips
        datacube_base_dir: Base directory for intermediate datacubes
        chip_output_dir: Output directory for final chips
        band_regex: Regex for static band filtering
        expected_static: Expected number of static bands
        max_workers: Number of worker processes (None for serial)
        max_entries: Limit number of samples (for testing)
        verbose: Verbose logging

    Returns:
        list of (output_files, status, timing, memory) tuples
    """
    logger = logging.getLogger(__name__)

    # Convert GeoDataFrame rows to dicts for multiprocessing
    entries = []
    for idx, row in train_gdf.iterrows():
        entry = row.to_dict()
        entry["index"] = idx
        entries.append(entry)

    # Limit entries if specified
    if max_entries:
        entries = entries[:max_entries]

    # Create partial function with fixed parameters
    process_func = partial(
        process_train_sample,
        TILE_DB_PATH=TILE_DB_PATH,
        CHIP_DIR=CHIP_DIR,
        datacube_base_dir=datacube_base_dir,
        chip_output_dir=chip_output_dir,
        band_regex=band_regex,
        expected_static=expected_static,
        zoom_level=zoom_level,
        verbose=verbose
    )

    # Set up multiprocessing or serial processing
    if max_workers and max_workers > 1:
        logger.info(f"Using multiprocessing with {max_workers} workers")
        pool_context = Pool(max_workers, initializer=init_worker)
    else:
        logger.info("Using serial processing")
        pool_context = nullcontext()

    # Initialize counters
    success_count = 0
    skipped_count = 0
    error_count = 0
    results = []

    # Process samples
    with pool_context as pool:
        if max_workers and max_workers > 1:
            # Multiprocessing with tqdm
            mapper = pool.imap(process_func, entries)
        else:
            # Serial processing
            mapper = map(process_func, entries)

        # Wrap with tqdm and update counts dynamically
        with tqdm(total=len(entries), desc="Processing samples", file=sys.stdout) as pbar:
            for result in mapper:
                # Extract status
                _, status, _, _ = result
                results.append(result)

                # Update counters based on status
                if status == "success":
                    success_count += 1
                elif status.startswith("error"):
                    error_count += 1
                else:
                    skipped_count += 1

                # Update progress bar with counts
                pbar.set_postfix({
                    'success': success_count,
                    'skipped': skipped_count,
                    'errors': error_count
                })
                pbar.update(1)

    return results

# ============================================================================
# MAIN
# ============================================================================
def create_chips(band_regex=r"^lola_kaguya.*", expected_static=5, zoom_level=5, working_dir="lfm_train_chips"):
    # Set up logging
    logger = setup_logging()

    # ========================================================================
    # PATHS
    # ========================================================================
    PROJECT_DIR = Path("/explore/nobackup/projects/lfm")
    DATA_DIR = PROJECT_DIR / "processed_data/Lunar/"
    WAC_DIR = DATA_DIR / "LRO_WAC_Pho_Sites"

    # Training paths
    TRAIN_DIR = PROJECT_DIR / "model_inputs/300_300_inputs/7_band_vis_uv/sem_seg"
    CHIP_DIR = TRAIN_DIR / "chips"
    GPKG_PATH = CHIP_DIR / "WAC_TILES.gpkg"

    # Tile database
    TILE_DB_PATH = WAC_DIR / "output_index.shp"

    # Output directories
    working_dir = Path(working_dir)
    datacube_base_dir = working_dir / "datacubes"
    chip_output_dir = working_dir / "chips"

    # Create output directories
    working_dir.mkdir(parents=True, exist_ok=True)
    datacube_base_dir.mkdir(parents=True, exist_ok=True)
    chip_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LUNAR TRAINING DATA - FULL WORKFLOW (PIPELINE + POSTPROCESSING)")
    logger.info("=" * 80)
    logger.info(f"Output directories:")
    logger.info(f"  Datacubes (intermediate): {datacube_base_dir}")
    logger.info(f"  Chips (final): {chip_output_dir}")

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    MAX_WORKERS = 16  # Test with 8 workers
    MAX_ENTRIES = None  # Process first 8 samples
    VERBOSE = False  # Set True for detailed worker logging

    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Workers: {MAX_WORKERS}")
    logger.info(f"  Max entries: {MAX_ENTRIES}")
    logger.info(f"  Verbose: {VERBOSE}")
    logger.info(f"  Expected static bands: {expected_static}")
    logger.info(f"  Zoom level: {zoom_level}")
    logger.info("=" * 80)

    # ========================================================================
    # LOAD TRAINING DATA
    # ========================================================================
    logger.info("Loading training GeoDataFrame...")
    train_gdf = gpd.read_file(GPKG_PATH)
    logger.info(f"  Total samples in GeoDataFrame: {len(train_gdf)}")

    # Limit to first MAX_ENTRIES for testing
    if MAX_ENTRIES:
        train_gdf = train_gdf.head(MAX_ENTRIES)
        logger.info(f"  Limited to first {MAX_ENTRIES} samples for testing")

    # ========================================================================
    # PROCESS SAMPLES
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Starting processing...")
    logger.info("=" * 80)

    start_time = time.time()

    results = create_chips_multiprocessing(
        train_gdf=train_gdf,
        TILE_DB_PATH=TILE_DB_PATH,
        CHIP_DIR=CHIP_DIR,
        datacube_base_dir=datacube_base_dir,
        chip_output_dir=chip_output_dir,
        band_regex=band_regex,
        expected_static=expected_static,
        zoom_level=zoom_level,
        max_workers=MAX_WORKERS,
        max_entries=MAX_ENTRIES,
        verbose=VERBOSE
    )

    total_time = time.time() - start_time

    # ========================================================================
    # REPORT RESULTS
    # ========================================================================
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)

    # Extract statuses
    statuses = [status for _, status, _, _ in results]
    status_counts = Counter(statuses)

    logger.info("Status Summary:")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")

    # Timing statistics
    timings = [timing for _, _, timing, _ in results if timing]
    if timings:
        total_pipeline_time = sum(t.get('pipeline', 0) for t in timings)
        total_postprocess_time = sum(t.get('postprocess', 0) for t in timings)

        logger.info("\nTiming Statistics:")
        logger.info(f"  Total wall time: {total_time:.2f}s")
        logger.info(f"  Total pipeline time: {total_pipeline_time:.2f}s")
        logger.info(f"  Total postprocess time: {total_postprocess_time:.2f}s")
        logger.info(f"  Avg per sample: {total_time/len(results):.2f}s")

    # Memory statistics
    if HAS_PSUTIL:
        memory_peaks = [mem for _, _, _, mem in results if mem]
        if memory_peaks:
            logger.info("\nMemory Statistics:")
            logger.info(f"  Peak memory per worker (avg): {np.mean(memory_peaks):.2f} MB")
            logger.info(f"  Peak memory per worker (max): {np.max(memory_peaks):.2f} MB")
            logger.info(f"  Peak memory per worker (min): {np.min(memory_peaks):.2f} MB")

    logger.info("=" * 80)
    logger.info("Workflow complete!")