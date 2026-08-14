#!/usr/bin/env python
# coding: utf-8

# # Imports, config

# In[ ]:


import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
import rioxarray as rxr
import rasterio
import numpy as np


# In[ ]:


BASE_DIR = Path("/explore/nobackup/projects/lfm/model_inputs").resolve()
DATA_DIR = (BASE_DIR / "300_300_inputs/fm_all_static_all_wac_iseg").resolve()
OLD_DATA_DIR = (BASE_DIR / "300_300_inputs/all_static_all_wac/inst_seg/chips").resolve()

STATIC_BANDS_0IDX = np.arange(7, 70)
STATIC_BANDS_RASTERIO = (STATIC_BANDS_0IDX + 1).tolist()

NUM_STATIC_CHANNELS = len(STATIC_BANDS_0IDX)

print(NUM_STATIC_CHANNELS)  # 63

MAX_WORKERS = 10


# In[ ]:


example_file = next(DATA_DIR.glob(f"train/chips/*.tif"))
example_file_ds = rxr.open_rasterio(example_file)
len(example_file_ds.band)


# # Sanity check number of channels

# In[ ]:


def get_num_channels(file_name):
    with rasterio.open(file_name) as src:
        return src.count


# In[ ]:


splits = ['train', 'val', 'test']
EXPECTED_CHANNELS = 70

for split in splits:
    file_list = list(DATA_DIR.glob(f"{split}/chips/*.tif"))

    failed_files = []
    incorrect_shapes = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        future_to_file = {
            executor.submit(get_num_channels, file_name): file_name
            for file_name in file_list
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc=f"Checking {split}",
        ):
            file_name = future_to_file[future]

            try:
                num_channels = future.result()

                if num_channels != EXPECTED_CHANNELS:
                    incorrect_shapes.append(
                        (file_name, num_channels)
                    )

            except Exception as exc:
                failed_files.append(file_name)

                print("\n" + "=" * 80)
                print("FAILED")
                print(f"Chip: {file_name}")
                print(f"Exception: {type(exc).__name__}: {exc}")
                print("=" * 80)
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__
                )

    print(f"\n{split}:")
    print(f"  Total:             {len(file_list)}")
    print(f"  Incorrect channels: {len(incorrect_shapes)}")
    print(f"  Failed to read:     {len(failed_files)}")

    for file_name, num_channels in incorrect_shapes:
        print(
            f"  BAD: {file_name.name}: "
            f"{num_channels} channels, expected {EXPECTED_CHANNELS}"
        )


# # Sanity check value counts + nodata

# ## 8/11 - initial checking of "problem values" in dataset

# In[ ]:


def inspect_chip(file_name):
    """
    Count invalid pixels in each static band for one chip.

    Returns arrays of shape (NUM_STATIC_CHANNELS,) containing
    pixel counts for each invalid-value category.
    """
    with rasterio.open(file_name) as src:
        nodata = src.nodata

        data = src.read(
            indexes=STATIC_BANDS_RASTERIO,
            masked=False,
        ).astype(np.float64)

    # Shape: (n_bands, height, width)

    n_nan = np.isnan(data).sum(axis=(1, 2))
    n_posinf = np.isposinf(data).sum(axis=(1, 2))
    n_neginf = np.isneginf(data).sum(axis=(1, 2))

    if nodata is None:
        n_nodata = np.zeros(data.shape[0], dtype=np.int64)
    else:
        n_nodata = (data == nodata).sum(axis=(1, 2))

    return {
        "nan": n_nan.astype(np.int64),
        "nodata": n_nodata.astype(np.int64),
        "posinf": n_posinf.astype(np.int64),
        "neginf": n_neginf.astype(np.int64),
    }


# In[ ]:


def inspect_dataset(file_list, dataset_name):
    totals = {
        "nan": np.zeros(NUM_STATIC_CHANNELS, dtype=np.int64),
        "nodata": np.zeros(NUM_STATIC_CHANNELS, dtype=np.int64),
        "posinf": np.zeros(NUM_STATIC_CHANNELS, dtype=np.int64),
        "neginf": np.zeros(NUM_STATIC_CHANNELS, dtype=np.int64),
    }

    failed_files = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(inspect_chip, file_name): file_name
            for file_name in file_list
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc=f"Checking {dataset_name}",
        ):
            file_name = future_to_file[future]

            try:
                result = future.result()

                for problem_type in totals:
                    totals[problem_type] += result[problem_type]

            except Exception as exc:
                failed_files.append(file_name)

                print("\n" + "=" * 80)
                print(f"FAILED: {file_name}")
                print(f"{type(exc).__name__}: {exc}")
                print("=" * 80)

                traceback.print_exception(
                    type(exc),
                    exc,
                    exc.__traceback__,
                )

    # Convert arrays into an easy-to-inspect per-band dictionary
    band_stats = {}

    for i, band_idx in enumerate(STATIC_BANDS_0IDX):
        band_stats[int(band_idx)] = {
            "nan": int(totals["nan"][i]),
            "nodata": int(totals["nodata"][i]),
            "neginf": int(totals["neginf"][i]),
            "posinf": int(totals["posinf"][i]),
        }

    return band_stats, failed_files


# In[ ]:


old_file_list = list(
    OLD_DATA_DIR.glob("*.tif")
)

current_file_list = list(
    DATA_DIR.glob("train/chips/*.tif")
)


old_stats, old_failed = inspect_dataset(
    old_file_list,
    "old dataset",
)

current_stats, current_failed = inspect_dataset(
    current_file_list,
    "current dataset",
)


# In[ ]:


def print_problem_bands(stats, name):
    print(f"\n{name}")
    print("=" * 70)

    for band_idx, counts in stats.items():
        total_problems = sum(counts.values())

        if total_problems:
            print(
                f"Band {band_idx:2d}: "
                f"NaN={counts['nan']:,}, "
                f"nodata={counts['nodata']:,}, "
                f"-inf={counts['neginf']:,}, "
                f"+inf={counts['posinf']:,}"
            )


print_problem_bands(old_stats, "OLD DATA")
print_problem_bands(current_stats, "CURRENT DATA")


# ## 8/12: initial source dataset checks

# In[ ]:


import numpy as np
import rasterio


source_file = (
    "/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/"
    "mini_rf/GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif"
)

counts = {
    "nodata": 0,
    "nan": 0,
    "neginf": 0,
    "posinf": 0,
}

with rasterio.open(source_file) as src:
    nodata = src.nodata

    print("dtype:", src.dtypes[0])
    print("nodata:", nodata)

    windows = list(src.block_windows(1))

    for _, window in tqdm(windows):
        data = src.read(
            1,
            window=window,
            masked=False,
        )

        counts["nodata"] += np.count_nonzero(data == nodata)
        counts["nan"] += np.count_nonzero(np.isnan(data))
        counts["neginf"] += np.count_nonzero(np.isneginf(data))
        counts["posinf"] += np.count_nonzero(np.isposinf(data))

print(counts)


# In[ ]:


import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


source_file = (
    "/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/"
    "mini_rf/GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif"
)

# Histogram range
HIST_MIN = -0.6
HIST_MAX = 3.0
N_BINS = 200

bin_edges = np.linspace(HIST_MIN, HIST_MAX, N_BINS + 1)
hist_counts = np.zeros(N_BINS, dtype=np.int64)

total_valid = 0
total_nodata = 0
total_nonfinite = 0

# Track actual valid data range
global_min = np.inf
global_max = -np.inf

# Optional: track valid pixels that fall outside histogram range
below_hist_range = 0
above_hist_range = 0


with rasterio.open(source_file) as src:
    nodata = src.nodata

    windows = list(src.block_windows(1))

    for _, window in tqdm(windows, desc="Computing histogram"):
        data = src.read(
            1,
            window=window,
            masked=False,
        )

        nodata_mask = data == nodata
        finite_mask = np.isfinite(data)

        valid = finite_mask & ~nodata_mask

        total_nodata += np.count_nonzero(nodata_mask)
        total_nonfinite += np.count_nonzero(~finite_mask)

        values = data[valid]

        if values.size == 0:
            continue

        total_valid += values.size

        # Track actual valid range
        global_min = min(global_min, float(values.min()))
        global_max = max(global_max, float(values.max()))

        # Track values outside chosen histogram bounds
        below_hist_range += np.count_nonzero(values < HIST_MIN)
        above_hist_range += np.count_nonzero(values > HIST_MAX)

        counts, _ = np.histogram(
            values,
            bins=bin_edges,
        )

        hist_counts += counts


print(f"Valid pixels:     {total_valid:,}")
print(f"Nodata pixels:    {total_nodata:,}")
print(f"Nonfinite pixels: {total_nonfinite:,}")

print()
print(f"Actual valid min: {global_min}")
print(f"Actual valid max: {global_max}")

print()
print(f"Below histogram range (< {HIST_MIN}): {below_hist_range:,}")
print(f"Above histogram range (> {HIST_MAX}): {above_hist_range:,}")


# In[ ]:


plt.figure(figsize=(12, 5))

plt.bar(
    bin_centers,
    hist_counts,
    width=np.diff(bin_edges),
)

plt.xlim(-0.6, 0.0)
plt.yscale("log")

plt.xlabel("CPR value")
plt.ylabel("Pixel count (log scale)")
plt.title("Source CPR Distribution — Low Values")

plt.show()


# ## 8/12: saving 2 samples to look at in QGIS

# In[ ]:


from pathlib import Path

import numpy as np
import rasterio

from rasterio.windows import Window
from tqdm.auto import tqdm


SOURCE_FILE = Path(
    "/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/"
    "mini_rf/GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif"
)

OUTPUT_DIR = Path("./qgis_nodata_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 1000


# In[ ]:


def get_nodata_stats(src, window, grid_size=5):
    """
    Calculate nodata statistics for one raster window.

    Returns
    -------
    nodata_fraction : float
        Fraction of pixels that are nodata.

    occupied_cells : int
        Number of grid cells containing at least one nodata pixel.
        For grid_size=5, the window is divided into 25 cells.

    total_cells : int
        Total number of grid cells.
    """
    mask = src.read_masks(
        1,
        window=window,
    )

    # Rasterio mask:
    #   0   = nodata / invalid
    #   255 = valid
    nodata = mask == 0

    nodata_fraction = nodata.mean()

    occupied_cells = 0

    row_chunks = np.array_split(nodata, grid_size, axis=0)

    for row_chunk in row_chunks:
        col_chunks = np.array_split(row_chunk, grid_size, axis=1)

        for cell in col_chunks:
            if np.any(cell):
                occupied_cells += 1

    total_cells = grid_size**2

    return nodata_fraction, occupied_cells, total_cells


def find_first_window(
    src,
    min_nodata,
    max_nodata,
    window_size=1000,
    min_occupied_cells=None,
    grid_size=5,
):
    """
    Search non-overlapping windows from:
        top-left -> right -> next row -> right ...

    Parameters
    ----------
    min_nodata, max_nodata : float
        Required nodata fraction.

    min_occupied_cells : int or None
        If provided, nodata must occur in at least this many spatial
        grid cells.

    grid_size : int
        Divide each candidate window into grid_size x grid_size cells
        when evaluating nodata distribution.

    Returns
    -------
    window : rasterio.windows.Window or None
    stats : dict or None
    """

    n_rows = src.height // window_size
    n_cols = src.width // window_size

    total_windows = n_rows * n_cols

    with tqdm(total=total_windows, desc="Searching windows") as pbar:

        for row_idx in range(n_rows):
            row_off = row_idx * window_size

            for col_idx in range(n_cols):
                col_off = col_idx * window_size

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=window_size,
                    height=window_size,
                )

                (
                    nodata_fraction,
                    occupied_cells,
                    total_cells,
                ) = get_nodata_stats(
                    src,
                    window,
                    grid_size=grid_size,
                )

                pbar.update(1)

                fraction_matches = (
                    min_nodata
                    <= nodata_fraction
                    <= max_nodata
                )

                distribution_matches = (
                    min_occupied_cells is None
                    or occupied_cells >= min_occupied_cells
                )

                if fraction_matches and distribution_matches:
                    stats = {
                        "nodata_fraction": nodata_fraction,
                        "occupied_cells": occupied_cells,
                        "total_cells": total_cells,
                        "row_offset": row_off,
                        "col_offset": col_off,
                    }

                    return window, stats

    return None, None


def export_window(src, window, output_file):
    """
    Export a source raster window as its own GeoTIFF while preserving
    the source CRS, transform, dtype, and nodata metadata.
    """
    data = src.read(
        1,
        window=window,
        masked=False,
    )

    profile = src.profile.copy()

    profile.update(
        height=int(window.height),
        width=int(window.width),
        transform=src.window_transform(window),
        count=1,
        compress="deflate",
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data, 1)

    return output_file


# In[ ]:


# SAMPLE 1: FIRST 25-75% NODATA SAMPLE

with rasterio.open(SOURCE_FILE) as src:

    mixed_window, mixed_stats = find_first_window(
        src,
        min_nodata=0.25,
        max_nodata=0.75,
        window_size=WINDOW_SIZE,
    )


if mixed_window is None:
    print("No 25-75% nodata window found.")

else:
    print("Found 25-75% nodata sample")
    print("--------------------------------")
    print("Window:", mixed_window)
    print(f"NoData: {mixed_stats['nodata_fraction']:.2%}")
    print(
        "Spatial cells containing nodata: "
        f"{mixed_stats['occupied_cells']}/"
        f"{mixed_stats['total_cells']}"
    )
    print(
        f"Pixel offset: "
        f"row={mixed_stats['row_offset']}, "
        f"col={mixed_stats['col_offset']}"
    )


# In[ ]:


# SAMPLE 2: SCATTERED NODATA, 0-15%

with rasterio.open(SOURCE_FILE) as src:

    scattered_window, scattered_stats = find_first_window(
        src,
        min_nodata=0.001,   # at least 0.1% nodata
        max_nodata=0.25,
        window_size=WINDOW_SIZE,
        min_occupied_cells=8,
        grid_size=5,
    )


if scattered_window is None:
    print("No scattered nodata window found.")

else:
    print("Found scattered nodata sample")
    print("--------------------------------")
    print("Window:", scattered_window)
    print(f"NoData: {scattered_stats['nodata_fraction']:.2%}")
    print(
        "Spatial cells containing nodata: "
        f"{scattered_stats['occupied_cells']}/"
        f"{scattered_stats['total_cells']}"
    )
    print(
        f"Pixel offset: "
        f"row={scattered_stats['row_offset']}, "
        f"col={scattered_stats['col_offset']}"
    )


# In[ ]:


# DOWNLOAD SAMPLES

with rasterio.open(SOURCE_FILE) as src:

    if mixed_window is not None:
        mixed_output = export_window(
            src,
            mixed_window,
            OUTPUT_DIR / "CPR_25_75pct_nodata.tif",
        )

        print(f"Exported: {mixed_output}")

    if scattered_window is not None:
        scattered_output = export_window(
            src,
            scattered_window,
            OUTPUT_DIR / "CPR_scattered_nodata.tif",
        )

        print(f"Exported: {scattered_output}")


# ## 8/12: getting static datacubes for comparison

# In[ ]:


from pathlib import Path
REPO_PATH = Path('/explore/nobackup/people/ajkerr1/Lunar_FM/full_model_lfm')
import sys

# Verify we're in the right location
if not (REPO_PATH / "lfm" / "model").exists():
    raise FileNotFoundError(
        "Cannot find lfm/model directory. "
        "Please ensure you're running this notebook from the lfm/notebooks/ directory."
    )

# Add the parent of the repo to sys.path for imports
sys.path.insert(0, str(REPO_PATH))

# Import required modules
from lfm.model.Pipeline import Pipeline
import shutil


def prepare_output_dir(output_dir: str | Path, delete_previous: bool = False) -> Path:
    output_dir = Path(output_dir)
    if delete_previous and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
print("✓ Successfully imported LFM modules")