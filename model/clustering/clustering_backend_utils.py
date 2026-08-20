# PROJ must be configured before rasterio/localtileserver are imported.
import os
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

from pathlib import Path
import sys
import ipysheet
from IPython.display import Markdown, display
import ipywidgets
import leafmap
import numpy as np
import pandas
import rasterio
from rasterio.windows import Window
from localtileserver import TileClient, get_leaflet_tile_layer
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from ipyleaflet import WidgetControl

repo_root = Path.cwd().parent
repo_root_str = str(repo_root).replace('/panfs/ccds02/nobackup', '/explore/nobackup')
repo_root = Path(repo_root_str)
NOTEBOOK_DIR = repo_root / "notebooks"

if not (repo_root / "lfm").exists():
  raise FileNotFoundError(
      "Cannot find lfm/ directory. Run this notebook from "
      "lfm/notebooks/full_model or update repo_root."
  )

sys.path.insert(0, str(repo_root))

from model.clustering.Clusterer import Clusterer
from model.clustering.ImageHelperSingleBand import ImageHelper

def crop_center(src_path: Path, dst_path: Path, size: int = 512) -> Path:
    """Write a centered square crop while preserving CRS/georeferencing."""
    with rasterio.open(src_path) as src:
        if src.width < size or src.height < size:
            raise ValueError(
                f"Raster is only {src.width}x{src.height}; "
                f"cannot make a {size}x{size} crop."
            )

        col_off = (src.width - size) // 2
        row_off = (src.height - size) // 2
        window = Window(col_off, row_off, size, size)

        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            width=size,
            height=size,
            transform=transform,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return dst_path


def handleClick(change: dict, output, sl, bt, opts: list, table: dict) -> None:
    with output:
        if change.new == "Next":
            if not sl.value:
                print("Select at least one cluster before clicking Next.")
                bt.value = "Select:"
                return

            nn = updateList(list(sl.options), list(sl.value))
            updateDict("N", sl, table)
            sl.options = nn
            bt.value = "Select:"

        if change.new == "Done":
            updateDict("D", sl, table)

        if change.new == "Start Over":
            sl.options = opts
            updateDict("S", sl, table)
            bt.value = "Select:"


def relabel(labelArray: np.ndarray, lookup: dict) -> np.ndarray:
    newLab = labelArray.copy()

    for k, v in lookup.items():
        if len(v) == 1 and k == v[0]:
            continue
        newLab = np.where(np.isin(newLab, v), k, newLab)

    return newLab


def updateDict(op: str, sl, table: dict) -> None:
    if op == "N":
        key = list(sl.value)[0]
        table[key] = list(sl.value)

    if op == "D":
        if len(sl.options) > 0:
            key = list(sl.options)[0]
            table[key] = list(sl.options)

        print("Final Groups : ", table)

    if op == "S":
        table.clear()


def updateList(old: list, out: list) -> list:
    return [ele for ele in old if ele not in out]