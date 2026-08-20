# PROJ must be configured before rasterio/localtileserver are imported.
import os
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

from pathlib import Path
import sys
import ipywidgets
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

CRATER_CLASS_ID = 1
NON_CRATER_CLASS_ID = 0
CRATER_ASSIGNMENT = "Crater"
NON_CRATER_ASSIGNMENT = "Non-crater"
IGNORE_ASSIGNMENT = "Ignore"


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


def cluster_color(cluster_id: int, cluster_ids: list[int], colormap: str = "tab20") -> str:
    sorted_ids = sorted(int(i) for i in cluster_ids)
    vmin = min(sorted_ids)
    vmax = max(sorted_ids)
    t = 0.5 if vmax == vmin else (int(cluster_id) - vmin) / (vmax - vmin)
    return mcolors.to_hex(cm.get_cmap(colormap)(t))


def create_cluster_assignment_widget(labels: np.ndarray, colormap: str = "tab20"):
    cluster_ids = sorted(int(i) for i in np.unique(labels))
    controls = {}
    rows = []

    header = ipywidgets.HTML(
        value=(
            "<div style='font-weight:600;margin-bottom:8px;'>"
            "Assign each cluster to Crater, Non-crater, or Ignore. "
            "Crater is written as class 1; non-crater is written as class 0."
            "</div>"
        )
    )

    for cluster_id in cluster_ids:
        color = cluster_color(cluster_id, cluster_ids, colormap)
        swatch = ipywidgets.HTML(
            value=(
                "<div style='width:18px;height:14px;"
                f"background:{color};border:1px solid #444;"
                "margin-top:3px;'></div>"
            ),
            layout=ipywidgets.Layout(width="28px"),
        )
        label = ipywidgets.Label(
            value=f"Cluster {cluster_id}",
            layout=ipywidgets.Layout(width="90px"),
        )
        assignment = ipywidgets.ToggleButtons(
            options=[IGNORE_ASSIGNMENT, CRATER_ASSIGNMENT, NON_CRATER_ASSIGNMENT],
            value=IGNORE_ASSIGNMENT,
            layout=ipywidgets.Layout(width="320px"),
        )
        controls[cluster_id] = assignment
        rows.append(
            ipywidgets.HBox(
                [swatch, label, assignment],
                layout=ipywidgets.Layout(align_items="center", margin="2px 0"),
            )
        )

    widget = ipywidgets.VBox([header, *rows])
    return widget, controls


def get_binary_cluster_mapping(assignment_controls: dict) -> dict:
    crater_clusters = []
    non_crater_clusters = []

    for cluster_id, control in assignment_controls.items():
        if control.value == CRATER_ASSIGNMENT:
            crater_clusters.append(int(cluster_id))
        elif control.value == NON_CRATER_ASSIGNMENT:
            non_crater_clusters.append(int(cluster_id))

    return {
        NON_CRATER_CLASS_ID: sorted(non_crater_clusters),
        CRATER_CLASS_ID: sorted(crater_clusters),
    }


def summarize_binary_cluster_mapping(mapping: dict, labels: np.ndarray | None = None) -> None:
    crater_clusters = mapping.get(CRATER_CLASS_ID, [])
    non_crater_clusters = mapping.get(NON_CRATER_CLASS_ID, [])

    print(f"Crater class 1 clusters: {crater_clusters}")
    print(f"Non-crater class 0 clusters: {non_crater_clusters}")

    if not crater_clusters:
        print("No crater clusters selected yet.")
    if not non_crater_clusters:
        print("No non-crater clusters selected yet.")

    if labels is not None:
        assigned = set(crater_clusters) | set(non_crater_clusters)
        ignored = sorted(int(i) for i in np.unique(labels) if int(i) not in assigned)
        print(f"Ignored clusters: {ignored}")


def relabel(labelArray: np.ndarray, lookup: dict) -> np.ndarray:
    newLab = np.zeros(labelArray.shape, dtype=np.int32)

    assigned = set()
    for class_id, cluster_ids in lookup.items():
        if not cluster_ids:
            continue
        assigned.update(int(i) for i in cluster_ids)
        newLab = np.where(np.isin(labelArray, cluster_ids), int(class_id), newLab)

    return newLab


# Backward-compatible helpers retained for older notebooks.
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