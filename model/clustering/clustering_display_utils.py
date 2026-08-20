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

CLUSTER_COLORMAP = "tab20"
FINAL_LABEL_COLORMAP = "coolwarm"

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

def _create_display_legend(labels, colormap: str = CLUSTER_COLORMAP):
    # Unique cluster IDs, used to generate labels
    cluster_ids = sorted(int(i) for i in np.unique(labels))
    vmin = min(cluster_ids)
    vmax = max(cluster_ids)

    cmap = cm.get_cmap(colormap)

    rows = []
    for cid in cluster_ids:
        # Match the same colormap normalization used by the layer
        if vmax == vmin:
            t = 0.5
        else:
            t = (cid - vmin) / (vmax - vmin)

        hex_color = mcolors.to_hex(cmap(t))

        rows.append(
            f"""
            <div style="display:flex; align-items:center; margin:2px 0;">
                <div style="
                    width:18px;
                    height:12px;
                    background:{hex_color};
                    border:1px solid #444;
                    margin-right:8px;
                    flex:0 0 auto;
                "></div>
                <div style="font-size:12px;">Cluster {cid}</div>
            </div>
            """
        )

    legend_html = ipywidgets.HTML(
        value=f"""
        <div style="
            background:white;
            color: #111;
            padding:8px 10px;
            border:1px solid #777;
            border-radius:4px;
            max-height:300px;
            min-width:140px;
            overflow-y:auto;
            box-shadow:0 1px 4px rgba(0,0,0,0.25);
        ">
            <div style="font-weight:bold; margin-bottom:6px;">
                Cluster legend
            </div>
            {''.join(rows)}
        </div>
        """
    )
    return legend_html


def display_images_labels(
    inputFile: Path,
    labelsFile: Path,
    labels: np.array,
    inHelper,
    lHelper,
    colormap: str = CLUSTER_COLORMAP,
):
    # Use localtileserver directly for raster serving. This path works with the
    # Jupyter/VS Code loopback bridge and avoids leafmap.add_raster().
    image_client = TileClient(str(inputFile), debug=True)
    labels_client = TileClient(str(labelsFile), debug=True)

    image_layer = get_leaflet_tile_layer(
        image_client,
        vmin=inHelper._minValue,
        vmax=inHelper._maxValue,
        nodata=inHelper._noDataValue,
        opacity=1.0,
    )
    image_layer.name = inputFile.name

    labels_layer = get_leaflet_tile_layer(
        labels_client,
        vmin=lHelper._minValue,
        vmax=lHelper._maxValue,
        nodata=lHelper._noDataValue,
        opacity=0.5,
        colormap=colormap,
    )
    labels_layer.name = labelsFile.name

    # Don't let ipyleaflet restrict bounds, creates buggy output
    image_layer.bounds = None
    labels_layer.bounds = None

    m = leafmap.Map(
        fullscreen_control=False,
        layers_control=True,
        search_control=False,
        draw_control=False,
        measure_control=False,
        scale_control=False,
        toolbar_control=True,
        center=image_client.center(),
        zoom=image_client.default_zoom,
    )

    # Remove the default Earth/OpenStreetMap basemap.
    m.remove(m.layers[0])

    # Add image/labels
    m.add(image_layer)
    m.add(labels_layer)

    m.layout.height = "600px"

    legend_html = _create_display_legend(labels, colormap)
    legend_control = WidgetControl(widget=legend_html, position="topright")
    m.add(legend_control)
    return m, legend_control


def _create_binary_legend(newClusters, colormap: str = FINAL_LABEL_COLORMAP):
    # Final grouped class IDs
    class_ids = sorted(int(x) for x in np.unique(newClusters))

    vmin = min(class_ids)
    vmax = max(class_ids)

    cmap = cm.get_cmap(colormap)

    rows = []

    for class_id in class_ids:
        # Match the same vmin/vmax normalization used by localtileserver
        if vmax == vmin:
            t = 0.5
        else:
            t = (class_id - vmin) / (vmax - vmin)

        hex_color = mcolors.to_hex(cmap(t))

        rows.append(
            f"""
            <div style="
                display:flex;
                align-items:center;
                margin:3px 0;
                color:#111;
            ">
                <div style="
                    width:18px;
                    height:14px;
                    background:{hex_color};
                    border:1px solid #444;
                    margin-right:8px;
                    flex:0 0 auto;
                "></div>

                <div style="
                    font-size:12px;
                    color:#111;
                    white-space:nowrap;
                ">
                    Class {class_id}
                </div>
            </div>
            """
        )

    final_legend_html = ipywidgets.HTML(
        value=f"""
        <div style="
            background:white;
            color:#111;
            padding:8px 10px;
            border:1px solid #777;
            border-radius:4px;
            min-width:120px;
            box-shadow:0 1px 4px rgba(0,0,0,0.25);
        ">
            <div style="
                font-weight:bold;
                margin-bottom:6px;
                color:#111;
            ">
                Final classes
            </div>

            {''.join(rows)}
        </div>
        """
    )
    return final_legend_html


def display_images_binary_labels(
    m,
    inHelper,
    clusterMapFile,
    labelsFile,
    newClusters,
    colormap: str = FINAL_LABEL_COLORMAP,
):
    # This is also clipped because inHelper._dataset is the 512x512 input clip.
    cmDataset = Clusterer.labelsToGeotiff(
        inHelper._dataset,
        clusterMapFile,
        newClusters,
    )

    cmHelper = ImageHelper()
    cmHelper.initFromDataset(cmDataset, inHelper._noDataValue)

    cluster_client = TileClient(str(clusterMapFile), debug=True)
    cluster_layer = get_leaflet_tile_layer(
        cluster_client,
        vmin=cmHelper._minValue,
        vmax=cmHelper._maxValue,
        nodata=cmHelper._noDataValue,
        opacity=0.5,
        colormap=colormap,
    )
    cluster_layer.name = clusterMapFile.name

    # Remove the old 30-cluster legend, if it is still on the map
    try:
        m.remove(legend_control)
    except Exception:
        pass

    final_legend_html = _create_binary_legend(newClusters, colormap)

    final_legend_control = WidgetControl(
        widget=final_legend_html,
        position="topright",
    )

    m.add(final_legend_control)

    # Remove original first-pass labels
    for layer in list(m.layers):
        if layer.name == labelsFile.name:
            m.remove(layer)

    # Remove an older final layer if this cell is being rerun
    for layer in list(m.layers):
        if layer.name == clusterMapFile.name:
            m.remove(layer)

    # Add the newly generated final labels
    m.add(cluster_layer)

    display(m)

