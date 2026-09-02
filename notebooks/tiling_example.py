#!/usr/bin/env python3
"""Create and plot WAC/static LTM cubes for AOIs declared in JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(
    str(Path(__file__).resolve().parents[1]).replace(
        "/panfs/ccds02/nobackup",
        "/explore/nobackup",
    )
)
if not (REPO_ROOT / "lfm").is_dir() or not (REPO_ROOT / "model").is_dir():
    raise FileNotFoundError(
        f"Could not find the lfm/ and model/ directories beneath {REPO_ROOT}."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import TileConfig, TileSourceConfig, create_tiles_for_aoi  # noqa: E402
from lfm.all_models.all_tasks.tiling_utils import (  # noqa: E402
    DEFAULT_WAC_BAND_NUMBER,
    DEFAULT_ZOOM_LEVEL,
    RUN_ID,
    make_static_source,
    validate_path_pairs,
)
from lfm.all_models.all_tasks.viz import (  # noqa: E402
    pair_dynamic_and_static,
    plot_cube_pairs,
    print_record_summary,
)


PROJECT_DATA_DIR = Path("/explore/nobackup/projects/lfm")
WAC_DATA_DIR = PROJECT_DATA_DIR / "processed_data/Lunar/LRO_WAC_Pho_Sites"
STATIC_DATA_DIR = PROJECT_DATA_DIR / "staticLinks"
WAC_INDEX = WAC_DATA_DIR / "output_index.shp"
STATIC_INDEX = STATIC_DATA_DIR / "db2.shp"
LOCATION_FIELD = "location"

ZOOM_LEVEL = DEFAULT_ZOOM_LEVEL
WAC_BAND_NUMBER = DEFAULT_WAC_BAND_NUMBER
STATIC_BAND_TO_PLOT = "lola_kaguya_60mpp_elv"
MAX_PLOT_TILES = 2

AOI_KEYS = ("ul_lat", "ul_lon", "lr_lat", "lr_lon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create WAC/static LTM cubes and one plot for every product/AOI "
            "entry in a JSON configuration file."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="JSON file containing an 'entries' list of product IDs and AOIs.",
    )
    return parser.parse_args()


def load_entries(config_path: Path) -> list[dict[str, Any]]:
    """Load and validate product-scoped geographic AOIs from JSON."""
    if not config_path.is_file():
        raise FileNotFoundError(f"JSON configuration does not exist: {config_path}")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    raw_entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            "The JSON configuration must contain a nonempty 'entries' list."
        )

    entries: list[dict[str, Any]] = []
    seen_product_ids: set[str] = set()
    for number, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Entry {number} must be a JSON object.")

        product_id = raw_entry.get("product_id")
        if (
            not isinstance(product_id, str)
            or not product_id.startswith("M")
            or not product_id.isalnum()
        ):
            raise ValueError(
                f"Entry {number} product_id must be a pure alphanumeric M... ID."
            )
        if product_id in seen_product_ids:
            raise ValueError(f"Duplicate product_id in entry {number}: {product_id}")
        seen_product_ids.add(product_id)

        raw_aoi = raw_entry.get("aoi")
        if not isinstance(raw_aoi, dict):
            raise TypeError(f"Entry {number} must contain an 'aoi' object.")
        missing = [key for key in AOI_KEYS if key not in raw_aoi]
        extra = sorted(set(raw_aoi) - set(AOI_KEYS))
        if missing or extra:
            raise ValueError(
                f"Entry {number} AOI keys are invalid; "
                f"missing={missing}, extra={extra}."
            )

        try:
            aoi = {key: float(raw_aoi[key]) for key in AOI_KEYS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Entry {number} AOI values must be numeric.") from exc
        if not all(math.isfinite(value) for value in aoi.values()):
            raise ValueError(f"Entry {number} AOI values must be finite.")
        if not (-90.0 <= aoi["lr_lat"] < aoi["ul_lat"] <= 90.0):
            raise ValueError(
                f"Entry {number} must have valid latitudes with ul_lat > lr_lat."
            )
        if aoi["ul_lon"] >= aoi["lr_lon"]:
            raise ValueError(
                f"Entry {number} must have ul_lon < lr_lon; "
                "antimeridian-crossing AOIs are not supported by this script."
            )

        entries.append({"product_id": product_id, "aoi": aoi})
    return entries


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    entries = load_entries(config_path)

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    validate_path_pairs(
        {
            "WAC data directory": WAC_DATA_DIR,
            "static data directory": STATIC_DATA_DIR,
        },
        path_type="directory",
    )
    validate_path_pairs(
        {
            "WAC index": WAC_INDEX,
            "static index": STATIC_INDEX,
        },
        path_type="file",
    )

    output_dir = REPO_ROOT / "outputs" / "tiling" / RUN_ID
    output_dir.mkdir(parents=True, exist_ok=False)

    wac_source = TileSourceConfig(
        name="wac",
        data_dir=WAC_DATA_DIR,
        index_path=WAC_INDEX,
        location_field=LOCATION_FIELD,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
    )
    static_source = make_static_source(
        data_dir=STATIC_DATA_DIR,
        index_path=STATIC_INDEX,
        location_field=LOCATION_FIELD,
    )

    print(f"Repository root: {REPO_ROOT}")
    print(f"Configuration: {config_path}")
    print(f"Requests: {len(entries)}")
    print(f"Outputs: {output_dir}")

    plot_paths: list[Path] = []
    for number, entry in enumerate(entries, start=1):
        product_id = entry["product_id"]
        aoi = entry["aoi"]
        print(f"\n[{number}/{len(entries)}] Tiling WAC product {product_id}")
        print(f"AOI: {aoi}")

        tile_config = TileConfig(
            output_dir=output_dir / "cubes" / product_id,
            zoom_level=ZOOM_LEVEL,
            sources=(wac_source, static_source),
        )
        records = create_tiles_for_aoi(
            tile_config,
            **aoi,
            selectors={"wac": product_id},
        )
        print_record_summary(records)

        pairs = pair_dynamic_and_static(records, "wac")
        print(f"WAC/static pairs available for plotting: {len(pairs)}")
        plot_path = output_dir / "plots" / f"wac_static_cubes_{product_id}.png"
        figure = plot_cube_pairs(
            pairs,
            dynamic_label=f"WAC {product_id}",
            dynamic_band_number=WAC_BAND_NUMBER,
            static_band_name=STATIC_BAND_TO_PLOT,
            output_path=plot_path,
            max_tiles=MAX_PLOT_TILES,
        )
        plt.close(figure)
        plot_paths.append(plot_path)

    print(f"\nCompleted {len(entries)} WAC/static tiling requests.")
    print(f"Created {len(plot_paths)} plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
