from pathlib import Path

PROJECT_DIR = Path("/explore/nobackup/projects/lfm")
DATA_DIR = PROJECT_DIR / "processed_data/Lunar/"
WAC_DIR = DATA_DIR / "LRO_WAC_Pho_Sites"

# Training paths
TRAIN_DIR = PROJECT_DIR / "model_inputs/300_300_inputs/7_band_vis_uv/sem_seg"
LABEL_DIR = TRAIN_DIR / "labels"
CHIP_DIR = TRAIN_DIR / "chips"
GPKG_PATH = CHIP_DIR / "WAC_TILES.gpkg"

# Tile database
TILE_DB_PATH = WAC_DIR / "output_index.shp"

ZOOM_LEVEL = 5
COMMON_NODATA = -3.40282265508890445e+38