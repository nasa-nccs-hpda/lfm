import geopandas as gpd
from shapely.geometry import box, Polygon
from pathlib import Path
import json
from pyproj import Transformer
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class FastLunarTileIndex:
    """
    Optimized lunar tile index with vectorized operations and smart filtering.
    """

    def __init__(self, gpkg_path: Path, tms_dir: Path):
        """Initialize with zoom 1 tiles and TMS metadata."""
        print("Loading zoom 1 tiles...")
        self.zoom1_tiles = gpd.read_file(gpkg_path, layer='zoom_1')
        self.zoom1_tiles.set_index('tile_index', inplace=True)

        print("Loading TMS metadata...")
        self.tms_metadata = {}
        self._load_tms_metadata(tms_dir)

        # Cache transformers for each zone (reuse instead of recreating)
        self._transformer_cache = {}

        print(f"✓ Loaded {len(self.zoom1_tiles)} zoom 1 tiles")
        print(f"✓ Loaded metadata for {len(self.tms_metadata)} zones")

    def _load_tms_metadata(self, tms_dir: Path):
        """Load tile matrix metadata for all zones."""
        zones = self._get_zones_to_process()

        for zone, tms_filename in tqdm(zones, desc="Loading zone metadata", unit="zone"):
            tms_path = tms_dir / tms_filename
            if not tms_path.exists():
                continue

            with open(tms_path, 'r') as f:
                tms = json.load(f)

            self.tms_metadata[zone] = {
                'crs': tms['crs'],
                'zoom_levels': {}
            }

            for zm in tms['tileMatrices']:
                zoom_id = int(zm['id'])
                self.tms_metadata[zone]['zoom_levels'][zoom_id] = {
                    'pointOfOrigin': zm['pointOfOrigin'],
                    'cellSize': zm['cellSize'],
                    'tileWidth': zm['tileWidth'],
                    'tileHeight': zm['tileHeight'],
                    'matrixWidth': zm['matrixWidth'],
                    'matrixHeight': zm['matrixHeight']
                }

    def _get_zones_to_process(self):
        """Get list of all zone names and filenames."""
        zones = []
        for zoneIDX in range(1, 46):
            zones.append((f"{zoneIDX}N", f'tms_LTM_{zoneIDX}NRG.json'))
            zones.append((f"{zoneIDX}S", f'tms_LTM_{zoneIDX}SRG.json'))
        zones.append(('NorthPole', 'tms_LPS_NRG.json'))
        zones.append(('SouthPole', 'tms_LPS_SRG.json'))
        return zones

    @lru_cache(maxsize=100)
    def _get_transformer(self, zone: str):
        """Get or create a cached transformer for a zone."""
        zone_crs = self.tms_metadata[zone]['crs']
        lat_lon_proj4 = '+proj=longlat +R=1737400 +no_defs'

        # Use pyproj instead of GDAL/OSR - it's faster and supports vectorization
        transformer = Transformer.from_crs(
            zone_crs,
            lat_lon_proj4,
            always_xy=True
        )
        return transformer

    def get_child_tiles_in_bbox(self, parent_zone: str, parent_row: int, parent_col: int,
                                parent_zoom: int, target_zoom: int,
                                query_bbox_proj: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """
        Calculate child tiles that intersect with query bbox (in projected coords).
        This pre-filters tiles BEFORE coordinate transformation - HUGE speedup!

        Args:
            parent_zone: Zone name
            parent_row: Parent tile row
            parent_col: Parent tile column
            parent_zoom: Parent zoom level
            target_zoom: Target zoom level
            query_bbox_proj: Query bbox in projected coordinates (minX, minY, maxX, maxY)

        Returns:
            List of (row, col) tuples that intersect the query bbox
        """
        if parent_zoom >= target_zoom:
            return [(parent_row, parent_col)]

        zoom_diff = target_zoom - parent_zoom
        factor = 2 ** zoom_diff

        start_row = parent_row * factor
        start_col = parent_col * factor

        # Get tile parameters for target zoom
        zm = self.tms_metadata[parent_zone]['zoom_levels'][target_zoom]
        origin_x = zm['pointOfOrigin'][0]
        origin_y = zm['pointOfOrigin'][1]
        cell_size = zm['cellSize']
        tile_width_pixels = zm['tileWidth']
        tile_height_pixels = zm['tileHeight']

        tile_width_meters = tile_width_pixels * cell_size
        tile_height_meters = tile_height_pixels * cell_size

        # Unpack query bbox
        query_minX, query_minY, query_maxX, query_maxY = query_bbox_proj

        # Pre-filter: only return tiles that intersect the query bbox in projected space
        child_tiles = []
        for dr in range(factor):
            for dc in range(factor):
                row = start_row + dr
                col = start_col + dc

                # Calculate tile bbox in projected coords (FAST - no transformation!)
                tile_minX = origin_x + (col * tile_width_meters)
                tile_maxX = tile_minX + tile_width_meters
                tile_maxY = origin_y - (row * tile_height_meters)
                tile_minY = tile_maxY - tile_height_meters

                # Check intersection (FAST - just comparing numbers)
                if (tile_maxX >= query_minX and tile_minX <= query_maxX and
                    tile_maxY >= query_minY and tile_minY <= query_maxY):
                    child_tiles.append((row, col))

        return child_tiles

    def calculate_tiles_bbox_vectorized(self, zone: str, zoom: int,
                                       tiles: List[Tuple[int, int]]) -> List[Polygon]:
        """
        Calculate bboxes for multiple tiles at once using vectorized operations.
        This is MUCH faster than calculating one at a time!

        Args:
            zone: Zone name
            zoom: Zoom level
            tiles: List of (row, col) tuples

        Returns:
            List of Shapely Polygons in geographic coordinates
        """
        if not tiles:
            return []

        zm = self.tms_metadata[zone]['zoom_levels'][zoom]
        origin_x = zm['pointOfOrigin'][0]
        origin_y = zm['pointOfOrigin'][1]
        cell_size = zm['cellSize']
        tile_width_pixels = zm['tileWidth']
        tile_height_pixels = zm['tileHeight']

        tile_width_meters = tile_width_pixels * cell_size
        tile_height_meters = tile_height_pixels * cell_size

        # Convert tiles to numpy arrays for vectorized operations
        tiles_array = np.array(tiles)
        rows = tiles_array[:, 0]
        cols = tiles_array[:, 1]

        # Vectorized bbox calculation (FAST!)
        minX_arr = origin_x + (cols * tile_width_meters)
        maxX_arr = minX_arr + tile_width_meters
        maxY_arr = origin_y - (rows * tile_height_meters)
        minY_arr = maxY_arr - tile_height_meters

        # Get cached transformer
        transformer = self._get_transformer(zone)

        # Transform all corners at once (VECTORIZED - HUGE speedup!)
        # Transform lower-left corners
        ll_x, ll_y = transformer.transform(minX_arr, minY_arr)
        # Transform lower-right corners
        lr_x, lr_y = transformer.transform(maxX_arr, minY_arr)
        # Transform upper-right corners
        ur_x, ur_y = transformer.transform(maxX_arr, maxY_arr)
        # Transform upper-left corners
        ul_x, ul_y = transformer.transform(minX_arr, maxY_arr)

        # Create polygons from transformed coordinates
        polygons = []
        for i in range(len(tiles)):
            coords = [
                (ll_x[i], ll_y[i]),
                (lr_x[i], lr_y[i]),
                (ur_x[i], ur_y[i]),
                (ul_x[i], ul_y[i]),
                (ll_x[i], ll_y[i])
            ]
            polygons.append(Polygon(coords))

        return polygons

    def transform_bbox_to_projected(self, bbox_geo: Tuple[float, float, float, float],
                                    zone: str) -> Tuple[float, float, float, float]:
        """
        Transform geographic bbox to zone's projected coordinates.
        Used for pre-filtering tiles.
        """
        transformer = self._get_transformer(zone)

        # Transform bbox corners (inverse transformation)
        minX_geo, minY_geo, maxX_geo, maxY_geo = bbox_geo

        # Transform corners
        ll_x, ll_y = transformer.transform(minX_geo, minY_geo, direction='INVERSE')
        lr_x, lr_y = transformer.transform(maxX_geo, minY_geo, direction='INVERSE')
        ur_x, ur_y = transformer.transform(maxX_geo, maxY_geo, direction='INVERSE')
        ul_x, ul_y = transformer.transform(minX_geo, maxY_geo, direction='INVERSE')

        # Get bounding box of transformed bbox
        all_x = [ll_x, lr_x, ur_x, ul_x]
        all_y = [ll_y, lr_y, ur_y, ul_y]

        return (min(all_x), min(all_y), max(all_x), max(all_y))

    def query_tiles(self, bbox: Tuple[float, float, float, float],
                   target_zoom: int,
                   zones: List[str] = None,
                   show_progress: bool = True,
                   batch_size: int = 1000) -> gpd.GeoDataFrame:
        """
        Optimized query tiles with vectorized operations and smart filtering.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in geographic coordinates
            target_zoom: Target zoom level
            zones: Optional list of zones to search
            show_progress: Whether to show progress bars
            batch_size: Number of tiles to process in each batch

        Returns:
            GeoDataFrame with tiles at target zoom level
        """
        query_geom = box(*bbox)

        # Step 1: Find zoom 1 tiles that intersect the query bbox
        if show_progress:
            print(f"Step 1: Querying zoom 1 tiles...")

        intersecting_zoom1 = self.zoom1_tiles[
            self.zoom1_tiles.geometry.intersects(query_geom)
        ]

        if zones:
            intersecting_zoom1 = intersecting_zoom1[
                intersecting_zoom1['zone_name'].isin(zones)
            ]

        if show_progress:
            print(f"  Found {len(intersecting_zoom1)} zoom 1 tiles")

        if len(intersecting_zoom1) == 0:
            return gpd.GeoDataFrame()

        # Step 2: Process each zone
        if show_progress:
            print(f"Step 2: Processing tiles at zoom {target_zoom}...")

        result_tiles = []

        iterator = tqdm(
            intersecting_zoom1.iterrows(),
            total=len(intersecting_zoom1),
            desc=f"Processing zones",
            unit="zone",
            disable=not show_progress
        )

        for idx, z1_tile in iterator:
            zone = z1_tile['zone_name']

            # Transform query bbox to this zone's projected coordinates
            query_bbox_proj = self.transform_bbox_to_projected(bbox, zone)

            # Get child tiles that intersect (pre-filtered in projected space!)
            child_tiles = self.get_child_tiles_in_bbox(
                zone,
                z1_tile['tile_row'],
                z1_tile['tile_col'],
                parent_zoom=1,
                target_zoom=target_zoom,
                query_bbox_proj=query_bbox_proj
            )

            if not child_tiles:
                continue

            # Process tiles in batches using vectorized transformation
            for i in range(0, len(child_tiles), batch_size):
                batch = child_tiles[i:i+batch_size]

                # Calculate geometries for entire batch at once (VECTORIZED!)
                tile_geometries = self.calculate_tiles_bbox_vectorized(
                    zone, target_zoom, batch
                )

                # Check precise intersection and add to results
                for (row, col), geom in zip(batch, tile_geometries):
                    if geom.intersects(query_geom):
                        result_tiles.append({
                            'geometry': geom,
                            'zone_name': zone,
                            'zoom_level': target_zoom,
                            'tile_row': row,
                            'tile_col': col,
                            'tile_index': f'{zone}_{target_zoom}_{row}_{col}'
                        })

            if show_progress:
                iterator.set_postfix({
                    'zone': zone,
                    'found': len(result_tiles)
                })

        if show_progress:
            print(f"  ✓ Found {len(result_tiles)} tiles at zoom {target_zoom}")

        # Step 3: Create GeoDataFrame
        if not result_tiles:
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(
            result_tiles,
            crs='+proj=longlat +R=1737400 +no_defs'
        )

        return gdf


# ============================================================================
# PARALLEL PROCESSING VERSION (Even Faster!)
# ============================================================================

def process_zone_parallel(args):
    """
    Process a single zone in parallel.
    This function is called by multiple processes.
    """
    zone, z1_tile, bbox, target_zoom, tms_metadata = args

    # Recreate transformer in this process
    from pyproj import Transformer

    zone_crs = tms_metadata[zone]['crs']
    lat_lon_proj4 = '+proj=longlat +R=1737400 +no_defs'
    transformer = Transformer.from_crs(zone_crs, lat_lon_proj4, always_xy=True)

    # Transform query bbox to projected coords
    minX_geo, minY_geo, maxX_geo, maxY_geo = bbox
    ll_x, ll_y = transformer.transform(minX_geo, minY_geo, direction='INVERSE')
    lr_x, lr_y = transformer.transform(maxX_geo, minY_geo, direction='INVERSE')
    ur_x, ur_y = transformer.transform(maxX_geo, maxY_geo, direction='INVERSE')
    ul_x, ul_y = transformer.transform(minX_geo, maxY_geo, direction='INVERSE')

    all_x = [ll_x, lr_x, ur_x, ul_x]
    all_y = [ll_y, lr_y, ur_y, ul_y]
    query_bbox_proj = (min(all_x), min(all_y), max(all_x), max(all_y))

    # Get child tiles (with pre-filtering)
    zoom_diff = target_zoom - 1
    factor = 2 ** zoom_diff
    start_row = z1_tile['tile_row'] * factor
    start_col = z1_tile['tile_col'] * factor

    zm = tms_metadata[zone]['zoom_levels'][target_zoom]
    origin_x = zm['pointOfOrigin'][0]
    origin_y = zm['pointOfOrigin'][1]
    cell_size = zm['cellSize']
    tile_width_pixels = zm['tileWidth']
    tile_height_pixels = zm['tileHeight']

    tile_width_meters = tile_width_pixels * cell_size
    tile_height_meters = tile_height_pixels * cell_size

    query_minX, query_minY, query_maxX, query_maxY = query_bbox_proj

    # Pre-filter tiles
    child_tiles = []
    for dr in range(factor):
        for dc in range(factor):
            row = start_row + dr
            col = start_col + dc

            tile_minX = origin_x + (col * tile_width_meters)
            tile_maxX = tile_minX + tile_width_meters
            tile_maxY = origin_y - (row * tile_height_meters)
            tile_minY = tile_maxY - tile_height_meters

            if (tile_maxX >= query_minX and tile_minX <= query_maxX and
                tile_maxY >= query_minY and tile_minY <= query_maxY):
                child_tiles.append((row, col))

    if not child_tiles:
        return []

    # Vectorized transformation
    tiles_array = np.array(child_tiles)
    rows = tiles_array[:, 0]
    cols = tiles_array[:, 1]

    minX_arr = origin_x + (cols * tile_width_meters)
    maxX_arr = minX_arr + tile_width_meters
    maxY_arr = origin_y - (rows * tile_height_meters)
    minY_arr = maxY_arr - tile_height_meters

    ll_x, ll_y = transformer.transform(minX_arr, minY_arr)
    lr_x, lr_y = transformer.transform(maxX_arr, minY_arr)
    ur_x, ur_y = transformer.transform(maxX_arr, maxY_arr)
    ul_x, ul_y = transformer.transform(minX_arr, maxY_arr)

    # Create results
    query_geom = box(*bbox)
    result_tiles = []

    for i in range(len(child_tiles)):
        coords = [
            (ll_x[i], ll_y[i]),
            (lr_x[i], lr_y[i]),
            (ur_x[i], ur_y[i]),
            (ul_x[i], ul_y[i]),
            (ll_x[i], ll_y[i])
        ]
        geom = Polygon(coords)

        if geom.intersects(query_geom):
            result_tiles.append({
                'geometry': geom,
                'zone_name': zone,
                'zoom_level': target_zoom,
                'tile_row': child_tiles[i][0],
                'tile_col': child_tiles[i][1],
                'tile_index': f'{zone}_{target_zoom}_{child_tiles[i][0]}_{child_tiles[i][1]}'
            })

    return result_tiles


class ParallelLunarTileIndex(FastLunarTileIndex):
    """
    Parallel processing version - uses all CPU cores.
    """

    def query_tiles_parallel(self, bbox: Tuple[float, float, float, float],
                            target_zoom: int,
                            zones: List[str] = None,
                            show_progress: bool = True,
                            max_workers: int = None) -> gpd.GeoDataFrame:
        """
        Query tiles using parallel processing across multiple CPU cores.
        """
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)

        query_geom = box(*bbox)

        # Step 1: Find zoom 1 tiles
        if show_progress:
            print(f"Step 1: Querying zoom 1 tiles...")

        intersecting_zoom1 = self.zoom1_tiles[
            self.zoom1_tiles.geometry.intersects(query_geom)
        ]

        if zones:
            intersecting_zoom1 = intersecting_zoom1[
                intersecting_zoom1['zone_name'].isin(zones)
            ]

        if show_progress:
            print(f"  Found {len(intersecting_zoom1)} zoom 1 tiles")

        if len(intersecting_zoom1) == 0:
            return gpd.GeoDataFrame()

        # Step 2: Process zones in parallel
        if show_progress:
            print(f"Step 2: Processing tiles at zoom {target_zoom} (using {max_workers} workers)...")

        # Prepare arguments for parallel processing
        args_list = [
            (row['zone_name'], row, bbox, target_zoom, self.tms_metadata)
            for idx, row in intersecting_zoom1.iterrows()
        ]

        # Process in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_zone_parallel, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures),
                             desc="Processing zones", disable=not show_progress):
                try:
                    zone_results = future.result()
                    all_results.extend(zone_results)
                except Exception as e:
                    print(f"Error processing zone: {e}")

        if show_progress:
            print(f"  ✓ Found {len(all_results)} tiles at zoom {target_zoom}")

        # Create GeoDataFrame
        if not all_results:
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(
            all_results,
            crs='+proj=longlat +R=1737400 +no_defs'
        )

        return gdf


# ============================================================================
# USAGE
# ============================================================================

def main():
    REPO_PATH = Path('/explore/nobackup/people/ajkerr1/Lunar_FM/model_view_cont/lfm')
    TMS_DIR = REPO_PATH / 'TMS/RG'
    GPKG_PATH = TMS_DIR / 'tile_database.gpkg'

    # Option 1: Fast single-threaded version
    # print("="*60)
    # print("Testing Fast Single-Threaded Version")
    # print("="*60)
    # tile_index = FastLunarTileIndex(GPKG_PATH, TMS_DIR)
    query_bbox = (-180, 0, -170, 20)

    # start = time.time()
    # result = tile_index.query_tiles(query_bbox, target_zoom=10, show_progress=True)
    # elapsed = time.time() - start

    # print(f"\n✓ Found {len(result)} tiles in {elapsed:.2f} seconds")

    # Option 2: Parallel version (even faster!)
    print("\n" + "="*60)
    print("Testing Parallel Version")
    print("="*60)
    tile_index_parallel = ParallelLunarTileIndex(GPKG_PATH, TMS_DIR)

    start = time.time()
    result = tile_index_parallel.query_tiles_parallel(
        query_bbox,
        target_zoom=15,
        show_progress=True,
        max_workers=8  # Adjust based on your CPU
    )
    elapsed = time.time() - start

    print(f"\n✓ Found {len(result)} tiles in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()