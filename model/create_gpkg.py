import json
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
from osgeo import osr
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


def get_zones_to_process():
    """Generate list of all zones with their file names."""
    zones = []
    for zoneIDX in range(1, 46):
        zones.append((f"{zoneIDX}N", f'tms_LTM_{zoneIDX}NRG.json'))
        zones.append((f"{zoneIDX}S", f'tms_LTM_{zoneIDX}SRG.json'))
    zones.append(('NorthPole', 'tms_LPS_NRG.json'))
    zones.append(('SouthPole', 'tms_LPS_SRG.json'))
    return zones


def transformTileBboxVectorized(bboxes: np.ndarray,
                                 sourceSRS: str,
                                 targetSRS_PROJ4: str,
                                 useLonLat: bool) -> list:
    """
    Transform bounding boxes using the same approach as ltmToLatLon.

    Args:
        bboxes: Nx4 array of (minX, minY, maxX, maxY) coordinates
        sourceSRS: WKT string of source CRS (LTM)
        targetSRS_PROJ4: PROJ4 string of target CRS (lat/lon)
        useLonLat: whether to swap lat/lon order during bbox transformation

    Returns:
        List of Shapely Polygons in target CRS
    """
    # Setup LTM SRS
    ltmSRS = osr.SpatialReference()
    ltmSRS.ImportFromWkt(sourceSRS)

    # Setup Lat/lon SRS
    latLonSRS = osr.SpatialReference()
    latLonSRS.ImportFromProj4(targetSRS_PROJ4)

    # Create coordinate transformation
    xform = osr.CoordinateTransformation(ltmSRS, latLonSRS)

    polygons = []

    # Process each bbox
    for bbox in bboxes:
        minX, minY, maxX, maxY = bbox

        # Transform all 4 corners
        llLatLon = xform.TransformPoint(minX, minY)  # lower-left
        lrLatLon = xform.TransformPoint(maxX, minY)  # lower-right
        urLatLon = xform.TransformPoint(maxX, maxY)  # upper-right
        ulLatLon = xform.TransformPoint(minX, maxY)  # upper-left

        coords = [
            (llLatLon[0], llLatLon[1]),  # lower-left
            (lrLatLon[0], lrLatLon[1]),  # lower-right
            (urLatLon[0], urLatLon[1]),  # upper-right
            (ulLatLon[0], ulLatLon[1]),  # upper-left
            (llLatLon[0], llLatLon[1])   # close the ring
        ]
        polygons.append(Polygon(coords))

    return polygons


def process_single_zone(zone_data: tuple,
                       tmsDir: Path,
                       zoomLevel: int,
                       LAT_LON_SRS_PROJ4: str,
                       useLonLat: bool) -> list:
    """
    Process a single zone and return tile features.
    Designed to be run in parallel or sequentially.
    """
    zone, tmsFileName = zone_data
    tmsPath = tmsDir / tmsFileName

    if not tmsPath.exists():
        return []

    with open(tmsPath, 'r') as f:
        tms = json.load(f)

    zoomLevelsList = tms['tileMatrices']
    zoneSRS = tms['crs']

    tileDef = next((zl for zl in zoomLevelsList if int(zl['id']) == zoomLevel), None)
    if tileDef is None:
        return []

    originX = tileDef['pointOfOrigin'][0]
    originY = tileDef['pointOfOrigin'][1]
    cellSize = tileDef['cellSize']
    tileWidthPixels = tileDef['tileWidth']
    tileHeightPixels = tileDef['tileHeight']
    matrixWidth = tileDef['matrixWidth']
    matrixHeight = tileDef['matrixHeight']

    tileWidthMeters = tileWidthPixels * cellSize
    tileHeightMeters = tileHeightPixels * cellSize

    # Pre-calculate all tile bounding boxes using numpy
    cols = np.arange(matrixWidth)
    rows = np.arange(matrixHeight)

    # Create meshgrid of all row/col combinations
    col_grid, row_grid = np.meshgrid(cols, rows)
    col_flat = col_grid.flatten()
    row_flat = row_grid.flatten()

    # Vectorized bbox calculation
    minX_arr = originX + (col_flat * tileWidthMeters)
    maxX_arr = minX_arr + tileWidthMeters
    maxY_arr = originY - (row_flat * tileHeightMeters)
    minY_arr = maxY_arr - tileHeightMeters

    # Stack into Nx4 array
    bboxes = np.column_stack([minX_arr, minY_arr, maxX_arr, maxY_arr])

    # Transform all bboxes at once
    polygons = transformTileBboxVectorized(bboxes, zoneSRS, LAT_LON_SRS_PROJ4, useLonLat)

    # Create features
    tileFeatures = []
    idx = 0
    for row in range(matrixHeight):
        for col in range(matrixWidth):
            tileFeature = {
                'geometry': polygons[idx],
                'properties': {
                    'zone_name': zone,
                    'zoom_level': zoomLevel,
                    'tile_row': row,
                    'tile_col': col,
                    'tile_index': f'{zone}_{zoomLevel}_{row}_{col}',
                    'matrix_width': matrixWidth,
                    'matrix_height': matrixHeight,
                    'min_x_proj': float(bboxes[idx, 0]),
                    'min_y_proj': float(bboxes[idx, 1]),
                    'max_x_proj': float(bboxes[idx, 2]),
                    'max_y_proj': float(bboxes[idx, 3]),
                }
            }
            tileFeatures.append(tileFeature)
            idx += 1

    print(f"  ✓ {zone}: {len(tileFeatures)} tiles")
    return tileFeatures


def createGPKG(repoPath: Path,
               outFilename: str,
               zoomLevels: list = None,
               max_workers: int = None,
               useLonLat: bool = False,
               parallel: bool = True) -> dict:
    """
    Optimized GPKG creation with optional parallel processing.

    Args:
        repoPath: Path to repository
        outFilename: Output filename for the GPKG
        zoomLevels: List of zoom levels to process
        max_workers: Number of parallel workers (default: CPU count - 1)
        useLonLat: whether to swap lat/lon order during bbox transformation
        parallel: If True, use parallel processing. If False, process sequentially.
    """
    if zoomLevels is None:
        zoomLevels = [1]

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    tmsDir = repoPath / 'TMS/RG'
    output_path = tmsDir / outFilename

    if output_path.exists():
        output_path.unlink()
        print(f"Removed existing file: {output_path}")

    LAT_LON_SRS_PROJ4 = '+proj=longlat +R=1737400 +no_defs'
    results = {}

    for zoomLevel in zoomLevels:
        print(f"\n{'='*60}")
        if parallel:
            print(f"Processing Zoom Level {zoomLevel} (PARALLEL with {max_workers} workers)")
        else:
            print(f"Processing Zoom Level {zoomLevel} (SEQUENTIAL)")
        print(f"{'='*60}")

        zones = get_zones_to_process()
        all_tile_features = []

        if parallel:
            # PARALLEL PROCESSING
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all zone processing tasks
                future_to_zone = {
                    executor.submit(
                        process_single_zone,
                        zone_data,
                        tmsDir,
                        zoomLevel,
                        LAT_LON_SRS_PROJ4,
                        useLonLat
                    ): zone_data[0]
                    for zone_data in zones
                }

                # Collect results as they complete
                for future in as_completed(future_to_zone):
                    zone_name = future_to_zone[future]
                    try:
                        tile_features = future.result()
                        all_tile_features.extend(tile_features)
                    except Exception as e:
                        print(f"  ✗ {zone_name}: Error - {e}")
        else:
            # SEQUENTIAL PROCESSING
            for zone_data in zones:
                try:
                    tile_features = process_single_zone(
                        zone_data,
                        tmsDir,
                        zoomLevel,
                        LAT_LON_SRS_PROJ4,
                        useLonLat
                    )
                    all_tile_features.extend(tile_features)
                except Exception as e:
                    print(f"  ✗ {zone_data[0]}: Error - {e}")

        print(f"\nTotal tiles collected: {len(all_tile_features):,}")

        # ========================================
        # DEBUGGING: Check sample geometries
        # ========================================
        if len(all_tile_features) > 0:
            print("\n" + "="*60)
            print("DEBUGGING: Sample Polygon Inspection")
            print("="*60)

            # Sample first 3 tiles
            for i, feature in enumerate(all_tile_features[:3]):
                geom = feature['geometry']
                props = feature['properties']
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                coords = list(geom.exterior.coords)

                print(f"\nTile {i+1}: {props['tile_index']}")
                print(f"  Zone: {props['zone_name']}")
                print(f"  Bounds: {bounds}")
                print(f"  Bounds format: (lon_min, lat_min, lon_max, lat_max)")
                print(f"    Longitude range: {bounds[0]:.6f} to {bounds[2]:.6f}")
                print(f"    Latitude range:  {bounds[1]:.6f} to {bounds[3]:.6f}")
                print(f"  First 3 coordinates:")
                for j, coord in enumerate(coords[:3]):
                    print(f"    Point {j+1}: lon={coord[0]:.6f}, lat={coord[1]:.6f}")

                # Validation checks
                warnings_list = []
                if abs(bounds[0]) > 360 or abs(bounds[2]) > 360:
                    warnings_list.append("⚠️  Longitude values > 360 (might be in meters!)")
                if abs(bounds[1]) > 90 or abs(bounds[3]) > 90:
                    warnings_list.append("⚠️  Latitude values > 90 (might be in meters!)")
                if bounds[0] > bounds[2]:
                    warnings_list.append("⚠️  min_lon > max_lon (axis order issue?)")
                if bounds[1] > bounds[3]:
                    warnings_list.append("⚠️  min_lat > max_lat (axis order issue?)")

                if warnings_list:
                    for warning in warnings_list:
                        print(f"  {warning}")
                else:
                    print(f"  ✓ Coordinates look valid")

            print("="*60 + "\n")

        # ========================================
        # DEBUGGING: Check for antimeridian issues
        # ========================================
        print("\n" + "="*60)
        print("DEBUGGING: Antimeridian/Dateline Crossing Check")
        print("="*60)

        antimeridian_tiles = []
        for idx, feature in enumerate(all_tile_features[:10]):
            geom = feature['geometry']
            props = feature['properties']
            coords = list(geom.exterior.coords)

            # Check if polygon crosses antimeridian
            lons = [c[0] for c in coords[:-1]]  # Exclude closing point
            lon_range = max(lons) - min(lons)

            if lon_range > 180:  # Likely crosses antimeridian
                antimeridian_tiles.append(props['tile_index'])
                print(f"\n⚠️  Tile {props['tile_index']} likely crosses antimeridian:")
                print(f"    Lon range: {min(lons):.2f} to {max(lons):.2f} (span: {lon_range:.2f}°)")
                print(f"    First 4 corners:")
                for i, (lon, lat) in enumerate(coords[:4]):
                    print(f"      Point {i+1}: ({lon:.2f}, {lat:.2f})")

        if antimeridian_tiles:
            print(f"\n⚠️  Found {len(antimeridian_tiles)} tiles crossing antimeridian (in first 10)")
            print(f"    This can cause spatial query issues!")
        else:
            print(f"\n✓ No antimeridian crossing detected in first 10 tiles")

        print("="*60 + "\n")

        # Create GeoDataFrame
        print("Creating GeoDataFrame...")
        geometries = [f['geometry'] for f in all_tile_features]
        properties = [f['properties'] for f in all_tile_features]
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=LAT_LON_SRS_PROJ4)

        # Build spatial index
        spatial_index = gdf.sindex

        # Write to GPKG
        layer_name = f'zoom_{zoomLevel}'
        write_mode = 'w' if zoomLevel == zoomLevels[0] else 'a'

        print(f"Writing layer '{layer_name}' to GPKG...")
        gdf.to_file(output_path, driver='GPKG', layer=layer_name, mode=write_mode)
        print(f"✓ Layer '{layer_name}' written")

        results[zoomLevel] = {
            'gdf': gdf,
            'spatial_index': spatial_index,
            'tile_count': len(all_tile_features)
        }

        # Clear memory
        del all_tile_features, geometries, properties, gdf

    print(f"\n{'='*60}")
    print(f"✓ Created GPKG: {output_path}")
    print(f"✓ Total layers: {len(zoomLevels)}")
    print(f"{'='*60}\n")

    return results, output_path


def main():
    REPO_PATH = Path('/explore/nobackup/people/ajkerr1/Lunar_FM/model_view_cont/lfm')
    OUT_DIR = Path('/explore/nobackup/people/ajkerr1/Lunar_FM/')
    ZOOM_LEVELS = list(range(1, 2))
    USE_LON_LAT = False
    OUT_FILENAME = "tile_db_fixed.gpkg"
    USE_PARALLEL = False  # Set to False for sequential processing

    print("Creating GPKG with multiple zoom level layers...")
    print(f"Parallelization: {'ENABLED' if USE_PARALLEL else 'DISABLED'}")

    results, output_path = createGPKG(
        REPO_PATH,
        OUT_FILENAME,
        ZOOM_LEVELS,
        max_workers=None,
        useLonLat=USE_LON_LAT,
        parallel=USE_PARALLEL
    )

    # Print summary
    print("\nSummary:")
    for zoom_level, data in results.items():
        print(f"  Zoom {zoom_level}: {data['tile_count']:,} tiles")


if __name__ == "__main__":
    main()