from osgeo import gdal, ogr, osr
import os
from glob import glob

def create_tile_index(geotiff_list, output_dir, output_filename, wkt_file=None, layer_name='tile_index'):
    """
    Create a shapefile index of GeoTIFF files (equivalent to gdaltindex)

    Args:
        geotiff_list: List of paths to GeoTIFF files
        output_dir: Absolute path to output directory
        output_shapefile: Filename of shapefile
        wkt_file: Absolute path to .wkt file for target srs; if None, first geotiff
            srs will be used.
    """
    # Create the shapefile driver
    if output_filename.endswith('.gpkg'):
        driver_name = 'GPKG'
    elif output_filename.endswith('.shp'):
        driver_name = 'ESRI Shapefile'
    else:
        raise ValueError("Output file must be .gpkg or .shp")

    driver = ogr.GetDriverByName(driver_name)

    # Remove existing shapefile if it exists
    output_path = os.path.join(output_dir, output_filename)

    # Handle existing file/layer
    if os.path.exists(output_path):
        if driver_name == 'GPKG':
            # For GeoPackage, open and delete the layer if it exists
            ds = driver.Open(output_path, 1)  # Open for writing
            for i in range(ds.GetLayerCount()):
                if ds.GetLayer(i).GetName() == layer_name:
                    ds.DeleteLayer(i)
                    break
            ds = None
            # Re-open or create
            ds = driver.Open(output_path, 1)
            if ds is None:
                ds = driver.CreateDataSource(output_path)
        else:
            # For Shapefile, delete the entire datasource
            driver.DeleteDataSource(output_path)
            ds = driver.CreateDataSource(output_path)
    else:
        ds = driver.CreateDataSource(output_path)

    # If wkt_file is supplied, retrieve srs from there; otherwise get srs from first GeoTIFF
    if wkt_file:
        # Read WKT from file
        with open(wkt_file, 'r') as f:
            wkt_string = f.read()
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(wkt_string)
    else:
        # Get from first GeoTIFF
        first_raster = gdal.Open(geotiff_list[0])
        target_srs = osr.SpatialReference(wkt=first_raster.GetProjection())
        first_raster = None

    # Create layer
    layer = ds.CreateLayer('tile_index', target_srs, ogr.wkbPolygon)

    # Add attribute field for filename
    field_location = ogr.FieldDefn('location', ogr.OFTString)
    field_location.SetWidth(254)
    layer.CreateField(field_location)

    # Process each GeoTIFF
    print(f"Processing geotiffs to create {output_path}...")
    total_files = len(geotiff_list)
    last_reported = -1
    for idx, geotiff_path in enumerate(geotiff_list):
        raster = gdal.Open(geotiff_path)
        if raster is None:
            print(f"Warning: Could not open {geotiff_path}")
            continue

        # Get source coordinate system
        source_srs = osr.SpatialReference(wkt=raster.GetProjection())

        # Create coordinate transformation
        transform = osr.CoordinateTransformation(source_srs, target_srs)

        # Get geotransform and dimensions
        gt = raster.GetGeoTransform()
        cols = raster.RasterXSize
        rows = raster.RasterYSize

        # Calculate corner coordinates in source SRS
        minx = gt[0]
        maxy = gt[3]
        maxx = gt[0] + cols * gt[1]
        miny = gt[3] + rows * gt[5]

        # Transform corners to target CRS
        # Note: transform.TransformPoint returns (x, y, z)
        ll = transform.TransformPoint(minx, miny)  # lower left
        lr = transform.TransformPoint(maxx, miny)  # lower right
        ur = transform.TransformPoint(maxx, maxy)  # upper right
        ul = transform.TransformPoint(minx, maxy)  # upper left

        # Create polygon geometry
        # Create polygon geometry with explicit 2D coordinates
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint_2D(ll[0], ll[1])  # Explicitly 2D
        ring.AddPoint_2D(lr[0], lr[1])
        ring.AddPoint_2D(ur[0], ur[1])
        ring.AddPoint_2D(ul[0], ul[1])
        ring.AddPoint_2D(ll[0], ll[1])

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # Create feature
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        feature.SetField('location', geotiff_path)
        layer.CreateFeature(feature)

        # Cleanup
        feature = None
        raster = None

        # Print progress every 10%
        current_percent = int((idx + 1) / total_files * 100)
        if current_percent % 10 == 0 and current_percent != last_reported:
            print(f"{current_percent}% done.")
            last_reported = current_percent

    # Cleanup
    ds = None
    print(f"Tile index created: {output_path}")

# Usage example
geotiff_paths = glob("/explore/nobackup/projects/lfm/Benchmarks/LFM_Data/testdata/*.tif")
print(f"Found {len(geotiff_paths)} files.")

create_tile_index(
    geotiff_list=geotiff_paths,
    output_dir='/explore/nobackup/people/ajkerr1/Lunar_FM',
    output_filename='tile_index_2026_03_05.gpkg',
    wkt_file='/explore/nobackup/projects/lfm/IAU_30100_2015.wkt'
)

