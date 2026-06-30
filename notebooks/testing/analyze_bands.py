#!/usr/bin/env python3
"""
Diagnostic script to determine if raster bands contain discrete or continuous data.
"""

from osgeo import gdal
import numpy as np
import sys

def analyze_band_type(band_array, band_num, threshold_unique=100):
    """
    Analyze a band to determine if it's discrete or continuous.
    
    Args:
        band_array: numpy array of band data
        band_num: band number for reporting
        threshold_unique: max unique values to consider discrete
    
    Returns:
        dict with analysis results
    """
    # Mask nodata values
    valid_data = band_array[~np.isnan(band_array)]
    
    if len(valid_data) == 0:
        return {"type": "EMPTY", "unique_count": 0}
    
    # Get unique values
    unique_values = np.unique(valid_data)
    unique_count = len(unique_values)
    
    # Check if all values are integers (even if stored as float)
    all_integers = np.allclose(valid_data, np.round(valid_data))
    
    # Determine type
    if unique_count <= threshold_unique:
        band_type = "DISCRETE"
    elif all_integers and unique_count < len(valid_data) * 0.01:
        band_type = "LIKELY DISCRETE"
    else:
        band_type = "CONTINUOUS"
    
    return {
        "type": band_type,
        "unique_count": unique_count,
        "all_integers": all_integers,
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "sample_values": unique_values[:10].tolist() if unique_count <= 20 else unique_values[:5].tolist()
    }

def analyze_raster(filepath, unique_threshold=1000):
    """Analyze all bands in a raster file."""
    ds = gdal.Open(filepath)
    if ds is None:
        print(f"Error: Could not open {filepath}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*70}\n")
    
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        
        # Read band as array
        band_array = band.ReadAsArray().astype(float)
        
        # Handle nodata
        nodata = band.GetNoDataValue()
        if nodata is not None:
            band_array[band_array == nodata] = np.nan
        
        # Analyze
        result = analyze_band_type(band_array, band_idx, unique_threshold)
        
        # Report
        print(f"Band {band_idx}:")
        print(f"  Type: {result['type']}")
        print(f"  Unique values: {result['unique_count']}")
        if result['type'] != 'EMPTY':
            print(f"  All integers: {result['all_integers']}")
            print(f"  Range: [{result['min']:.4f}, {result['max']:.4f}]")
            print(f"  Sample values: {result['sample_values']}")
        print()
    
    ds = None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_bands.py <raster_file> [unique_threshold]")
        print("\nExample: python analyze_bands.py data.tif 50")
        sys.exit(1)
    
    filepath = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    analyze_raster(filepath, threshold)