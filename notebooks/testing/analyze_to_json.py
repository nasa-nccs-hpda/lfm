#!/usr/bin/env python3
"""
Analyze static data bands that match a reference geotiff and save to JSON.
Matches based on filename (static file stems should match reference band names).
"""

from osgeo import gdal
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from glob import glob

# Constants
STATIC_LINKS = Path("/explore/nobackup/projects/lfm/staticLinks")


def get_band_names(filepath):
    """Extract band names from a raster file."""
    ds = gdal.Open(str(filepath))
    if ds is None:
        print(f"Warning: Could not open {filepath}")
        return []

    band_names = []
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        band_metadata = band.GetMetadata()
        band_name = band_metadata.get('Name', f"band_{band_idx}")
        band_names.append(band_name)

    ds = None
    return band_names


def analyze_band_type(band_array, threshold_unique=100):
    """Analyze a band to determine if it's discrete or continuous."""
    # Mask nodata values
    valid_data = band_array[~np.isnan(band_array)]

    if len(valid_data) == 0:
        return {
            "type": "empty",
            "unique_count": 0,
            "all_integers": False,
            "min": None,
            "max": None,
            "sample_values": []
        }

    # Get unique values
    unique_values = np.unique(valid_data)
    unique_count = len(unique_values)

    # Check if all values are integers (even if stored as float)
    all_integers = np.allclose(valid_data, np.round(valid_data))

    # Determine type
    if unique_count <= threshold_unique:
        band_type = "discrete"
    elif all_integers and unique_count < len(valid_data) * 0.01:
        band_type = "likely_discrete"
    else:
        band_type = "continuous"

    return {
        "type": band_type,
        "unique_count": int(unique_count),
        "all_integers": bool(all_integers),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "sample_values": unique_values[:20].tolist() if unique_count <= 20 else unique_values[:10].tolist()
    }


def analyze_band_type_overview(band, threshold_unique=100):
    """Analyze using overview/pyramid if available (much faster for large files)."""

    # Check for overviews
    overview_count = band.GetOverviewCount()

    if overview_count > 0:
        # Use the smallest overview for quick analysis
        overview = band.GetOverview(overview_count - 1)
        print(f"    Using overview: {overview.XSize}x{overview.YSize}")
        data = overview.ReadAsArray().astype(float)
    else:
        # Subsample the full resolution (every Nth pixel)
        subsample = 10  # Read every 10th pixel
        print(f"    Subsampling full resolution (1:{subsample})...")
        data = band.ReadAsArray(0, 0, band.XSize, band.YSize,
                                 band.XSize // subsample,
                                 band.YSize // subsample).astype(float)

    # Now analyze the smaller array
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan

    return analyze_band_type(data, threshold_unique)  # Use your original function


def analyze_matching_bands(reference_tif, output_json=None, unique_threshold=100):
    """
    Analyze bands from static files that match the reference geotiff.
    Matches based on filename stem matching reference band names.

    Args:
        reference_tif: Path to reference geotiff with subset of bands
        output_json: Output JSON filepath
        unique_threshold: Max unique values to consider discrete
    """

    # Get band names from reference geotiff
    print(f"Reading reference bands from: {reference_tif}")
    reference_bands = get_band_names(reference_tif)
    print(f"Found {len(reference_bands)} bands in reference file")
    print(f"Reference bands: {reference_bands[:5]}..." if len(reference_bands) > 5 else f"Reference bands: {reference_bands}")

    # Glob all static files
    print(f"\nSearching for static files in: {STATIC_LINKS}")
    static_files = list(STATIC_LINKS.glob("*.tif")) + list(STATIC_LINKS.glob("*.tiff"))
    print(f"Found {len(static_files)} static files")

    # Build mapping of band_name -> file path
    # Match by: remove .tif/.tiff extension and match to reference band name
    print("\nIndexing static files by filename...")
    static_file_map = {}
    for static_file in static_files:
        # Get filename without .tif extension
        # e.g., "LDRM_32_N_FLOAT.iau.tif" -> "LDRM_32_N_FLOAT.iau"
        file_stem = static_file.name.replace('.tif', '').replace('.tiff', '')

        if file_stem in reference_bands:
            static_file_map[file_stem] = static_file

    print(f"Matched {len(static_file_map)} bands")

    # Prepare output structure
    result = {
        "metadata": {
            "reference_file": str(reference_tif),
            "static_links_path": str(STATIC_LINKS),
            "analysis_date": datetime.now().isoformat(),
            "num_reference_bands": len(reference_bands),
            "num_matched_bands": len(static_file_map),
            "unmatched_bands": [b for b in reference_bands if b not in static_file_map]
        },
        "bands": {}
    }

    # Analyze each matched band
    print(f"\nAnalyzing {len(static_file_map)} matched bands...")
    for idx, (band_name, static_file) in enumerate(sorted(static_file_map.items()), 1):
        print(f"  [{idx}/{len(static_file_map)}] {band_name} from {static_file.name}")

        try:
            ds = gdal.Open(str(static_file))
            if ds is None:
                print(f"    Error: Could not open {static_file}")
                continue

            band = ds.GetRasterBand(1)  # These are single-band files

            # Get metadata
            band_metadata = {
                "band_number": idx,  # Sequential in output
                "name": band_name,
                "source_file": static_file.name,
                "source_band_idx": 1,
                "data_type": gdal.GetDataTypeName(band.DataType),
                "nodata_value": band.GetNoDataValue(),
                "description": band.GetDescription() or "",
                "metadata": band.GetMetadata(),
                "units": "",  # To be filled manually
                "source": "",  # To be filled manually
            }

            # Read and analyze band data
            band_array = band.ReadAsArray().astype(float)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                band_array[band_array == nodata] = np.nan

            # Analyze band type
            analysis = analyze_band_type_overview(band, unique_threshold)
            band_metadata.update(analysis)

            result["bands"][band_name] = band_metadata

            ds = None

        except Exception as e:
            print(f"    Error analyzing {band_name}: {e}")
            continue

    # Determine output filename
    if output_json is None:
        output_json = "static_bands_matched.json"

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(result['bands'])} bands to: {output_json}")
    print(f"  Expected: 63 bands")
    print(f"  Matched: {len(result['bands'])} bands")
    if result['metadata']['unmatched_bands']:
        print(f"  Unmatched: {len(result['metadata']['unmatched_bands'])} bands")
        print(f"    First 5: {result['metadata']['unmatched_bands'][:5]}")
    print(f"{'='*70}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_matched_bands.py <reference_geotiff> [output.json] [unique_threshold]")
        print("\nExamples:")
        print("  python analyze_matched_bands.py StaticCube-LTM22S_Zoom-5_Tile-1-4.tif")
        print("  python analyze_matched_bands.py StaticCube-LTM22S_Zoom-5_Tile-1-4.tif static_bands.json")
        print("  python analyze_matched_bands.py StaticCube-LTM22S_Zoom-5_Tile-1-4.tif static_bands.json 50")
        print(f"\nStatic files will be read from: {STATIC_LINKS}")
        print("\nMatching: Reference band names will be matched to static filenames (without .tif extension)")
        sys.exit(1)

    reference_tif = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    analyze_matching_bands(reference_tif, output_json, threshold)