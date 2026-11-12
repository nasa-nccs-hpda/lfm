#!/usr/bin/env bash
# Run on HPC: bash clip_m1182737038re_300px.sh [DESTINATION_FOLDER]
# With no argument, the script prompts for the output folder on HPC.
# SOURCE_TIF and SOURCE_GPKG can override the input paths (e.g. for testing).
#
# This is a native-resolution subset, not a resize of the annotated extent.
# The supplied gdalinfo reports 9280 x 70492 pixels at 1 x 1 metre/pixel.
# The selected 300 x 300 window contains crater centers 22, 23, 24, and 25:
# four centers is the maximum possible in a window this size for these labels.
# Several windows tie; this one shows four nearby craters with only small
# boundary cuts to craters 24 and 25. Other annotations are outside the crop.
# Pixel offsets below are zero-based. Source files are never modified.
#
# GDAL option references:
# https://gdal.org/en/stable/programs/gdal_translate.html#cmdoption-gdal_translate-srcwin
# https://gdal.org/en/stable/programs/ogr2ogr.html#cmdoption-ogr2ogr-clipdst
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Usage: bash %s [DESTINATION_FOLDER]\n' "$0"
    printf 'Run on HPC. Without an argument, prompts for the output folder.\n'
    exit 0
fi
if (( $# > 1 )); then
    printf 'Expected at most one destination-folder argument.\n' >&2
    exit 2
fi

source_tif=${SOURCE_TIF:-/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_NAC_IMP/M1182737038RE.ech.cog.tif}
source_gpkg=${SOURCE_GPKG:-/explore/nobackup/people/ajkerr1/Lunar_FM/latest_lfm/lfm/notebooks/outputs/labels/M1182737038RE.ech.cog_label_craters.gpkg}
xoff=5437
yoff=24605
size=300
stem=M1182737038RE.ech.cog_300px

destination=${1:-}
if [[ -z $destination ]]; then
    read -r -p 'Destination folder on HPC for the clipped TIFF and GeoPackage: ' destination
fi
if [[ -z ${destination//[[:space:]]/} ]]; then
    printf 'A destination folder is required.\n' >&2
    exit 2
fi
# Support a pasted ~/path without evaluating shell expressions in user input.
if [[ $destination == '~' ]]; then
    destination=$HOME
elif [[ $destination == '~/'* ]]; then
    destination="$HOME/${destination:2}"
fi

if ! type module >/dev/null 2>&1; then
    printf 'The module command is unavailable; run this in an HPC shell with environment modules initialized.\n' >&2
    exit 1
fi
module load gdal
for command_name in gdalinfo gdal_translate ogr2ogr python3; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Required command missing after module load gdal: %s\n' "$command_name" >&2
        exit 1
    fi
done
for input_path in "$source_tif" "$source_gpkg"; do
    if [[ ! -r $input_path ]]; then
        printf 'Cannot read source file: %s\n' "$input_path" >&2
        exit 1
    fi
done

mkdir -p -- "$destination"
for output_name in "$stem.tif" "${stem}_label_craters.gpkg" "$stem.json"; do
    if [[ -e "$destination/$output_name" || -L "$destination/$output_name" ]]; then
        printf 'Output already exists; choose another folder: %s\n' "$destination/$output_name" >&2
        exit 1
    fi
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/m1182737038re-300px.XXXXXXXX")
trap 'rm -rf -- "$work_dir"' EXIT
# Avoid creating auxiliary metadata files alongside the source raster.
export GDAL_PAM_ENABLED=NO

# Metadata only: no -stats, -hist, checksum, or whole-raster read.
gdalinfo -json "$source_tif" > "$work_dir/source_info.json"
python3 - "$work_dir" "$source_tif" "$source_gpkg" "$xoff" "$yoff" "$size" <<'PY'
import json
import math
import pathlib
import sys

work = pathlib.Path(sys.argv[1])
source_tif, source_gpkg = sys.argv[2:4]
xoff, yoff, size = map(int, sys.argv[4:7])
info = json.loads((work / 'source_info.json').read_text())
expected_transform = [281609.5, 1.0, 0.0, 1254602.5, 0.0, -1.0]
transform = info.get('geoTransform', [])
if info.get('size') != [9280, 70492] or len(transform) != 6 or any(
    not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    for a, b in zip(transform, expected_transform)
):
    raise SystemExit('Source grid differs from the supplied gdalinfo; recalculate the AOI before clipping.')
wkt = info.get('coordinateSystem', {}).get('wkt')
if not wkt:
    raise SystemExit('Source raster has no CRS.')
(work / 'source.wkt').write_text(wkt)
xmin = transform[0] + xoff * transform[1]
ymax = transform[3] + yoff * transform[5]
xmax = xmin + size * transform[1]
ymin = ymax + size * transform[5]
(work / 'bounds.txt').write_text(f'{xmin:.17g} {ymin:.17g} {xmax:.17g} {ymax:.17g}\n')
manifest = {
    'source_tif': source_tif,
    'source_gpkg': source_gpkg,
    'source_layer': 'craters',
    'selection': 'One of the densest 300x300 native-pixel windows; centers 22, 23, 24, 25.',
    'srcwin': [xoff, yoff, size, size],
    'bounds_native_metres': [xmin, ymin, xmax, ymax],
    'pixel_size_metres': [transform[1], transform[5]],
    'output_transform': [xmin, transform[1], 0.0, ymax, 0.0, transform[5]],
    'source_band_types': [band['type'] for band in info['bands']],
    'source_nodata': [band.get('noDataValue') for band in info['bands']],
    'source_crs_wkt': wkt,
    'boundary_behavior': 'Intersecting crater polygons are clipped to the raster footprint; attributes and FIDs are retained.',
}
(work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Native pixel window: xoff={xoff}, yoff={yoff}, width={size}, height={size}')
print(f'Native bounds (xmin ymin xmax ymax): {xmin} {ymin} {xmax} {ymax}')
PY
read -r xmin ymin xmax ymax < "$work_dir/bounds.txt"

# Only the requested native window is extracted. No warping or rescaling.
gdal_translate -of GTiff -epo \
    -srcwin "$xoff" "$yoff" "$size" "$size" \
    -co TILED=YES -co COMPRESS=DEFLATE -co PREDICTOR=3 \
    "$source_tif" "$work_dir/$stem.tif"

# Clip geometries, rather than merely selecting whole polygons that intersect.
# -clipdst is expressed in the output CRS, explicitly taken from the raster.
ogr2ogr -f GPKG -preserve_fid -nlt PROMOTE_TO_MULTI \
    -t_srs "$work_dir/source.wkt" \
    -clipdst "$xmin" "$ymin" "$xmax" "$ymax" \
    "$work_dir/${stem}_label_craters.gpkg" "$source_gpkg" craters

# Validate the small outputs before copying them into the destination folder.
gdalinfo -json "$work_dir/$stem.tif" > "$work_dir/output_info.json"
ogr2ogr -f GeoJSON "$work_dir/clipped_labels.geojson" \
    "$work_dir/${stem}_label_craters.gpkg" craters
python3 - "$work_dir" "$size" <<'PY'
import json
import math
import pathlib
import sys

work = pathlib.Path(sys.argv[1])
size = int(sys.argv[2])
info = json.loads((work / 'output_info.json').read_text())
manifest = json.loads((work / 'manifest.json').read_text())
if info['size'] != [size, size] or any(
    not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    for a, b in zip(info['geoTransform'], manifest['output_transform'])
):
    raise SystemExit('Output raster dimensions or native alignment are incorrect.')
if [b['type'] for b in info['bands']] != manifest['source_band_types'] or [
    b.get('noDataValue') for b in info['bands']
] != manifest['source_nodata']:
    raise SystemExit('Raster band types or NoData changed unexpectedly.')
features = json.loads((work / 'clipped_labels.geojson').read_text())['features']
if not features:
    raise SystemExit('The crop contains no crater labels; check the source GeoPackage.')

def points(coordinates):
    if coordinates and isinstance(coordinates[0], (int, float)):
        yield coordinates[:2]
    else:
        for child in coordinates:
            yield from points(child)

xmin, ymin, xmax, ymax = manifest['bounds_native_metres']
for feature in features:
    geometry = feature.get('geometry')
    if not geometry or geometry['type'] not in ('Polygon', 'MultiPolygon'):
        raise SystemExit('Unexpected empty or non-polygon label geometry.')
    for x, y in points(geometry['coordinates']):
        if not (xmin - 1e-7 <= x <= xmax + 1e-7 and ymin - 1e-7 <= y <= ymax + 1e-7):
            raise SystemExit('A clipped label extends outside the raster footprint.')
manifest['output_crater_count'] = len(features)
manifest['output_crater_ids'] = [f['properties'].get('crater_id') for f in features]
(work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Validated {size} x {size} raster and {len(features)} clipped crater labels.')
PY

cp -- "$work_dir/$stem.tif" "$work_dir/${stem}_label_craters.gpkg" "$destination/"
cp -- "$work_dir/manifest.json" "$destination/$stem.json"
printf '\nSaved on HPC in: %s\n' "$destination"
printf '  %s\n' "$stem.tif" "${stem}_label_craters.gpkg" "$stem.json"
