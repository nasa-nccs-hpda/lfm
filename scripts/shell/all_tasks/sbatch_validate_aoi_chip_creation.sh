#!/usr/bin/env bash
#SBATCH --job-name=validate_aoi_chips
#SBATCH --partition=grace
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/validate_aoi_chips_%j.out
#SBATCH --error=scripts/logs/validate_aoi_chips_%j.err

# Submit from the repository root. Use inline environment assignments (no &&):
# LABEL_GPKG=/path/labels.gpkg SOURCE_RASTER=/path/product.prj.vis.mos.tif \
# WAC_PRODUCT_ID=M123CE AOI_1="N W S E" AOI_2="N W S E" sbatch <this-script>
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/validate_aoi_chip_creation.py"
[[ -f "${REPO_DIR}/${SCRIPT_REL}" ]] || { echo "Submit from the repository root." >&2; exit 1; }
: "${LABEL_GPKG:?Set LABEL_GPKG to the finished crater GeoPackage}"
: "${SOURCE_RASTER:?Set SOURCE_RASTER to the original WAC VIS TIFF}"
: "${WAC_PRODUCT_ID:?Set WAC_PRODUCT_ID to the imagery product prefix}"
: "${AOI_1:?Set AOI_1 to NORTH WEST SOUTH EAST}"
: "${AOI_2:?Set AOI_2 to NORTH WEST SOUTH EAST}"
(( ${SLURM_CPUS_PER_TASK:-2} >= 2 )) || { echo "Two allocated CPUs are required." >&2; exit 1; }
read -r -a FIRST_AOI <<< "${AOI_1}"
read -r -a SECOND_AOI <<< "${AOI_2}"
(( ${#FIRST_AOI[@]} == 4 && ${#SECOND_AOI[@]} == 4 )) || { echo "Each AOI needs four numbers." >&2; exit 1; }
CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/notebooks/outputs/chip_a5_${SLURM_JOB_ID:-manual}}"
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--label-gpkg "${LABEL_GPKG}" --source-raster "${SOURCE_RASTER}"
      --product-id "${WAC_PRODUCT_ID}" --aoi "${FIRST_AOI[@]}" --aoi "${SECOND_AOI[@]}"
      --output-root "${OUTPUT_ROOT}" --layer "${LABEL_LAYER:-craters}")
if [[ -n "${WAC_DATA_DIR:-}" ]]; then ARGS+=(--wac-data-dir "${WAC_DATA_DIR}"); fi
if [[ -n "${STATIC_DATA_DIR:-}" ]]; then ARGS+=(--static-data-dir "${STATIC_DATA_DIR}"); fi
cd "${REPO_DIR}"
"${APPTAINER_BIN:-apptainer}" exec \
  --bind "${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}" \
  --bind "${REPO_DIR}" --pwd "${REPO_DIR}" "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" "${ARGS[@]}" "$@"
echo "A5 checks passed. Report: ${OUTPUT_ROOT}/a5_validation.json"
echo "Review overlays: ${OUTPUT_ROOT}/inspection_plots"
