#!/usr/bin/env bash
#SBATCH --job-name=real_index_progress
#SBATCH --partition=grace
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=scripts/logs/real_index_progress_%j.out
#SBATCH --error=scripts/logs/real_index_progress_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/shell/all_tasks/sbatch_validate_real_data_index_progress.sh"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_validate_real_data_index_progress.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate the LFM repository from: ${SUBMIT_DIR}" >&2
  echo "Submit this script from the repository root or its own directory." >&2
  exit 1
fi

REPO_PARENT="$(dirname "${REPO_DIR}")"
CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
SOURCE_DIR="${SOURCE_DIR:-/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites}"
IMAGE_GLOB="${IMAGE_GLOB:-*.tif}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-8}"
WORK_DIR="${REPO_DIR}/test_outputs/real_data_index_progress_${SLURM_JOB_ID:-manual}"
REPORT_PATH="${WORK_DIR}/report.json"

cd "${REPO_DIR}"
mkdir -p scripts/logs test_outputs

echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Real-data source: ${SOURCE_DIR}"
echo "Image glob: ${IMAGE_GLOB}"
echo "Sample limit: ${SAMPLE_LIMIT}"
echo "Work directory: ${WORK_DIR}"
echo "Report: ${REPORT_PATH}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_PARENT}" \
  "${CONTAINER_PATH}" \
  python lfm/scripts/python/all_tasks/validate_real_data_index_progress.py \
    --source-dir "${SOURCE_DIR}" \
    --image-glob "${IMAGE_GLOB}" \
    --limit "${SAMPLE_LIMIT}" \
    --work-dir "${WORK_DIR}" \
    --report "${REPORT_PATH}"

echo
echo "Real-data raster-index progress validation completed."
