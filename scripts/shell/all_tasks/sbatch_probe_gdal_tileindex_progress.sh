#!/usr/bin/env bash
#SBATCH --job-name=probe_tileindex_progress
#SBATCH --partition=grace
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=scripts/logs/probe_tileindex_progress_%j.out
#SBATCH --error=scripts/logs/probe_tileindex_progress_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/shell/all_tasks/sbatch_probe_gdal_tileindex_progress.sh"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_probe_gdal_tileindex_progress.sh" ]]; then
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
REPORT_DIR="${REPO_DIR}/test_outputs"
REPORT_PATH="${REPORT_DIR}/gdal_tileindex_progress_probe_${SLURM_JOB_ID:-manual}.json"

cd "${REPO_DIR}"
mkdir -p scripts/logs "${REPORT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Report: ${REPORT_PATH}"
echo

APPTAINER_ARGS=(
  exec
  --bind "${APPTAINER_BIND_PATHS}"
  --bind "${REPO_DIR}"
  --pwd "${REPO_PARENT}"
  "${CONTAINER_PATH}"
)

echo "Probing gdal.TileIndex callback behavior..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python \
  lfm/scripts/python/all_tasks/probe_gdal_tileindex_progress.py \
  --report "${REPORT_PATH}"

echo
echo "Running vector-index builder tests with GDAL enabled..."
"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" python -m unittest \
  lfm.model.tests.test_vector_index_builder

echo
echo "GDAL TileIndex progress probe and focused tests completed."
