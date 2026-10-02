#!/usr/bin/env bash
#SBATCH --job-name=shared_tile_indexes
#SBATCH --partition=grace
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/shared_tile_indexes_%j.out
#SBATCH --error=scripts/logs/shared_tile_indexes_%j.err

set -euo pipefail
umask 0002

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/shell/all_tasks/sbatch_create_shared_default_indexes.sh"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_create_shared_default_indexes.sh" ]]; then
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
PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-/explore/nobackup/projects/lfm}"
INDEX_NAME="${INDEX_NAME:-output_index.gpkg}"
WORKER_COUNT="${WORKER_COUNT:-}"
REPORT_PATH="${REPO_DIR}/test_outputs/shared_default_indexes_${SLURM_JOB_ID:-manual}.json"

WORKER_ARGS=()
if [[ -n "${WORKER_COUNT}" ]]; then
  WORKER_ARGS=(--worker-count "${WORKER_COUNT}")
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs test_outputs

echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Project data root: ${PROJECT_DATA_DIR}"
echo "Shared index filename: ${INDEX_NAME}"
echo "Worker override: ${WORKER_COUNT:-SLURM_CPUS_PER_TASK (${SLURM_CPUS_PER_TASK:-unset})}"
echo "Report: ${REPORT_PATH}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_PARENT}" \
  "${CONTAINER_PATH}" \
  env PYTHONPATH="${REPO_PARENT}" \
  python lfm/scripts/python/all_tasks/create_shared_default_indexes.py \
    --project-data-dir "${PROJECT_DATA_DIR}" \
    --index-name "${INDEX_NAME}" \
    "${WORKER_ARGS[@]}" \
    --report "${REPORT_PATH}"

echo
echo "Shared default tiling indexes created and validated."
