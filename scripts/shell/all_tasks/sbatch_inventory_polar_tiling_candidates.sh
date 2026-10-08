#!/usr/bin/env bash
#SBATCH --job-name=polar_data_inventory
#SBATCH --partition=grace
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/polar_data_inventory_%j.out
#SBATCH --error=scripts/logs/polar_data_inventory_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/inventory_polar_tiling_candidates.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  echo "Submit this script from the repository root or its own directory." >&2
  exit 1
fi

DEFAULT_CONTAINER="/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet"
CONTAINER_PATH="${CONTAINER_PATH:-${DEFAULT_CONTAINER}}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
DEFAULT_ROOTS="/explore/nobackup/projects/lfm/data"
DEFAULT_ROOTS+=":/explore/nobackup/projects/lfm/processed_data"
DEFAULT_ROOTS+=":/explore/nobackup/projects/lfm/rawdata"
ROOTS="${ROOTS:-${DEFAULT_ROOTS}}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
EDGE_SAMPLES="${EDGE_SAMPLES:-21}"
MAX_FILES_PER_DIRECTORY="${MAX_FILES_PER_DIRECTORY:-0}"
MAX_TOTAL_FILES="${MAX_TOTAL_FILES:-0}"
INCLUDE_NON_CANDIDATES="${INCLUDE_NON_CANDIDATES:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_REPORT="${REPO_DIR}/test_outputs"
DEFAULT_REPORT+="/polar_tiling_candidates_${SLURM_JOB_ID:-manual}.json"
REPORT_PATH="${REPORT_PATH:-${DEFAULT_REPORT}}"

cd "${REPO_DIR}"
mkdir -p scripts/logs test_outputs

ROOT_ARGS=()
IFS=':' read -r -a ROOT_VALUES <<< "${ROOTS}"
for ROOT_PATH in "${ROOT_VALUES[@]}"; do
  ROOT_ARGS+=(--root "${ROOT_PATH}")
done

OPTIONAL_ARGS=()
if [[ "${INCLUDE_NON_CANDIDATES}" == "1" ]]; then
  OPTIONAL_ARGS+=(--include-non-candidates)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  OPTIONAL_ARGS+=(--dry-run)
fi

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Roots: ${ROOTS}"
echo "Workers: ${WORKERS}"
echo "Edge samples: ${EDGE_SAMPLES}"
echo "Per-directory limit: ${MAX_FILES_PER_DIRECTORY} (0 means all)"
echo "Global limit: ${MAX_TOTAL_FILES} (0 means all)"
echo "Report: ${REPORT_PATH}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u lfm/${SCRIPT_REL} \
    "${ROOT_ARGS[@]}" \
    --workers "${WORKERS}" \
    --edge-samples "${EDGE_SAMPLES}" \
    --max-files-per-directory "${MAX_FILES_PER_DIRECTORY}" \
    --max-total-files "${MAX_TOTAL_FILES}" \
    --report "${REPORT_PATH}" \
    "${OPTIONAL_ARGS[@]}"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Polar tiling candidate inventory completed."
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
echo "Report: ${REPORT_PATH}"
