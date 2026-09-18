#!/usr/bin/env bash
#SBATCH --job-name=validate_nac_tiling
#SBATCH --partition=grace
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/validate_nac_tiling_%j.out
#SBATCH --error=scripts/logs/validate_nac_tiling_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/validate_nac_tiling.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  echo "Submit this script from the repository root or its own directory." >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
OUTPUT_DIR="${OUTPUT_DIR:-/explore/nobackup/people/${USER}/lfm_tiling_validation/t6_2_nac_${SLURM_JOB_ID:-manual}}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "NAC tiling validation completed successfully."
echo "Report: ${OUTPUT_DIR}/validation_report.json"
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
