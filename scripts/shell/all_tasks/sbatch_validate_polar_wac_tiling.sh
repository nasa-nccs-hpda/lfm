#!/usr/bin/env bash
#SBATCH --job-name=polar_wac_tiling
#SBATCH --partition=grace
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/validate_polar_wac_tiling_%j.out
#SBATCH --error=scripts/logs/validate_polar_wac_tiling_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/validate_polar_wac_tiling.py"

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
mkdir -p scripts/logs test_outputs

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
SOURCE_PATH="${SOURCE_PATH:-/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/LROC/WAC/wac_glob_morf_mos/WAC_GLOBAL_P900N0000_100M.eqc.iau2.LPS_N.vrt}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-/explore/nobackup/people/${USER}/lfm_polar_wac_validation/${RUN_ID}}"
REPORT_PATH="${REPORT_PATH:-${WORK_DIR}/validation_report.json}"

export GDAL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Source VRT: ${SOURCE_PATH}"
echo "Work directory: ${WORK_DIR}"
echo "Report: ${REPORT_PATH}"
echo "GDAL threads: ${GDAL_NUM_THREADS}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" \
    --source-path "${SOURCE_PATH}" \
    --work-dir "${WORK_DIR}" \
    --report "${REPORT_PATH}" \
    "$@"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"
printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Polar WAC tiling validation completed successfully."
echo "Report: ${REPORT_PATH}"
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
