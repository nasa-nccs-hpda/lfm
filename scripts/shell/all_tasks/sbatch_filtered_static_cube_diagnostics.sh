#!/usr/bin/env bash
#SBATCH --job-name=static_cube_diag
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/static_cube_diag_%j.out
#SBATCH --error=scripts/logs/static_cube_diag_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/python/run_filtered_static_cube_diagnostics.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../python/run_filtered_static_cube_diagnostics.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../../scripts/python/run_filtered_static_cube_diagnostics.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate scripts/python/run_filtered_static_cube_diagnostics.py from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/explore,/explore/nobackup,/explore/nobackup/people/ajkerr1}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Bind paths: ${APPTAINER_BIND_PATHS}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  "${CONTAINER_PATH}" \
  python -u scripts/python/run_filtered_static_cube_diagnostics.py "$@"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
