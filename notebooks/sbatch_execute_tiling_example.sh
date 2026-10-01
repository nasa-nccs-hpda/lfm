#!/usr/bin/env bash
#SBATCH --job-name=tiling_notebook
#SBATCH --partition=grace
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/tiling_notebook_%j.out
#SBATCH --error=scripts/logs/tiling_notebook_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f "${SUBMIT_DIR}/notebooks/tiling_example.ipynb" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/tiling_example.ipynb" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not locate notebooks/tiling_example.ipynb from ${SUBMIT_DIR}." >&2
  echo "Submit this script from the repository root or notebooks directory." >&2
  exit 1
fi

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
KERNEL_NAME="${KERNEL_NAME:-python3}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EXECUTION_DIR="${EXECUTION_DIR:-${REPO_DIR}/outputs/tiling_notebook_execution/${RUN_ID}}"
JUPYTER_STATE_DIR="${EXECUTION_DIR}/jupyter"
EXECUTED_NOTEBOOK="tiling_example_executed.ipynb"

mkdir -p \
  "${REPO_DIR}/scripts/logs" \
  "${EXECUTION_DIR}" \
  "${JUPYTER_STATE_DIR}/data" \
  "${JUPYTER_STATE_DIR}/runtime" \
  "${JUPYTER_STATE_DIR}/config" \
  "${JUPYTER_STATE_DIR}/ipython" \
  "${JUPYTER_STATE_DIR}/matplotlib"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-not submitted through Slurm}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Kernel: ${KERNEL_NAME}"
echo "Executed notebook: ${EXECUTION_DIR}/${EXECUTED_NOTEBOOK}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}/notebooks" \
  "${CONTAINER_PATH}" \
  env \
    JUPYTER_DATA_DIR="${JUPYTER_STATE_DIR}/data" \
    JUPYTER_RUNTIME_DIR="${JUPYTER_STATE_DIR}/runtime" \
    JUPYTER_CONFIG_DIR="${JUPYTER_STATE_DIR}/config" \
    IPYTHONDIR="${JUPYTER_STATE_DIR}/ipython" \
    MPLCONFIGDIR="${JUPYTER_STATE_DIR}/matplotlib" \
  jupyter nbconvert \
    --to notebook \
    --execute tiling_example.ipynb \
    --output "${EXECUTED_NOTEBOOK}" \
    --output-dir "${EXECUTION_DIR}" \
    --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
    --ExecutePreprocessor.timeout=-1

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"
printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Tiling notebook execution passed."
echo "Executed notebook: ${EXECUTION_DIR}/${EXECUTED_NOTEBOOK}"
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
