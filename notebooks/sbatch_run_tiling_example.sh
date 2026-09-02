#!/usr/bin/env bash
#SBATCH --job-name=tiling_notebook
#SBATCH --partition=grace
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=tiling_notebook_%j.out
#SBATCH --error=tiling_notebook_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f "${SUBMIT_DIR}/notebooks/tiling_example.ipynb" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/tiling_example.ipynb" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not locate notebooks/tiling_example.ipynb from: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or its notebooks/ directory." >&2
  exit 1
fi

NOTEBOOK_DIR="${REPO_DIR}/notebooks"

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
KERNEL_NAME="${KERNEL_NAME:-lfm}"
EXECUTED_NOTEBOOK_DIR="${EXECUTED_NOTEBOOK_DIR:-${NOTEBOOK_DIR}/executed}"
EXECUTED_NOTEBOOK_NAME="tiling_example_executed_${SLURM_JOB_ID:-manual}"

mkdir -p "${EXECUTED_NOTEBOOK_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Kernel: ${KERNEL_NAME}"
echo "Executed notebook: ${EXECUTED_NOTEBOOK_DIR}/${EXECUTED_NOTEBOOK_NAME}.ipynb"

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${NOTEBOOK_DIR}" \
  "${CONTAINER_PATH}" \
  jupyter nbconvert \
    --to notebook \
    --execute tiling_example.ipynb \
    --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
    --ExecutePreprocessor.timeout=-1 \
    --output "${EXECUTED_NOTEBOOK_NAME}" \
    --output-dir "${EXECUTED_NOTEBOOK_DIR}" \
    "$@"

echo "Tiling notebook completed successfully."
