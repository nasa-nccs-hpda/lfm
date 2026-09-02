#!/usr/bin/env bash
#SBATCH --job-name=tiling_wac_json
#SBATCH --partition=grace
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --output=tiling_wac_json_%j.out
#SBATCH --error=tiling_wac_json_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f "${SUBMIT_DIR}/notebooks/tiling_example.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/tiling_example.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not locate notebooks/tiling_example.py from: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or its notebooks/ directory." >&2
  exit 1
fi

NOTEBOOK_DIR="${REPO_DIR}/notebooks"
CONFIG_PATH="${1:-${NOTEBOOK_DIR}/bad_cube_aoi.json}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  if [[ -f "${SUBMIT_DIR}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${SUBMIT_DIR}/${CONFIG_PATH}"
  else
    CONFIG_PATH="${NOTEBOOK_DIR}/${CONFIG_PATH}"
  fi
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "JSON configuration does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Configuration: ${CONFIG_PATH}"

"${APPTAINER_BIN}" exec \
  --nv \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${NOTEBOOK_DIR}" \
  "${CONTAINER_PATH}" \
  python -u tiling_example.py "${CONFIG_PATH}"

echo "WAC/static JSON tiling job completed successfully."
