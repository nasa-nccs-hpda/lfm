#!/usr/bin/env bash
#SBATCH --job-name=lfm_dependencies
#SBATCH --partition=compute
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=container_dependencies_%j.log

# Submit from the repo root:
#   sbatch test_container.sbatch /absolute/path/to/sandbox-or-image.sif
# Override partition/account/resources with sbatch flags as needed for ARM64.
set -euo pipefail

CONTAINER_PATH="${1:?Usage: sbatch test_container.sbatch /path/to/container}"
shift
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
test -f "${REPO_DIR}/test_container_dependencies.py"
test -e "${CONTAINER_PATH}"
CONTAINER_PATH="$(realpath "${CONTAINER_PATH}")"

echo "Container: ${CONTAINER_PATH}"
echo "Host: $(hostname) ($(uname -m)); job: ${SLURM_JOB_ID:-interactive}"
echo "Repository: ${REPO_DIR}"

# Preserve Slurm's GPU visibility while excluding host Python/Conda settings.
gpu_env=()
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    gpu_env=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
"${APPTAINER_BIN:-apptainer}" exec --cleanenv --nv \
    --bind "${REPO_DIR}:/lfm-repo:ro" --pwd /lfm-repo \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env MPLBACKEND=Agg --env HF_HUB_OFFLINE=1 \
    "${gpu_env[@]}" "${CONTAINER_PATH}" \
    python -u /lfm-repo/test_container_dependencies.py "$@"
