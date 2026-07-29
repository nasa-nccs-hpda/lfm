#!/usr/bin/env bash
#SBATCH --job-name=sem_seg_plot
#SBATCH --partition=compute
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=12:00:00
#SBATCH --output=scripts/logs/semantic_checkpoint_comparison_plot_%j.out
#SBATCH --error=scripts/logs/semantic_checkpoint_comparison_plot_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/python/semantic_seg/semantic_checkpoint_comparison_plot.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../python/semantic_seg/semantic_checkpoint_comparison_plot.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../../scripts/python/semantic_seg/semantic_checkpoint_comparison_plot.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate scripts/python/semantic_seg/semantic_checkpoint_comparison_plot.py from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${REPO_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

module load miniforge
mamba activate graha-lunar-fm
python -u scripts/python/semantic_seg/semantic_checkpoint_comparison_plot.py "$@"

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
