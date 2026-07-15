#!/usr/bin/env bash
#SBATCH --job-name=sem_seg_finetune
#SBATCH --partition=compute
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=24:00:00
#SBATCH --output=notebooks/graha-flm-finetuning/logs/sem_seg_finetune_%j.out
#SBATCH --error=notebooks/graha-flm-finetuning/logs/sem_seg_finetune_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/notebooks/graha-flm-finetuning/lfm_seg_finetuning_direct.py" ]]; then
  SCRIPT_DIR="${SUBMIT_DIR}/notebooks/graha-flm-finetuning"
elif [[ -f "${SUBMIT_DIR}/lfm_seg_finetuning_direct.py" ]]; then
  SCRIPT_DIR="${SUBMIT_DIR}"
else
  echo "Could not locate lfm_seg_finetuning_direct.py from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${SCRIPT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

python lfm_seg_finetuning_direct.py "$@"

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
