#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_semantic_seg_single_finetune.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate repo root from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

BASE_OUTPUT_DIR="${REPO_DIR}/scripts/outputs/semantic_parallel_finetune"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--base-output-dir" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    BASE_OUTPUT_DIR="${ARGS[$((i + 1))]}"
  fi
done

TIMESTAMP="$(date +date_%Y_%m_%d-time_%H_%M_%S)"
RUN_ROOT="${BASE_OUTPUT_DIR%/}/${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"

echo "Submitting parallel semantic fine-tuning jobs"
echo "Shared output root: ${RUN_ROOT}"

TOY_JOB="$(
  sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
    --model toy \
    "${ARGS[@]}" \
    --base-output-dir "${RUN_ROOT}" \
    --run-epoch-test-suite
)"
echo "Submitted semantic Toy/DINO job: ${TOY_JOB}"

GRAHA_JOB="$(
  sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
    --model graha \
    "${ARGS[@]}" \
    --base-output-dir "${RUN_ROOT}" \
    --run-epoch-test-suite
)"
echo "Submitted semantic Graha job: ${GRAHA_JOB}"
