#!/usr/bin/env bash

set -euo pipefail

: "${GFFT_SEMANTIC_DATA_ROOT:?Set GFFT_SEMANTIC_DATA_ROOT to the split semantic dataset root.}"
: "${GFFT_CONFIG_PATH:?Set GFFT_CONFIG_PATH to the TerraTorch-style GFFT YAML.}"
: "${GFFT_OUTPUT_DIR:?Set GFFT_OUTPUT_DIR to the smoke-test output directory.}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/shell/semantic_seg/sbatch_semantic_gfft_single_finetune.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_semantic_gfft_single_finetune.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate repo root from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs "${GFFT_OUTPUT_DIR}"

sbatch scripts/shell/semantic_seg/sbatch_semantic_gfft_single_finetune.sh \
  --data-root "${GFFT_SEMANTIC_DATA_ROOT}" \
  --base-output-dir "${GFFT_OUTPUT_DIR}" \
  --gfft-config-path "${GFFT_CONFIG_PATH}" \
  --max-epochs 1 \
  --target-size 256 \
  --image-glob '*.tif' \
  --label-glob '*_label.npy' \
  --image-suffix '_input_nac_chip' \
  --label-suffix '_label' \
  --max-train-samples 5 \
  --max-val-samples 5 \
  --max-test-samples 5 \
  --batch-size 2 \
  --graha-batch-size 2 \
  --graha-stats-batch-size 4 \
  --band-filter 0 \
  --normalize-inputs \
  --normalization-source pretrain \
  --normalization-modality nac \
  --toy-loss-type dice \
  --graha-shape-loss-weight 0.0 \
  --disable-toy-gradient-clipping \
  --dataset-modality nac \
  --prediction-split val \
  --prediction-n-samples 2 \
  --run-epoch-test-suite \
  --epoch-test-split test \
  --epoch-test-n-samples 2 \
  --epoch-test-every-n-epochs 1 \
  "$@"
