#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SWEEP_SCRIPT="${SCRIPT_DIR}/sbatch_semantic_train_then_checkpoint_sweep.sh"

DATA_ROOT="${DATA_ROOT:-/panfs/ccds02/nobackup/projects/lfm/model_inputs/300_300_inputs/full_model_inst_seg_v2}"
BASE_OUTPUT_PARENT="${BASE_OUTPUT_PARENT:-/explore/nobackup/people/ajkerr1/Lunar_FM}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAHA_BATCH_SIZE="${GRAHA_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GRAHA_NUM_WORKERS="${GRAHA_NUM_WORKERS:-10}"
GRAHA_STATS_BATCH_SIZE="${GRAHA_STATS_BATCH_SIZE:-16}"
SWEEP_MAX_SAMPLES="${SWEEP_MAX_SAMPLES:-100}"
PREDICTION_N_SAMPLES="${PREDICTION_N_SAMPLES:-5}"
MIN_FREE_GB="${MIN_FREE_GB:-500}"

check_output_space() {
  mkdir -p "${BASE_OUTPUT_PARENT}"
  local available_kb
  local available_gb
  available_kb="$(df -Pk "${BASE_OUTPUT_PARENT}" | awk 'NR == 2 {print $4}')"
  available_gb="$((available_kb / 1024 / 1024))"

  echo "Output parent: ${BASE_OUTPUT_PARENT}"
  echo "Available filesystem space: ${available_gb} GB"
  echo "Required minimum free space: ${MIN_FREE_GB} GB"

  if (( available_gb < MIN_FREE_GB )); then
    echo "Not enough free filesystem space for four checkpoint-heavy experiments." >&2
    echo "Set BASE_OUTPUT_PARENT to a larger filesystem or lower MIN_FREE_GB if you intentionally want to proceed." >&2
    exit 1
  fi
}

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --max-epochs "${MAX_EPOCHS}"
  --target-size 256
  --spatial-transform crop
  --semantic-label-source instance
  --batch-size "${BATCH_SIZE}"
  --graha-batch-size "${GRAHA_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --graha-num-workers "${GRAHA_NUM_WORKERS}"
  --graha-stats-batch-size "${GRAHA_STATS_BATCH_SIZE}"
  --normalize-inputs
  --toy-loss-type dice
  --disable-toy-gradient-clipping
  --graha-input-modality-mode vis-uv
  --graha-vis-uv-merge-method mean
  --prediction-split val
  --prediction-n-samples "${PREDICTION_N_SAMPLES}"
  --sweep-split test
  --sweep-max-samples "${SWEEP_MAX_SAMPLES}"
)

submit_experiment() {
  local name="$1"
  local normalization_source="$2"
  local use_shape_loss="$3"
  local output_dir="${BASE_OUTPUT_PARENT}/${name}_epochs-${MAX_EPOCHS}_test-${SWEEP_MAX_SAMPLES}"
  shift 3

  local args=(
    --time="${TIME_LIMIT}"
    "${TRAIN_SWEEP_SCRIPT}"
    --base-output-dir "${output_dir}"
    "${COMMON_ARGS[@]}"
    --normalization-source "${normalization_source}"
  )

  if [[ "${use_shape_loss}" == "true" ]]; then
    args+=(
      --use-toy-shape-loss
      --toy-shape-loss-weight 0.05
      --toy-shape-loss-pad-frac 0.3
      --graha-shape-loss-weight 0.05
      --graha-shape-loss-pad-frac 0.3
    )
  else
    args+=(
      --graha-shape-loss-weight 0.0
    )
  fi

  args+=("$@")

  echo
  echo "Submitting ${name}"
  echo "  output: ${output_dir}"
  echo "  normalization: ${normalization_source}"
  echo "  spatial+dice: ${use_shape_loss}"
  sbatch "${args[@]}"
}

# submit_experiment \
#   "exp01_semseg_from-instlabels_7band-wac_crop256_toy-finetune-norm_dice_train-sweep" \
#   "finetune" \
#   "false" \
#   "$@"

# submit_experiment \
#   "exp02_semseg_from-instlabels_7band-wac_crop256_terramind-pretrain-norm_dice_train-sweep" \
#   "pretrain" \
#   "false" \
#   "$@"

# submit_experiment \
#   "exp03_semseg_from-instlabels_7band-wac_crop256_toy-finetune-norm_dice-plus-spatial-w0p05_train-sweep" \
#   "finetune" \
#   "true" \
#   "$@"



submit_experiment \
  "exp04_semseg_from-instlabels_7band-wac_crop256_terramind-pretrain-norm_dice-plus-spatial-w0p05_train-sweep" \
  "pretrain" \
  "true" \
  "$@"
