#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sbatch_instance_seg_single_finetune.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate repo root from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

BASE_OUTPUT_DIR="${REPO_DIR}/scripts/outputs/instance_three_model_finetune"
FINAL_COMPARISON_OUTPUT_DIR=""
SUBMIT_FINAL_COMPARISON_PLOT=1
ARGS=("$@")
JOB_ARGS=()
PLOT_ARGS=()
UNSUPPORTED_FLAGS=(
  "--spatial-transform"
  "--loss-type"
  "--toy-loss-type"
  "--use-toy-shape-loss"
  "--toy-shape-loss-weight"
  "--toy-shape-loss-pad-frac"
  "--graha-shape-loss-weight"
  "--graha-shape-loss-pad-frac"
)

append_plot_arg_with_value() {
  local flag="$1"
  local value="$2"
  PLOT_ARGS+=("${flag}" "${value}")
}

append_plot_arg_with_values_until_next_flag() {
  local flag="$1"
  local start_index="$2"
  PLOT_NEXT_INDEX="${start_index}"
  PLOT_ARGS+=("${flag}")
  while [[ "${PLOT_NEXT_INDEX}" -lt "${#ARGS[@]}" && "${ARGS[${PLOT_NEXT_INDEX}]}" != --* ]]; do
    PLOT_ARGS+=("${ARGS[${PLOT_NEXT_INDEX}]}")
    PLOT_NEXT_INDEX=$((PLOT_NEXT_INDEX + 1))
  done
}

for ((i = 0; i < ${#ARGS[@]}; i++)); do
  arg="${ARGS[$i]}"
  if [[ "${arg}" == "--base-output-dir" ]]; then
    if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
      echo "--base-output-dir requires a value." >&2
      exit 2
    fi
    BASE_OUTPUT_DIR="${ARGS[$((i + 1))]}"
    i=$((i + 1))
    continue
  fi
  if [[ "${arg}" == "--final-comparison-output-dir" ]]; then
    if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
      echo "--final-comparison-output-dir requires a value." >&2
      exit 2
    fi
    FINAL_COMPARISON_OUTPUT_DIR="${ARGS[$((i + 1))]}"
    i=$((i + 1))
    continue
  fi
  if [[ "${arg}" == "--skip-final-comparison-plot" ]]; then
    SUBMIT_FINAL_COMPARISON_PLOT=0
    continue
  fi
  if [[ "${arg}" == "--toy-architecture" ]]; then
    echo "Do not pass --toy-architecture to this three-model orchestrator." >&2
    echo "It submits fixed Toy jobs for mask2former and dino-terratorch-mask-rcnn." >&2
    exit 2
  fi
  for unsupported_flag in "${UNSUPPORTED_FLAGS[@]}"; do
    if [[ "${arg}" == "${unsupported_flag}" ]]; then
      echo "Unsupported instance segmentation argument: ${unsupported_flag}" >&2
      echo "Semantic-only spatial/loss/shape-loss options do not affect instance segmentation; remove them from the instance command." >&2
      exit 2
    fi
  done
  JOB_ARGS+=("${arg}")

  case "${arg}" in
    --normalize-inputs|--ignore-nodata-in-loss)
      PLOT_ARGS+=("${arg}")
      ;;
    --prediction-n-samples)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "${arg} requires a value." >&2
        exit 2
      fi
      append_plot_arg_with_value "--n-samples" "${ARGS[$((i + 1))]}"
      ;;
    --prediction-score-threshold)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "${arg} requires a value." >&2
        exit 2
      fi
      append_plot_arg_with_value "--score-threshold" "${ARGS[$((i + 1))]}"
      ;;
    --band-filter)
      append_plot_arg_with_values_until_next_flag "${arg}" "$((i + 1))"
      if [[ "${PLOT_NEXT_INDEX}" -eq $((i + 1)) ]]; then
        echo "--band-filter requires at least one value." >&2
        exit 2
      fi
      ;;
    --mask-shift)
      if [[ $((i + 2)) -ge ${#ARGS[@]} ]]; then
        echo "--mask-shift requires two values." >&2
        exit 2
      fi
      PLOT_ARGS+=("${arg}" "${ARGS[$((i + 1))]}" "${ARGS[$((i + 2))]}")
      ;;
    --data-root|--dino-checkpoint|--graha-pretrain-dir|--graha-input-modality-mode|\
    --graha-vis-uv-merge-method|--target-size|--image-glob|--label-glob|\
    --image-suffix|--label-suffix|--batch-size|--num-workers|\
    --graha-stats-batch-size|--graha-batch-size|--graha-num-workers|\
    --normalization-source|--normalization-modality|--graha-backbone-lr|\
    --graha-head-lr|--graha-layer-decay|--graha-weight-decay|\
    --graha-warmup-steps|--graha-anchor-sizes|--graha-anchor-aspect-ratios|\
    --graha-score-threshold|--prediction-split|--nodata-ignore-index|--seed)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "${arg} requires a value." >&2
        exit 2
      fi
      append_plot_arg_with_value "${arg}" "${ARGS[$((i + 1))]}"
      ;;
  esac
done

TIMESTAMP="$(date +date_%Y_%m_%d-time_%H_%M_%S)"
RUN_ROOT="${BASE_OUTPUT_DIR%/}/${TIMESTAMP}"
MASK2FORMER_OUTPUT_DIR="${RUN_ROOT}/toy_mask2former"
TOY_TERRATORCH_OUTPUT_DIR="${RUN_ROOT}/toy_dino_terratorch_mask_rcnn"
GRAHA_OUTPUT_DIR="${RUN_ROOT}/graha_mask_rcnn"
if [[ -z "${FINAL_COMPARISON_OUTPUT_DIR}" ]]; then
  FINAL_COMPARISON_OUTPUT_DIR="${RUN_ROOT}/final_checkpoint_comparison"
fi
mkdir -p "${MASK2FORMER_OUTPUT_DIR}" "${TOY_TERRATORCH_OUTPUT_DIR}" "${GRAHA_OUTPUT_DIR}"

echo "Submitting three instance fine-tuning jobs"
echo "Shared run root: ${RUN_ROOT}"
echo "Toy Mask2Former output: ${MASK2FORMER_OUTPUT_DIR}"
echo "Toy DINO TerraTorch Mask R-CNN output: ${TOY_TERRATORCH_OUTPUT_DIR}"
echo "Graha Mask R-CNN output: ${GRAHA_OUTPUT_DIR}"
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  echo "Final checkpoint comparison output: ${FINAL_COMPARISON_OUTPUT_DIR}"
else
  echo "Final checkpoint comparison plot submission: skipped"
fi

MASK2FORMER_JOB="$(
  sbatch --parsable scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh \
    --model toy \
    "${JOB_ARGS[@]}" \
    --toy-architecture mask2former \
    --base-output-dir "${MASK2FORMER_OUTPUT_DIR}" \
    --run-epoch-test-suite
)"
echo "Submitted Toy Mask2Former job: ${MASK2FORMER_JOB}"

TOY_TERRATORCH_JOB="$(
  sbatch --parsable scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh \
    --model toy \
    "${JOB_ARGS[@]}" \
    --toy-architecture dino-terratorch-mask-rcnn \
    --base-output-dir "${TOY_TERRATORCH_OUTPUT_DIR}" \
    --run-epoch-test-suite
)"
echo "Submitted Toy DINO TerraTorch Mask R-CNN job: ${TOY_TERRATORCH_JOB}"

GRAHA_JOB="$(
  sbatch --parsable scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh \
    --model graha \
    "${JOB_ARGS[@]}" \
    --base-output-dir "${GRAHA_OUTPUT_DIR}" \
    --run-epoch-test-suite
)"
echo "Submitted Graha Mask R-CNN job: ${GRAHA_JOB}"

FINAL_COMPARISON_JOB=""
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  FINAL_COMPARISON_JOB="$(
    sbatch --parsable \
      --dependency=afterok:${MASK2FORMER_JOB}:${TOY_TERRATORCH_JOB}:${GRAHA_JOB} \
      scripts/shell/instance_seg/sbatch_instance_checkpoint_comparison_plot.sh \
      --run-root "${RUN_ROOT}" \
      --output-dir "${FINAL_COMPARISON_OUTPUT_DIR}" \
      "${PLOT_ARGS[@]}"
  )"
  echo "Submitted final checkpoint comparison plot job: ${FINAL_COMPARISON_JOB}"
  echo "Plot job dependency: afterok:${MASK2FORMER_JOB}:${TOY_TERRATORCH_JOB}:${GRAHA_JOB}"
fi

cat > "${RUN_ROOT}/submitted_jobs.txt" <<EOF
run_root=${RUN_ROOT}
toy_mask2former_output_dir=${MASK2FORMER_OUTPUT_DIR}
toy_mask2former_job=${MASK2FORMER_JOB}
toy_dino_terratorch_mask_rcnn_output_dir=${TOY_TERRATORCH_OUTPUT_DIR}
toy_dino_terratorch_mask_rcnn_job=${TOY_TERRATORCH_JOB}
graha_mask_rcnn_output_dir=${GRAHA_OUTPUT_DIR}
graha_mask_rcnn_job=${GRAHA_JOB}
final_comparison_output_dir=${FINAL_COMPARISON_OUTPUT_DIR}
final_comparison_job=${FINAL_COMPARISON_JOB}
EOF

echo "Wrote job manifest: ${RUN_ROOT}/submitted_jobs.txt"
