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
FINAL_COMPARISON_OUTPUT_DIR=""
SUBMIT_FINAL_COMPARISON_PLOT=1
ARGS=("$@")
JOB_ARGS=()
PLOT_ARGS=()

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
    --band-filter)
      append_plot_arg_with_values_until_next_flag "${arg}" "$((i + 1))"
      if [[ "${PLOT_NEXT_INDEX}" -eq $((i + 1)) ]]; then
        echo "--band-filter requires at least one value." >&2
        exit 2
      fi
      ;;
    --data-root|--dataset-modality|--dino-checkpoint|--graha-pretrain-dir|--graha-input-modality-mode|\
    --graha-vis-uv-merge-method|--target-size|--semantic-label-source|\
    --image-glob|--label-glob|--image-suffix|--label-suffix|--batch-size|\
    --num-workers|--graha-stats-batch-size|--graha-batch-size|\
    --graha-num-workers|--normalization-source|--normalization-modality|\
    --graha-shape-loss-weight|--graha-shape-loss-pad-frac|--prediction-split|\
    --nodata-ignore-index|--seed)
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
if [[ -z "${FINAL_COMPARISON_OUTPUT_DIR}" ]]; then
  FINAL_COMPARISON_OUTPUT_DIR="${RUN_ROOT}/final_checkpoint_comparison"
fi
mkdir -p "${RUN_ROOT}"

echo "Submitting parallel semantic fine-tuning jobs"
echo "Shared output root: ${RUN_ROOT}"
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  echo "Final checkpoint comparison output: ${FINAL_COMPARISON_OUTPUT_DIR}"
else
  echo "Final checkpoint comparison plot submission: skipped"
fi

TOY_JOB="$(
  sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
    --model toy \
    "${JOB_ARGS[@]}" \
    --base-output-dir "${RUN_ROOT}" \
    --run-epoch-test-suite
)"
echo "Submitted semantic Toy/DINO job: ${TOY_JOB}"

GRAHA_JOB="$(
  sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
    --model graha \
    "${JOB_ARGS[@]}" \
    --base-output-dir "${RUN_ROOT}" \
    --run-epoch-test-suite
)"
echo "Submitted semantic Graha job: ${GRAHA_JOB}"

FINAL_COMPARISON_JOB=""
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  FINAL_COMPARISON_JOB="$(
    sbatch --parsable \
      --dependency=afterok:${TOY_JOB}:${GRAHA_JOB} \
      scripts/shell/semantic_seg/sbatch_semantic_checkpoint_comparison_plot.sh \
      --run-root "${RUN_ROOT}" \
      --output-dir "${FINAL_COMPARISON_OUTPUT_DIR}" \
      "${PLOT_ARGS[@]}"
  )"
  echo "Submitted final semantic checkpoint comparison plot job: ${FINAL_COMPARISON_JOB}"
  echo "Plot job dependency: afterok:${TOY_JOB}:${GRAHA_JOB}"
fi

cat > "${RUN_ROOT}/submitted_jobs.txt" <<EOF
run_root=${RUN_ROOT}
toy_job=${TOY_JOB}
graha_job=${GRAHA_JOB}
final_comparison_output_dir=${FINAL_COMPARISON_OUTPUT_DIR}
final_comparison_job=${FINAL_COMPARISON_JOB}
EOF

echo "Wrote job manifest: ${RUN_ROOT}/submitted_jobs.txt"
