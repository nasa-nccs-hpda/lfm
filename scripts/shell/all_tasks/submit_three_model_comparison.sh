#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/submit_three_model_comparison.sh" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate repo root from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

TASK=""
BASE_OUTPUT_DIR="${REPO_DIR}/scripts/outputs/three_model_comparison"
FINAL_COMPARISON_OUTPUT_DIR=""
SUBMIT_FINAL_COMPARISON_PLOT=1
TOY_ARCHITECTURE="mask2former"
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
  case "${arg}" in
    --task)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "--task requires semantic or instance." >&2
        exit 2
      fi
      TASK="${ARGS[$((i + 1))]}"
      i=$((i + 1))
      continue
      ;;
    --base-output-dir)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "--base-output-dir requires a value." >&2
        exit 2
      fi
      BASE_OUTPUT_DIR="${ARGS[$((i + 1))]}"
      i=$((i + 1))
      continue
      ;;
    --final-comparison-output-dir)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "--final-comparison-output-dir requires a value." >&2
        exit 2
      fi
      FINAL_COMPARISON_OUTPUT_DIR="${ARGS[$((i + 1))]}"
      i=$((i + 1))
      continue
      ;;
    --skip-final-comparison-plot)
      SUBMIT_FINAL_COMPARISON_PLOT=0
      continue
      ;;
    --model|--models)
      echo "Do not pass ${arg} to this orchestrator; it always submits toy, graha, and gfft." >&2
      exit 2
      ;;
    --toy-architecture)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "--toy-architecture requires a value." >&2
        exit 2
      fi
      TOY_ARCHITECTURE="${ARGS[$((i + 1))]}"
      JOB_ARGS+=("${arg}" "${TOY_ARCHITECTURE}")
      i=$((i + 1))
      continue
      ;;
  esac

  JOB_ARGS+=("${arg}")

  case "${arg}" in
    --normalize-inputs|--toy-normalize-inputs|--ignore-nodata-in-loss)
      if [[ "${arg}" == "--toy-normalize-inputs" ]]; then
        PLOT_ARGS+=("--normalize-inputs")
      else
        PLOT_ARGS+=("${arg}")
      fi
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
    --data-root|--dataset-modality|--dino-checkpoint|--graha-pretrain-dir|\
    --gfft-config-path|--gfft-backbone-checkpoint|--graha-input-modality-mode|\
    --graha-vis-uv-merge-method|--target-size|--semantic-label-source|\
    --image-glob|--label-glob|--image-suffix|--label-suffix|--batch-size|\
    --toy-batch-size|--num-workers|--graha-stats-batch-size|--graha-batch-size|\
    --graha-num-workers|--normalization-source|--normalization-modality|\
    --graha-shape-loss-weight|--graha-shape-loss-pad-frac|--graha-backbone-lr|\
    --graha-head-lr|--graha-layer-decay|--graha-weight-decay|\
    --graha-warmup-steps|--graha-anchor-sizes|--graha-anchor-aspect-ratios|\
    --graha-score-threshold|--prediction-split|--nodata-ignore-index|--seed)
      if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
        echo "${arg} requires a value." >&2
        exit 2
      fi
      if [[ "${arg}" == "--toy-batch-size" ]]; then
        append_plot_arg_with_value "--batch-size" "${ARGS[$((i + 1))]}"
      else
        append_plot_arg_with_value "${arg}" "${ARGS[$((i + 1))]}"
      fi
      ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  echo "Pass --task semantic or --task instance." >&2
  exit 2
fi
if [[ "${TASK}" != "semantic" && "${TASK}" != "instance" ]]; then
  echo "--task must be semantic or instance, got: ${TASK}" >&2
  exit 2
fi
if [[ "${TASK}" == "semantic" && "${TOY_ARCHITECTURE}" != "mask2former" ]]; then
  echo "--toy-architecture is only valid for --task instance." >&2
  exit 2
fi

TIMESTAMP="$(date +date_%Y_%m_%d-time_%H_%M_%S)"
RUN_ROOT="${BASE_OUTPUT_DIR%/}/${TASK}/${TIMESTAMP}"
TOY_OUTPUT_DIR="${RUN_ROOT}/toy_model"
GRAHA_OUTPUT_DIR="${RUN_ROOT}/graha_model"
GFFT_OUTPUT_DIR="${RUN_ROOT}/gfft_model"
LOG_OUTPUT_DIR="${RUN_ROOT}/logs"
if [[ -z "${FINAL_COMPARISON_OUTPUT_DIR}" ]]; then
  FINAL_COMPARISON_OUTPUT_DIR="${RUN_ROOT}/final_checkpoint_comparison"
fi
mkdir -p "${TOY_OUTPUT_DIR}" "${GRAHA_OUTPUT_DIR}" "${GFFT_OUTPUT_DIR}" "${LOG_OUTPUT_DIR}"

echo "Submitting ${TASK} Toy/Graha/GFFT fine-tuning jobs"
echo "Shared run root: ${RUN_ROOT}"
echo "Toy output: ${TOY_OUTPUT_DIR}"
echo "Graha output: ${GRAHA_OUTPUT_DIR}"
echo "GFFT output: ${GFFT_OUTPUT_DIR}"
echo "Collected sbatch logs: ${LOG_OUTPUT_DIR}"
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  echo "Final checkpoint comparison output: ${FINAL_COMPARISON_OUTPUT_DIR}"
else
  echo "Final checkpoint comparison plot submission: skipped"
fi

if [[ "${TASK}" == "semantic" ]]; then
  TOY_JOB="$(
    sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
      --model toy \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${TOY_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
  GRAHA_JOB="$(
    sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_seg_single_finetune.sh \
      --model graha \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${GRAHA_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
  GFFT_JOB="$(
    sbatch --parsable scripts/shell/semantic_seg/sbatch_semantic_gfft_single_finetune.sh \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${GFFT_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
else
  TOY_JOB="$(
    sbatch --parsable scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh \
      --model toy \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${TOY_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
  GRAHA_JOB="$(
    sbatch --parsable scripts/shell/instance_seg/sbatch_instance_seg_single_finetune.sh \
      --model graha \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${GRAHA_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
  GFFT_JOB="$(
    sbatch --parsable scripts/shell/instance_seg/sbatch_instance_gfft_single_finetune.sh \
      "${JOB_ARGS[@]}" \
      --base-output-dir "${GFFT_OUTPUT_DIR}" \
      --run-epoch-test-suite
  )"
fi

echo "Submitted Toy job: ${TOY_JOB}"
echo "Submitted Graha job: ${GRAHA_JOB}"
echo "Submitted GFFT job: ${GFFT_JOB}"

FINAL_COMPARISON_JOB=""
if [[ "${SUBMIT_FINAL_COMPARISON_PLOT}" -eq 1 ]]; then
  if [[ "${TASK}" == "semantic" ]]; then
    FINAL_COMPARISON_JOB="$(
      sbatch --parsable \
        --dependency=afterok:${TOY_JOB}:${GRAHA_JOB}:${GFFT_JOB} \
        scripts/shell/semantic_seg/sbatch_semantic_checkpoint_comparison_plot.sh \
        --toy-checkpoint-dir "${TOY_OUTPUT_DIR}/checkpoints/toy_model" \
        --graha-checkpoint-dir "${GRAHA_OUTPUT_DIR}/checkpoints/full_model" \
        --gfft-checkpoint-dir "${GFFT_OUTPUT_DIR}/checkpoints/gfft_model" \
        --output-dir "${FINAL_COMPARISON_OUTPUT_DIR}" \
        "${PLOT_ARGS[@]}"
    )"
  else
    FINAL_COMPARISON_JOB="$(
      sbatch --parsable \
        --dependency=afterok:${TOY_JOB}:${GRAHA_JOB}:${GFFT_JOB} \
        scripts/shell/instance_seg/sbatch_instance_checkpoint_comparison_plot.sh \
        --toy-checkpoint-dir "${TOY_OUTPUT_DIR}/checkpoints/toy_model" \
        --toy-plot-architecture "${TOY_ARCHITECTURE}" \
        --graha-checkpoint-dir "${GRAHA_OUTPUT_DIR}/checkpoints/full_model" \
        --gfft-checkpoint-dir "${GFFT_OUTPUT_DIR}/checkpoints/gfft_model" \
        --output-dir "${FINAL_COMPARISON_OUTPUT_DIR}" \
        "${PLOT_ARGS[@]}"
    )"
  fi
  echo "Submitted final checkpoint comparison plot job: ${FINAL_COMPARISON_JOB}"
  echo "Plot job dependency: afterok:${TOY_JOB}:${GRAHA_JOB}:${GFFT_JOB}"
fi

LOG_COLLECTION_SCRIPT="${RUN_ROOT}/collect_sbatch_logs.sh"
cat > "${LOG_COLLECTION_SCRIPT}" <<EOF
#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR}"
LOG_SOURCE_DIR="\${REPO_DIR}/scripts/logs"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR}"

mkdir -p "\${LOG_OUTPUT_DIR}"

copy_job_logs() {
  local label="\$1"
  local raw_job_id="\$2"
  local job_id="\${raw_job_id%%;*}"
  local copied=0

  if [[ -z "\${job_id}" ]]; then
    return
  fi

  for log_file in "\${LOG_SOURCE_DIR}"/*_"\${job_id}".out "\${LOG_SOURCE_DIR}"/*_"\${job_id}".err; do
    if [[ -e "\${log_file}" ]]; then
      cp -p "\${log_file}" "\${LOG_OUTPUT_DIR}/\${label}_\$(basename "\${log_file}")"
      copied=1
    fi
  done

  if [[ "\${copied}" -eq 0 ]]; then
    echo "No scripts/logs files found for \${label} job \${job_id}."
  fi
}

copy_job_logs "toy" "${TOY_JOB}"
copy_job_logs "graha" "${GRAHA_JOB}"
copy_job_logs "gfft" "${GFFT_JOB}"
EOF

LOG_DEPENDENCY="afterany:${TOY_JOB}:${GRAHA_JOB}:${GFFT_JOB}"
if [[ -n "${FINAL_COMPARISON_JOB}" ]]; then
  cat >> "${LOG_COLLECTION_SCRIPT}" <<EOF
copy_job_logs "final_comparison" "${FINAL_COMPARISON_JOB}"
EOF
  LOG_DEPENDENCY="${LOG_DEPENDENCY}:${FINAL_COMPARISON_JOB}"
fi
cat >> "${LOG_COLLECTION_SCRIPT}" <<'EOF'
copy_job_logs "log_collection" "${SLURM_JOB_ID:-}"
EOF
chmod +x "${LOG_COLLECTION_SCRIPT}"

LOG_COLLECTION_JOB="$(
  sbatch --parsable \
    --dependency="${LOG_DEPENDENCY}" \
    --output=scripts/logs/three_model_collect_logs_%j.out \
    --error=scripts/logs/three_model_collect_logs_%j.err \
    "${LOG_COLLECTION_SCRIPT}"
)"
echo "Submitted sbatch log collection job: ${LOG_COLLECTION_JOB}"
echo "Log collection dependency: ${LOG_DEPENDENCY}"
echo "Log collection output: ${LOG_OUTPUT_DIR}"

cat > "${RUN_ROOT}/submitted_jobs.txt" <<EOF
task=${TASK}
run_root=${RUN_ROOT}
toy_output_dir=${TOY_OUTPUT_DIR}
toy_job=${TOY_JOB}
graha_output_dir=${GRAHA_OUTPUT_DIR}
graha_job=${GRAHA_JOB}
gfft_output_dir=${GFFT_OUTPUT_DIR}
gfft_job=${GFFT_JOB}
final_comparison_output_dir=${FINAL_COMPARISON_OUTPUT_DIR}
final_comparison_job=${FINAL_COMPARISON_JOB}
log_output_dir=${LOG_OUTPUT_DIR}
log_collection_script=${LOG_COLLECTION_SCRIPT}
log_collection_job=${LOG_COLLECTION_JOB}
EOF

echo "Wrote job manifest: ${RUN_ROOT}/submitted_jobs.txt"
