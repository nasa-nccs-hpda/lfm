#!/usr/bin/env bash
#SBATCH --job-name=chip_parallel_profile
#SBATCH --partition=grace
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/chip_parallel_profile_%j.out
#SBATCH --error=scripts/logs/chip_parallel_profile_%j.err

# Required submission inputs:
#   REFERENCE_DIR=/path/to/chips LABEL_SOURCE=/path/to/labels sbatch <this-script>
# If assignments are separate commands, export them before calling sbatch.
# For more workers than the SBATCH default, also pass --cpus-per-task.
# Useful overrides:
#   SAMPLE_LIMIT=16 PARALLEL_WORKERS=4 INCLUDE_STATIC=1 RUN_ORDER=parallel-first

set -euo pipefail

START_TIME="$(date +%s)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/profile_chip_creation_parallelism.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or this script's directory." >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

if [[ -z "${REFERENCE_DIR:-}" || -z "${LABEL_SOURCE:-}" ]]; then
  echo "REFERENCE_DIR and LABEL_SOURCE were not exported to this Slurm job." >&2
  echo "Use inline assignments without &&, or export the variables before sbatch." >&2
  exit 1
fi
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-8}"
ZOOM_LEVEL="${ZOOM_LEVEL:-5}"
INCLUDE_STATIC="${INCLUDE_STATIC:-0}"
RECURSIVE="${RECURSIVE:-0}"
RUN_ORDER="${RUN_ORDER:-serial-first}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-8}"

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
PROFILE_ROOT="${PROFILE_ROOT:-/explore/nobackup/people/${USER}/lfm_chip_parallel_profile_${SLURM_JOB_ID:-manual}}"
HOST_PYTHON="${HOST_PYTHON:-python3}"

if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "REFERENCE_DIR does not exist: ${REFERENCE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${LABEL_SOURCE}" ]]; then
  echo "LABEL_SOURCE does not exist: ${LABEL_SOURCE}" >&2
  exit 1
fi
if ! command -v "${HOST_PYTHON}" >/dev/null 2>&1; then
  echo "Host Python executable does not exist: ${HOST_PYTHON}" >&2
  exit 1
fi
if (( PARALLEL_WORKERS < 2 )); then
  echo "PARALLEL_WORKERS must be at least 2." >&2
  exit 1
fi
if (( PARALLEL_WORKERS > ALLOCATED_CPUS )); then
  echo "PARALLEL_WORKERS (${PARALLEL_WORKERS}) exceeds allocated CPUs (${ALLOCATED_CPUS})." >&2
  exit 1
fi
if (( SAMPLE_LIMIT < PARALLEL_WORKERS )); then
  echo "SAMPLE_LIMIT must be at least PARALLEL_WORKERS." >&2
  exit 1
fi

mkdir -p "${PROFILE_ROOT}"
if [[ -e "${PROFILE_ROOT}/comparison.json" ]]; then
  echo "Refusing to replace ${PROFILE_ROOT}/comparison.json." >&2
  exit 1
fi

export GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

APPTAINER_ARGS=(
  exec
  --bind "${APPTAINER_BIND_PATHS}"
  --bind "${REPO_DIR}"
  --pwd "${REPO_DIR}"
  "${CONTAINER_PATH}"
)

COMMON_ARGS=(
  --reference-dir "${REFERENCE_DIR}"
  --label-source "${LABEL_SOURCE}"
  --sample-limit "${SAMPLE_LIMIT}"
  --zoom-level "${ZOOM_LEVEL}"
)

if [[ "${INCLUDE_STATIC}" == "1" ]]; then
  COMMON_ARGS+=(--include-static)
elif [[ "${INCLUDE_STATIC}" != "0" ]]; then
  echo "INCLUDE_STATIC must be 0 or 1." >&2
  exit 1
fi

if [[ "${RECURSIVE}" == "1" ]]; then
  COMMON_ARGS+=(--recursive)
elif [[ "${RECURSIVE}" != "0" ]]; then
  echo "RECURSIVE must be 0 or 1." >&2
  exit 1
fi

if [[ -n "${WAC_DATA_DIR:-}" ]]; then
  COMMON_ARGS+=(--wac-data-dir "${WAC_DATA_DIR}")
fi
if [[ -n "${WAC_INDEX:-}" ]]; then
  COMMON_ARGS+=(--wac-index "${WAC_INDEX}")
fi
if [[ -n "${STATIC_DATA_DIR:-}" ]]; then
  COMMON_ARGS+=(--static-data-dir "${STATIC_DATA_DIR}")
fi
if [[ -n "${STATIC_INDEX:-}" ]]; then
  COMMON_ARGS+=(--static-index "${STATIC_INDEX}")
fi

run_case() {
  local case_name="$1"
  local worker_count="$2"
  local dataset_root="${PROFILE_ROOT}/${case_name}_dataset"
  local report_path="${PROFILE_ROOT}/${case_name}_profile.json"
  local measurement_path="${PROFILE_ROOT}/${case_name}_measurement.json"

  if [[ -e "${dataset_root}" || -e "${report_path}" || -e "${measurement_path}" ]]; then
    echo "Refusing to replace an existing ${case_name} profile under ${PROFILE_ROOT}." >&2
    exit 1
  fi

  echo
  echo "Starting ${case_name} case with ${worker_count} worker(s)..."
  "${HOST_PYTHON}" "${SCRIPT_REL}" measure \
    --output-path "${measurement_path}" \
    -- \
      "${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" \
      python -u "${SCRIPT_REL}" run \
        --case-name "${case_name}" \
        --output-root "${dataset_root}" \
        --report-path "${report_path}" \
        --max-workers "${worker_count}" \
        "${COMMON_ARGS[@]}"
}

echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Reference directory: ${REFERENCE_DIR}"
echo "Label source: ${LABEL_SOURCE}"
echo "Profile root: ${PROFILE_ROOT}"
echo "Sample limit: ${SAMPLE_LIMIT}"
echo "Parallel workers: ${PARALLEL_WORKERS}"
echo "Include static: ${INCLUDE_STATIC}"
echo "Run order: ${RUN_ORDER}"

if [[ "${RUN_ORDER}" == "serial-first" ]]; then
  run_case serial 1
  run_case parallel "${PARALLEL_WORKERS}"
elif [[ "${RUN_ORDER}" == "parallel-first" ]]; then
  run_case parallel "${PARALLEL_WORKERS}"
  run_case serial 1
else
  echo "RUN_ORDER must be serial-first or parallel-first." >&2
  exit 1
fi

"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" \
  python -u "${SCRIPT_REL}" compare \
    --serial-report "${PROFILE_ROOT}/serial_profile.json" \
    --parallel-report "${PROFILE_ROOT}/parallel_profile.json" \
    --serial-measurement "${PROFILE_ROOT}/serial_measurement.json" \
    --parallel-measurement "${PROFILE_ROOT}/parallel_measurement.json" \
    --output-path "${PROFILE_ROOT}/comparison.json"

END_TIME="$(date +%s)"
echo
echo "Chip parallelism profile completed successfully."
echo "Comparison report: ${PROFILE_ROOT}/comparison.json"
echo "Job finished at: $(date)"
echo "Total wrapper elapsed seconds: $((END_TIME - START_TIME))"
