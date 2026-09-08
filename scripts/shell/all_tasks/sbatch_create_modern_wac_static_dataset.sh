#!/usr/bin/env bash
#SBATCH --job-name=create_wac_static
#SBATCH --partition=grace
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=scripts/logs/create_wac_static_%j.out
#SBATCH --error=scripts/logs/create_wac_static_%j.err

# Required submission inputs:
#   REFERENCE_DIR=/path/to/reference/chips \
#   LABEL_SOURCE=/path/to/labels \
#   OUTPUT_ROOT=/path/to/new/dataset \
#   sbatch <this-script>

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

if [[ -z "${REFERENCE_DIR:-}" || -z "${LABEL_SOURCE:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "REFERENCE_DIR, LABEL_SOURCE, and OUTPUT_ROOT must be supplied." >&2
  exit 1
fi

MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"
ZOOM_LEVEL="${ZOOM_LEVEL:-5}"
RECURSIVE="${RECURSIVE:-0}"
INTERMEDIATE_RETENTION="${INTERMEDIATE_RETENTION:-on_failure}"
CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_ROOT}/batch_report.json}"

if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "REFERENCE_DIR does not exist: ${REFERENCE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${LABEL_SOURCE}" ]]; then
  echo "LABEL_SOURCE does not exist: ${LABEL_SOURCE}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to replace existing OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if (( MAX_WORKERS < 1 || MAX_WORKERS > ALLOCATED_CPUS )); then
  echo "MAX_WORKERS must be between 1 and allocated CPUs (${ALLOCATED_CPUS})." >&2
  exit 1
fi
if [[ "${RECURSIVE}" != "0" && "${RECURSIVE}" != "1" ]]; then
  echo "RECURSIVE must be 0 or 1." >&2
  exit 1
fi

export GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ARGS=(
  run
  --case-name full-wac-static-dataset
  --reference-dir "${REFERENCE_DIR}"
  --label-source "${LABEL_SOURCE}"
  --output-root "${OUTPUT_ROOT}"
  --report-path "${REPORT_PATH}"
  --max-workers "${MAX_WORKERS}"
  --all-samples
  --split-policy default
  --zoom-level "${ZOOM_LEVEL}"
  --include-static
  --intermediate-retention "${INTERMEDIATE_RETENTION}"
  --progress
  --progress-mode log
  --summary-only
)

if [[ "${RECURSIVE}" == "1" ]]; then
  ARGS+=(--recursive)
fi
if [[ -n "${WAC_DATA_DIR:-}" ]]; then
  ARGS+=(--wac-data-dir "${WAC_DATA_DIR}")
fi
if [[ -n "${WAC_INDEX:-}" ]]; then
  ARGS+=(--wac-index "${WAC_INDEX}")
fi
if [[ -n "${STATIC_DATA_DIR:-}" ]]; then
  ARGS+=(--static-data-dir "${STATIC_DATA_DIR}")
fi
if [[ -n "${STATIC_INDEX:-}" ]]; then
  ARGS+=(--static-index "${STATIC_INDEX}")
fi

echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Reference directory: ${REFERENCE_DIR}"
echo "Label source: ${LABEL_SOURCE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Report: ${REPORT_PATH}"
echo "Workers: ${MAX_WORKERS}"
echo "Zoom: ${ZOOM_LEVEL}"
echo "Split policy: default (100 test, then 90/10 train/validation)"
echo "Intermediate retention: ${INTERMEDIATE_RETENTION}"

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" "${ARGS[@]}" "$@"

END_TIME="$(date +%s)"
echo
echo "Full WAC + static batch completed."
echo "Dataset: ${OUTPUT_ROOT}"
echo "Report: ${REPORT_PATH}"
echo "Job finished at: $(date)"
echo "Elapsed seconds: $((END_TIME - START_TIME))"
