#!/usr/bin/env bash
#SBATCH --job-name=validate_wac_chips
#SBATCH --partition=grace
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=scripts/logs/validate_wac_chips_%j.out
#SBATCH --error=scripts/logs/validate_wac_chips_%j.err

# Required submission inputs:
#   REFERENCE_DIR=/path/to/chips LABEL_SOURCE=/path/to/labels sbatch <this-script>
# Optional overrides:
#   SAMPLE_LIMIT=4 MAX_WORKERS=4 OUTPUT_ROOT=/path/to/output sbatch <this-script>

set -euo pipefail

START_TIME="$(date +%s)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/validate_wac_chip_creation.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

if [[ -z "${REFERENCE_DIR:-}" || -z "${LABEL_SOURCE:-}" ]]; then
  echo "REFERENCE_DIR and LABEL_SOURCE must be supplied to the Slurm job." >&2
  exit 1
fi

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/explore/nobackup/people/${USER}/lfm_c8_1_wac_${SLURM_JOB_ID:-manual}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-4}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "REFERENCE_DIR does not exist: ${REFERENCE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${LABEL_SOURCE}" ]]; then
  echo "LABEL_SOURCE does not exist: ${LABEL_SOURCE}" >&2
  exit 1
fi
if (( SAMPLE_LIMIT < 1 )); then
  echo "SAMPLE_LIMIT must be positive." >&2
  exit 1
fi
if (( MAX_WORKERS < 1 || MAX_WORKERS > ALLOCATED_CPUS )); then
  echo "MAX_WORKERS must be between 1 and allocated CPUs (${ALLOCATED_CPUS})." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to replace existing OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
  exit 1
fi

export GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ARGS=(
  --reference-dir "${REFERENCE_DIR}"
  --label-source "${LABEL_SOURCE}"
  --output-root "${OUTPUT_ROOT}"
  --sample-limit "${SAMPLE_LIMIT}"
  --max-workers "${MAX_WORKERS}"
  --progress
  --progress-mode log
)
if [[ -n "${WAC_DATA_DIR:-}" ]]; then
  ARGS+=(--wac-data-dir "${WAC_DATA_DIR}")
fi
if [[ -n "${WAC_INDEX:-}" ]]; then
  ARGS+=(--wac-index "${WAC_INDEX}")
fi
if [[ "${RECURSIVE:-0}" == "1" ]]; then
  ARGS+=(--recursive)
fi
if [[ "${NO_PLOTS:-0}" == "1" ]]; then
  ARGS+=(--no-plots)
fi

echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Reference directory: ${REFERENCE_DIR}"
echo "Label source: ${LABEL_SOURCE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Samples per API path: ${SAMPLE_LIMIT}"
echo "Workers: ${MAX_WORKERS}"

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" "${ARGS[@]}" "$@"

END_TIME="$(date +%s)"
echo
echo "C8.1 WAC chip validation completed successfully."
echo "Report: ${OUTPUT_ROOT}/c8_1_wac_validation.json"
echo "Plots: ${OUTPUT_ROOT}/inspection_plots"
echo "Job finished at: $(date)"
echo "Elapsed seconds: $((END_TIME - START_TIME))"
