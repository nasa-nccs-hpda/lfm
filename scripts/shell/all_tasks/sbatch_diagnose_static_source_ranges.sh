#!/usr/bin/env bash
#SBATCH --job-name=static_src_ranges
#SBATCH --partition=grace
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/static_src_ranges_%j.out
#SBATCH --error=scripts/logs/static_src_ranges_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/all_tasks/diagnose_static_source_ranges.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  echo "Submit this script from the repository root or its own directory." >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
OUTPUT_DIR="${OUTPUT_DIR:-/explore/nobackup/people/${USER}/lfm_static_source_ranges_${SLURM_JOB_ID:-manual}}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Static source-range diagnostic completed successfully."
echo "JSON report: ${OUTPUT_DIR}/static_source_range_report.json"
echo "TSV report: ${OUTPUT_DIR}/static_source_ranges.tsv"
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
