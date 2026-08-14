#!/usr/bin/env bash
#SBATCH --job-name=iseg_chip_regress
#SBATCH --partition=grace
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/compare_common_instance_chips_%j.out
#SBATCH --error=scripts/logs/compare_common_instance_chips_%j.err

set -euo pipefail

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL="scripts/python/instance_seg/compare_common_instance_chips.py"

if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../python/instance_seg/compare_common_instance_chips.py" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Bind paths: ${APPTAINER_BIND_PATHS}"
echo "Python script: ${SCRIPT_REL}"
echo

apptainer exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" "$@"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
