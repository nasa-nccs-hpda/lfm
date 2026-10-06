#!/usr/bin/env bash
#SBATCH --job-name=find_polar_nac
#SBATCH --partition=grace
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/find_polar_nac_%j.out
#SBATCH --error=scripts/logs/find_polar_nac_%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_REL=scripts/python/all_tasks/find_polar_nac.py
if [[ ! -f "${REPO_DIR}/${SCRIPT_REL}" ]]; then
  echo "Submit from the repository root." >&2
  exit 2
fi
CONTAINER_PATH="${CONTAINER_PATH:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/notebooks/outputs/diagnostics/polar_nac_${SLURM_JOB_ID:-local}.json}"
apptainer exec \
  --bind "/panfs/ccds02:/explore,${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  "${CONTAINER_PATH}" \
  /usr/bin/python3 -u "${SCRIPT_REL}" --report "${REPORT_PATH}" "$@"
