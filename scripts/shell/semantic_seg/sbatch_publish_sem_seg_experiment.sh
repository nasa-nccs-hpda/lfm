#!/usr/bin/env bash
#SBATCH --job-name=publish_sseg_exp
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=scripts/logs/publish_sem_seg_experiment_%j.out
#SBATCH --error=scripts/logs/publish_sem_seg_experiment_%j.err

set -euo pipefail

SOURCE_DIR=""
DEST_DIR="/explore/nobackup/projects/lfm/model_experiments/graha_vs_toy/exp_1"
OVERWRITE="false"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/shell/semantic_seg/sbatch_publish_sem_seg_experiment.sh \
    --source-dir /path/to/best/experiment/output \
    [--dest-dir /explore/nobackup/projects/lfm/model_experiments/graha_vs_toy/exp_1] \
    [--overwrite]

Copies the selected semantic segmentation experiment output directory to DEST_DIR,
then runs:
  chmod -R 755 DEST_DIR
  chgrp -R j1123 DEST_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_DIR}" ]]; then
  echo "Missing required --source-dir." >&2
  usage >&2
  exit 2
fi

SOURCE_DIR="$(realpath "${SOURCE_DIR}")"
DEST_DIR="$(realpath -m "${DEST_DIR}")"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -e "${DEST_DIR}" && "${OVERWRITE}" != "true" ]]; then
  echo "Destination already exists: ${DEST_DIR}" >&2
  echo "Pass --overwrite to replace/update contents with rsync --delete." >&2
  exit 1
fi

START_TIME="$(date +%s)"
START_READABLE="$(date)"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../../../scripts" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
else
  REPO_DIR="${SUBMIT_DIR}"
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${REPO_DIR}"
echo "Source directory: ${SOURCE_DIR}"
echo "Destination directory: ${DEST_DIR}"
echo "Overwrite: ${OVERWRITE}"
echo

mkdir -p "$(dirname "${DEST_DIR}")"

RSYNC_ARGS=(-a --info=progress2)
if [[ "${OVERWRITE}" == "true" ]]; then
  RSYNC_ARGS+=(--delete)
fi

echo "Copying experiment output..."
rsync "${RSYNC_ARGS[@]}" "${SOURCE_DIR}/" "${DEST_DIR}/"

echo
echo "Setting permissions: chmod -R 755 ${DEST_DIR}"
chmod -R 755 "${DEST_DIR}"

echo "Setting group: chgrp -R j1123 ${DEST_DIR}"
chgrp -R j1123 "${DEST_DIR}"

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
echo "Published experiment: ${DEST_DIR}"
