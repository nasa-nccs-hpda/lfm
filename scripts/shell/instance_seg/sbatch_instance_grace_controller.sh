#!/usr/bin/env bash
# CPU bootstrap job. Resource/account/log options come from the login wrapper.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Run this controller inside a Slurm compute-node allocation.\n' >&2
    exit 1
fi
CONTAINER="${1:?container sandbox required}"
APPTAINER="${2:?compute-node Apptainer command required}"
SBATCH="${3:?sbatch command required}"
REPO_DIR="${4:?repository required}"
shift 4
exec bash "${REPO_DIR}/scripts/shell/instance_seg/run_instance_grace_controller.sh" \
    "$CONTAINER" "$APPTAINER" "$SBATCH" "$REPO_DIR" \
    "${REPO_DIR}/scripts/python/instance_seg/submit_instance_grace_experiment.py" "$@"
