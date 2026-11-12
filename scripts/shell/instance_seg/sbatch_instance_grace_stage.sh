#!/usr/bin/env bash
# Scheduler resources and log paths are supplied by the experiment submitter.
set -euo pipefail
CONTAINER="${1:?container sandbox required}"
REPO_DIR="${2:?repository required}"
APPTAINER="${3:?Apptainer executable required}"
MANIFEST="${4:?submission manifest required}"
STAGE="${5:?stage required}"
exec bash "${REPO_DIR}/scripts/shell/instance_seg/run_instance_grace_controller.sh" \
    "$CONTAINER" "$APPTAINER" sbatch "$REPO_DIR" \
    "${REPO_DIR}/scripts/python/instance_seg/run_instance_grace_stage.py" \
    --submission "${MANIFEST}" --stage "${STAGE}"
