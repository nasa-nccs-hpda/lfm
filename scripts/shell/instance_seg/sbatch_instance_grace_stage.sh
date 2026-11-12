#!/usr/bin/env bash
# Scheduler resources and log paths are supplied by the experiment submitter.
set -euo pipefail
HOST_PYTHON="${1:?host Python required}"
REPO_DIR="${2:?repository required}"
MANIFEST="${3:?submission manifest required}"
STAGE="${4:?stage required}"
exec "${HOST_PYTHON}" "${REPO_DIR}/scripts/python/instance_seg/run_instance_grace_stage.py" \
    --submission "${MANIFEST}" --stage "${STAGE}"
