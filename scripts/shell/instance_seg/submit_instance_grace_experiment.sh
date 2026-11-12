#!/usr/bin/env bash
# Usage: bash scripts/shell/instance_seg/submit_instance_grace_experiment.sh --help
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${PYTHON_BIN:-python3}" "${REPO_DIR}/scripts/python/instance_seg/submit_instance_grace_experiment.py" "$@"
