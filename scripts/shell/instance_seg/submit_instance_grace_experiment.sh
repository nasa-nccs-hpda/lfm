#!/usr/bin/env bash
# Usage: bash scripts/shell/instance_seg/submit_instance_grace_experiment.sh --help
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONTAINER="${LFM_CONTAINER:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet-37938321}"
APPTAINER="${APPTAINER_BIN:-apptainer}"
SBATCH="${SBATCH_BIN:-sbatch}"

# Resolve bootstrap options without a host Python interpreter. Argparse inside
# the container remains responsible for all experiment options and validation.
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    case "${arguments[index]}" in
        --container|--apptainer|--sbatch)
            option="${arguments[index]}"
            if ((index + 1 >= ${#arguments[@]})); then
                printf 'Missing value for %s\n' "$option" >&2
                exit 2
            fi
            ((index+=1))
            case "$option" in
                --container) CONTAINER="${arguments[index]}" ;;
                --apptainer) APPTAINER="${arguments[index]}" ;;
                --sbatch) SBATCH="${arguments[index]}" ;;
            esac
            ;;
        --container=*) CONTAINER="${arguments[index]#*=}" ;;
        --apptainer=*) APPTAINER="${arguments[index]#*=}" ;;
        --sbatch=*) SBATCH="${arguments[index]#*=}" ;;
    esac
done

if [[ -e "$CONTAINER" ]]; then
    CONTAINER=$(realpath -- "$CONTAINER")
fi
# The allocated job may run with a different cwd/PATH. Retain host executable
# paths when they are available; normal invocation reports missing commands.
APPTAINER=$(command -v "$APPTAINER" || printf '%s' "$APPTAINER")
SBATCH=$(command -v "$SBATCH" || printf '%s' "$SBATCH")
if [[ "$APPTAINER" == */* && -e "$APPTAINER" ]]; then
    APPTAINER=$(realpath -- "$APPTAINER")
fi
if [[ "$SBATCH" == */* && -e "$SBATCH" ]]; then
    SBATCH=$(realpath -- "$SBATCH")
fi

exec bash "${REPO_DIR}/scripts/shell/instance_seg/run_instance_grace_controller.sh" \
    "$CONTAINER" "$APPTAINER" "$SBATCH" "$REPO_DIR" \
    "${REPO_DIR}/scripts/python/instance_seg/submit_instance_grace_experiment.py" \
    "$@" --container "$CONTAINER" --apptainer "$APPTAINER" --sbatch "$SBATCH"
