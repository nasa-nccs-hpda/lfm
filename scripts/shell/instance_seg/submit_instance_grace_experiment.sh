#!/usr/bin/env bash
# Usage: bash scripts/shell/instance_seg/submit_instance_grace_experiment.sh --help
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONTAINER="${LFM_CONTAINER:-/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet-37938321}"
APPTAINER="${APPTAINER_BIN:-apptainer}"
SBATCH="${SBATCH_BIN:-sbatch}"
PARTITION=grace
ACCOUNT=
QOS=

for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        cat <<'USAGE'
Usage: bash scripts/shell/instance_seg/submit_instance_grace_experiment.sh [options]

From a login node, queue a CPU controller on a compute node using sbatch.
The controller runs container Python and submits Toy/GRAHA training plus comparison.

Common options (additional experiment options are forwarded unchanged):
  --max-epochs N       Total training epochs per model (default: 50; smoke: 1)
  --partition NAME     Controller and model-job partition (default: grace)
  --account NAME       Account for controller and model jobs
  --qos NAME           QoS for controller and model jobs
  --container PATH     Apptainer sandbox (default: lfm-container-ipyleaflet-37938321)
  --apptainer PATH     Apptainer command on COMPUTE nodes (default: apptainer)
  --sbatch PATH        Slurm submission command (default: sbatch)
  --run-root PATH      Fresh experiment directory
  --data-root PATH     Dataset directory
  --dino-checkpoint PATH
  --graha-pretrain-dir PATH

Controller resources: LFM_CONTROLLER_CPUS=1, LFM_CONTROLLER_MEMORY=8G,
LFM_CONTROLLER_TIME=00:30:00. LFM_CONTROLLER_LOG_DIR selects its log directory.
No Python or Apptainer is needed on the login node.
USAGE
        exit 0
    fi
done

# Resolve bootstrap options without a host Python interpreter. Argparse inside
# the container remains responsible for all experiment options and validation.
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    case "${arguments[index]}" in
        --container|--apptainer|--sbatch|--partition|--account|--qos)
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
                --partition) PARTITION="${arguments[index]}" ;;
                --account) ACCOUNT="${arguments[index]}" ;;
                --qos) QOS="${arguments[index]}" ;;
            esac
            ;;
        --container=*) CONTAINER="${arguments[index]#*=}" ;;
        --apptainer=*) APPTAINER="${arguments[index]#*=}" ;;
        --sbatch=*) SBATCH="${arguments[index]#*=}" ;;
        --partition=*) PARTITION="${arguments[index]#*=}" ;;
        --account=*) ACCOUNT="${arguments[index]#*=}" ;;
        --qos=*) QOS="${arguments[index]#*=}" ;;
    esac
done

if [[ -e "$CONTAINER" ]]; then
    CONTAINER=$(realpath -- "$CONTAINER")
fi
# Preserve explicit filesystem paths, but leave bare command names for the
# appropriate node's PATH. In particular, Apptainer is absent on login nodes.
if [[ "$APPTAINER" == */* && -e "$APPTAINER" ]]; then
    APPTAINER=$(realpath -- "$APPTAINER")
fi
if [[ "$SBATCH" == */* && -e "$SBATCH" ]]; then
    SBATCH=$(realpath -- "$SBATCH")
fi

controller_script="${REPO_DIR}/scripts/shell/instance_seg/sbatch_instance_grace_controller.sh"
controller_arguments=("$CONTAINER" "$APPTAINER" "$SBATCH" "$REPO_DIR"
                      "$@" --container "$CONTAINER" --apptainer "$APPTAINER" --sbatch "$SBATCH")
controller_logs="${LFM_CONTROLLER_LOG_DIR:-$REPO_DIR/scripts/outputs/instance_grace/controller_logs}"
mkdir -p -- "$controller_logs"
controller_logs=$(cd "$controller_logs" && pwd -P)
submission_options=(--parsable --job-name=iseg_submit --ntasks=1
                    "--partition=$PARTITION"
                    "--cpus-per-task=${LFM_CONTROLLER_CPUS:-1}"
                    "--mem=${LFM_CONTROLLER_MEMORY:-8G}"
                    "--time=${LFM_CONTROLLER_TIME:-00:30:00}"
                    "--chdir=$REPO_DIR"
                    "--output=$controller_logs/controller_%j.out"
                    "--error=$controller_logs/controller_%j.err")
[[ -z "$ACCOUNT" ]] || submission_options+=("--account=$ACCOUNT")
[[ -z "$QOS" ]] || submission_options+=("--qos=$QOS")
receipt=$("$SBATCH" "${submission_options[@]}" "$controller_script" "${controller_arguments[@]}")
printf 'Controller submission: %s\n' "$receipt"
controller_id="${receipt%%;*}"
printf 'Controller stdout: %s/controller_%s.out\n' "$controller_logs" "$controller_id"
printf 'Controller stderr: %s/controller_%s.err\n' "$controller_logs" "$controller_id"
printf 'The controller log will report the experiment directory and pipeline.log when its allocation starts.\n'
