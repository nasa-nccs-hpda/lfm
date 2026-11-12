#!/usr/bin/env bash
# Run the Python controller in Apptainer; execute its sbatch/Apptainer calls here
# on the host, where site binaries, authentication, and runtime libraries exist.
set -euo pipefail
container="${1:?container sandbox required}"
apptainer_executable="${2:?Apptainer executable required}"
scheduler_executable="${3:?sbatch executable required}"
repo_dir="${4:?repository required}"
controller="${5:?controller script required}"
shift 5

if [[ ! -e "$container" ]]; then
    printf 'Apptainer sandbox/container is unavailable: %s\n' "$container" >&2
    exit 1
fi

transport=$(mktemp -d "${TMPDIR:-/tmp}/lfm-grace-controller.XXXXXXXX")
controller_pid=
cleanup() {
    if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
        kill "$controller_pid" 2>/dev/null || true
        wait "$controller_pid" 2>/dev/null || true
    fi
    rm -rf -- "$transport"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkfifo "$transport/requests"
exec {request_fd}<>"$transport/requests"

# --cleanenv alone does not suppress these host injection channels.
for variable in "${!APPTAINER@}" "${!SINGULARITY@}"; do
    [[ -z "$variable" ]] || unset "$variable"
done

binds=(--bind "$repo_dir:$repo_dir" --bind "$transport:$transport")
if [[ -d /panfs/ccds02/nobackup ]]; then
    binds+=(--bind /panfs/ccds02/nobackup:/explore/nobackup
            --bind /panfs/ccds02/nobackup:/panfs/ccds02/nobackup)
fi
allocation=()
for variable in CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_JOB_NODELIST; do
    if [[ -v "$variable" ]]; then
        allocation+=(--env "$variable=${!variable}")
    fi
done
"$apptainer_executable" exec --cleanenv "${binds[@]}" --pwd "$repo_dir" \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env MPLBACKEND=Agg --env "LFM_HOST_TRANSPORT=$transport" \
    --env "LFM_HOST_COMMAND_BRIDGE=$repo_dir/scripts/shell/instance_seg/instance_grace_host_command.sh" \
    "${allocation[@]}" "$container" python -u "$controller" "$@" &
controller_pid=$!

while kill -0 "$controller_pid" 2>/dev/null; do
    if ! IFS= read -r -t 1 request <&"$request_fd"; then
        continue
    fi
    # Only this private transport directory may supply command arguments.
    if [[ "$request" != "$transport"/request.* || ! -d "$request" ]]; then
        printf 'Invalid controller request: %s\n' "$request" >&2
        exit 1
    fi
    command=()
    while IFS= read -r -d '' argument; do
        command+=("$argument")
    done < "$request/arguments"
    status=0
    case "${command[0]:-}" in
        sbatch|"$scheduler_executable") command[0]="$scheduler_executable" ;;
        apptainer|"$apptainer_executable") command[0]="$apptainer_executable" ;;
        *)
            { printf 'Unsupported host controller command: %s\n' "${command[0]:-}" >&2; } \
                > "$request/stdout" 2> "$request/stderr"
            status=126
            ;;
    esac
    if ((status == 0)); then
        "${command[@]}" > "$request/stdout" 2> "$request/stderr" || status=$?
    fi
    printf '%s\n' "$status" > "$request/status"
done

status=0
wait "$controller_pid" || status=$?
controller_pid=
exit "$status"
