#!/usr/bin/env bash
# Shell-only doubles exercise bootstrap/host orchestration without Python or Slurm.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/sandbox" "$scratch/tmp"
export LFM_FAKE_ROOT="$scratch" TMPDIR="$scratch/tmp"

cat > "$scratch/bin/python3" <<'STUB'
#!/usr/bin/env bash
touch "$LFM_FAKE_ROOT/host-python-used"
printf 'SyntaxError: future feature annotations is not defined\n' >&2
exit 86
STUB
cp "$scratch/bin/python3" "$scratch/bin/python"

cat > "$scratch/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\0' "$@" > "$LFM_FAKE_ROOT/scheduler-arguments"
printf '1001;grace-cluster\n'
if [[ "${LFM_FAKE_SCHEDULER_STATUS:-0}" != 0 ]]; then
    printf 'scheduler refused\n' >&2
fi
exit "${LFM_FAKE_SCHEDULER_STATUS:-0}"
STUB

cat > "$scratch/bin/apptainer" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == gpu-work ]]; then
    # GPU work must reach the host, rather than a container's nested runtime.
    [[ ! -v LFM_HOST_COMMAND_BRIDGE ]]
    printf 'allocated output\n'
    printf 'allocated stderr\n' >&2
    if [[ "${LFM_FAKE_HOLD:-0}" == 1 ]]; then
        for ((attempt=0; attempt<250; attempt++)); do
            [[ ! -e "$LFM_FAKE_ROOT/release" ]] || break
            sleep 0.02
        done
        [[ -e "$LFM_FAKE_ROOT/release" ]]
    fi
    exit "${LFM_FAKE_GPU_STATUS:-0}"
fi
[[ "$1" == exec && "$2" == --cleanenv ]]
[[ ! -v APPTAINERENV_PYTHONPATH && ! -v SINGULARITYENV_GDAL_DATA ]]
printf '%s\0' "$@" > "$LFM_FAKE_ROOT/bootstrap-arguments"
while [[ "$1" != python ]]; do
    if [[ "$1" == --env ]]; then
        export "$2"
        shift
    fi
    shift
done
[[ "$2" == -u ]]
[[ "$3" == */submit_instance_grace_experiment.py || "$3" == */run_instance_grace_stage.py ]]
printf 'container Python selected\n'
if [[ "$3" == */submit_instance_grace_experiment.py ]]; then
    bash "$LFM_HOST_COMMAND_BRIDGE" "$LFM_FAKE_ROOT/bin/sbatch" \
        'value with spaces' "quote's" '*literal*' $'two\nlines'
fi
bash "$LFM_HOST_COMMAND_BRIDGE" "$LFM_FAKE_ROOT/bin/apptainer" gpu-work
STUB

chmod +x "$scratch/bin/"*
export PATH="$scratch/bin:$PATH" PYTHON_BIN="$scratch/bin/python3"
export APPTAINERENV_PYTHONPATH=/host/conda SINGULARITYENV_GDAL_DATA=/host/gdal
submit="$repo_dir/scripts/shell/instance_seg/submit_instance_grace_experiment.sh"
arguments=(--container "$scratch/sandbox" --apptainer "$scratch/bin/apptainer"
           --sbatch "$scratch/bin/sbatch" --max-epochs 1)
bash "$submit" "${arguments[@]}" > "$scratch/output" 2> "$scratch/error"
[[ ! -e "$scratch/host-python-used" ]]
[[ $(<"$scratch/output") == *'container Python selected'* ]]
[[ $(<"$scratch/output") == *'allocated output'* ]]
[[ $(<"$scratch/error") == *'allocated stderr'* ]]
expected=('value with spaces' "quote's" '*literal*' $'two\nlines')
index=0
while IFS= read -r -d '' argument; do
    [[ "$argument" == "${expected[index]}" ]]
    ((index+=1))
done < "$scratch/scheduler-arguments"
[[ "$index" == 4 ]]

for failure in scheduler gpu; do
    status=0
    if [[ "$failure" == scheduler ]]; then
        LFM_FAKE_SCHEDULER_STATUS=17 bash "$submit" "${arguments[@]}" \
            > "$scratch/output" 2> "$scratch/error" || status=$?
        [[ "$status" == 17 && $(<"$scratch/error") == *'scheduler refused'* ]]
    else
        LFM_FAKE_GPU_STATUS=137 bash "$submit" "${arguments[@]}" \
            > "$scratch/output" 2> "$scratch/error" || status=$?
        [[ "$status" == 137 ]]
    fi
done

# Test the allocated wrapper with the new container/executable arguments too.
bash "$repo_dir/scripts/shell/instance_seg/sbatch_instance_grace_stage.sh" \
    "$scratch/sandbox" "$repo_dir" "$scratch/bin/apptainer" /unused/submission.json toy \
    > "$scratch/output" 2> "$scratch/error"
[[ ! -e "$scratch/host-python-used" ]]

: > "$scratch/live-output"
LFM_FAKE_HOLD=1 bash "$submit" "${arguments[@]}" \
    > "$scratch/live-output" 2> "$scratch/live-error" &
launcher_pid=$!
for ((attempt=0; attempt<200; attempt++)); do
    [[ $(<"$scratch/live-output") != *'allocated output'* ]] || break
    sleep 0.02
done
[[ $(<"$scratch/live-output") == *'allocated output'* ]]
kill -0 "$launcher_pid"
touch "$scratch/release"
wait "$launcher_pid"
if compgen -G "$scratch/tmp/lfm-grace-controller.*" > /dev/null; then
    printf 'Controller transport was not cleaned up\n' >&2
    exit 1
fi
printf 'PASS: container bootstrap, host commands, argument boundaries, live output, exit statuses, cleanup\n'
