#!/usr/bin/env bash
# Container-side adapter for the host's two concrete command roles.
set -euo pipefail
transport="${LFM_HOST_TRANSPORT:?host transport required}"
request=$(mktemp -d "$transport/request.XXXXXXXX")
stdout_pid=
stderr_pid=
cleanup() {
    for pid in "$stdout_pid" "$stderr_pid"; do
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    rm -rf -- "$request"
}
trap cleanup EXIT
mkfifo "$request/stdout" "$request/stderr" "$request/status"
printf '%s\0' "$@" > "$request/arguments"
cat "$request/stdout" &
stdout_pid=$!
cat "$request/stderr" >&2 &
stderr_pid=$!
printf '%s\n' "$request" > "$transport/requests"
IFS= read -r status < "$request/status"
wait "$stdout_pid"
stdout_pid=
wait "$stderr_pid"
stderr_pid=
exit "$status"
