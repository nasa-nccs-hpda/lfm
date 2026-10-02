#!/usr/bin/env bash
#SBATCH --job-name=lfm-build
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=lfm-build-%j.out
#SBATCH --error=lfm-build-%j.err
#SBATCH --partition=grace

# From the repository root on a Slurm submission host:
#   sbatch scripts/shell/build_ipyleaflet_container.sh
# Optional first argument: a NEW absolute destination sandbox path.
# Do not run with bash on a login node. GPU login nodes without sbatch cannot
# submit this job: use a Slurm submission host, not a direct bash invocation.
# The grace partition selects ARM64; an AMD/x86_64 allocation is rejected too.
# Requires Apptainer >= 1.5, buildctl, and a reachable ARM64 BuildKit worker.
# BuildKit must run on the allocated compute node (or a site-approved service),
# not on a login node. Docker/Podman are not used by this script.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fail "Submit with 'sbatch scripts/shell/build_ipyleaflet_container.sh' from the repo root. If this GPU login node has no sbatch command, use a Slurm submission host. Do not build here with bash."
fi
[[ "$(uname -m)" == aarch64 ]] || fail "This build requires ARM64/aarch64; this node is $(uname -m). Use sbatch --partition=grace, not bash on an AMD/x86_64 node."
[[ $# -le 1 ]] || fail "Usage: sbatch scripts/shell/build_ipyleaflet_container.sh [/absolute/new/sandbox/path]"

REPO_DIR="${LFM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$REPO_DIR" ]] || fail "Set LFM_REPO_DIR to the repository root."
REPO_DIR="$(realpath "$REPO_DIR")"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
for tool in "$APPTAINER_BIN" buildctl rsync python3; do
    command -v "$tool" >/dev/null || fail "Required command not found: $tool. Load the site's Apptainer/BuildKit tools in the job environment."
done
version_text="$("$APPTAINER_BIN" --version)"
python3 - "$version_text" <<'PY'
import re
import sys
match = re.search(r'(\d+)\.(\d+)\.(\d+)', sys.argv[1])
if not match or tuple(map(int, match.groups())) < (1, 5, 0):
    sys.exit('Dockerfile builds require Apptainer >= 1.5; found: ' + sys.argv[1])
PY

# Use the same BuildKit endpoint for the preflight and Apptainer invocation.
export BUILDKIT_HOST="${APPTAINER_BUILDKIT_HOST:-${BUILDKIT_HOST:-unix:///run/buildkit/buildkitd.sock}}"
export APPTAINER_BUILDKIT_HOST="$BUILDKIT_HOST"
if ! workers="$(buildctl --addr "$BUILDKIT_HOST" debug workers 2>&1)"; then
    fail "Cannot reach BuildKit at $BUILDKIT_HOST. Direct Dockerfile builds require a running BuildKit daemon. Configure a compute-node/site-approved service and BUILDKIT_HOST. Details: $workers"
fi
[[ "$workers" == *linux/arm64* ]] || fail "BuildKit has no linux/arm64 worker: $workers"

DEST="${1:-${LFM_CONTAINER_DIR:-/explore/nobackup/projects/lfm/containers}/lfm-container-ipyleaflet-${SLURM_JOB_ID}}"
[[ "$DEST" == /* ]] || fail "Destination must be an absolute path."
[[ ! -e "$DEST" && ! -L "$DEST" ]] || fail "Destination already exists; choose a new path: $DEST"

# Snapshot the current Dockerfile's build context, including uncommitted edits.
# Keep this list aligned with Dockerfile COPY instructions if they change.
inputs=(Dockerfile .dockerignore requirements_container.txt scripts/shell/install_container_dependencies.sh)
for input in "${inputs[@]}"; do
    [[ -f "$REPO_DIR/$input" ]] || fail "Missing build input: $REPO_DIR/$input. Submit from the repo root or set LFM_REPO_DIR."
done
scratch_root="${LFM_BUILD_SCRATCH:-/lscratch/${USER}}"
mkdir -p "$scratch_root"
BUILD_DIR="$(mktemp -d "$scratch_root/lfm-build-${SLURM_JOB_ID}.XXXXXX")"
export APPTAINER_TMPDIR="$BUILD_DIR/tmp"
export APPTAINER_CACHEDIR="$BUILD_DIR/cache"
mkdir -p "$BUILD_DIR/context/scripts/shell" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
for input in "${inputs[@]}"; do
    cp "$REPO_DIR/$input" "$BUILD_DIR/context/$input"
done

echo "Host: $(hostname); architecture: $(uname -m); Slurm job: $SLURM_JOB_ID"
echo "Apptainer: $version_text; BuildKit: $BUILDKIT_HOST"
echo "Build workspace: $BUILD_DIR; destination: $DEST"
echo "BuildKit worker storage is separate from APPTAINER_TMPDIR; ensure both have space."
trap 'status=$?; if (( status != 0 )); then echo "Build/publish failed ($status). Workspace retained: $BUILD_DIR" >&2; fi' EXIT
cd "$BUILD_DIR/context"

# Build directly from the Dockerfile. No .def file is read or generated.
"$APPTAINER_BIN" build --sandbox --arch arm64 "$BUILD_DIR/sandbox" dockerfile:.

# Publish only a completed sandbox; never delete an existing container.
mkdir -p "$(dirname "$DEST")"
STAGING_DIR="$(mktemp -d "$(dirname "$DEST")/.lfm-build-${SLURM_JOB_ID}.XXXXXX")"
rsync -a --no-owner --no-group "$BUILD_DIR/sandbox/" "$STAGING_DIR/"
mv -T --no-clobber "$STAGING_DIR" "$DEST"
[[ ! -d "$STAGING_DIR" ]] || fail "Destination appeared during the build; completed copy retained at $STAGING_DIR"
echo "Container successfully published: $DEST"
echo "Workspace retained for inspection: $BUILD_DIR"
