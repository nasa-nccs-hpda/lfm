#!/usr/bin/env bash
#SBATCH --job-name=lfm-build-def
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=lfm-build-def-%j.out
#SBATCH --error=lfm-build-def-%j.err
#SBATCH --partition=grace

# From the repository root on a Slurm submission host:
#   sbatch scripts/shell/build_ipyleaflet_container_def.sh
# Optional first argument: a NEW absolute destination sandbox path.
# Do not run with bash on a login node. GPU login nodes without sbatch cannot
# submit this job: use a Slurm submission host, not a direct bash invocation.
# The grace partition selects ARM64; an AMD/x86_64 allocation is rejected too.
# Requires Apptainer with working unprivileged/fakeroot build support.
# No Docker, Podman, BuildKit, or host Python installation is required.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fail "Submit with 'sbatch scripts/shell/build_ipyleaflet_container_def.sh' from the repo root. If this GPU login node has no sbatch command, use a Slurm submission host. Do not build here with bash."
fi
[[ "$(uname -m)" == aarch64 ]] || fail "This build requires ARM64/aarch64; this node is $(uname -m). Use sbatch --partition=grace, not bash on an AMD/x86_64 node."
[[ $# -le 1 ]] || fail "Usage: sbatch scripts/shell/build_ipyleaflet_container_def.sh [/absolute/new/sandbox/path]"

REPO_DIR="${LFM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$REPO_DIR" ]] || fail "Set LFM_REPO_DIR to the repository root."
REPO_DIR="$(realpath "$REPO_DIR")"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
for tool in "$APPTAINER_BIN" rsync; do
    command -v "$tool" >/dev/null || fail "Required command not found: $tool. Load the site's Apptainer tools in the job environment."
done
version_text="$("$APPTAINER_BIN" --version)"

DEST="${1:-${LFM_CONTAINER_DIR:-/explore/nobackup/projects/lfm/containers}/lfm-container-ipyleaflet-${SLURM_JOB_ID}}"
[[ "$DEST" == /* ]] || fail "Destination must be an absolute path."
[[ ! -e "$DEST" && ! -L "$DEST" ]] || fail "Destination already exists; choose a new path: $DEST"

# Snapshot the definition and its %files inputs, including uncommitted edits.
# Keep this list aligned with the definition if its %files section changes.
inputs=(lfm_container-latest.def requirements_container.txt scripts/shell/install_container_dependencies.sh)
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
echo "Apptainer: $version_text"
echo "Build workspace: $BUILD_DIR; destination: $DEST"
trap 'status=$?; if (( status != 0 )); then echo "Build/publish failed ($status). Workspace retained: $BUILD_DIR" >&2; fi' EXIT
cd "$BUILD_DIR/context"

# Build from the repository definition; relative %files paths use this snapshot.
"$APPTAINER_BIN" build --fakeroot --sandbox "$BUILD_DIR/sandbox" lfm_container-latest.def

# Publish only a completed sandbox; never delete an existing container.
mkdir -p "$(dirname "$DEST")"
STAGING_DIR="$(mktemp -d "$(dirname "$DEST")/.lfm-build-${SLURM_JOB_ID}.XXXXXX")"
rsync -a --no-owner --no-group "$BUILD_DIR/sandbox/" "$STAGING_DIR/"
mv -T --no-clobber "$STAGING_DIR" "$DEST"
[[ ! -d "$STAGING_DIR" ]] || fail "Destination appeared during the build; completed copy retained at $STAGING_DIR"
echo "Container successfully published: $DEST"
echo "Workspace retained for inspection: $BUILD_DIR"
