#!/usr/bin/env bash
set -euo pipefail

echo "Copying kernel info..."

REPO_ROOT="$(pwd)"
KERNEL_NAME="lfm-full-env"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"

module load miniforge
mamba activate /explore/nobackup/projects/lfm/lfm-full-env

python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "lfm-full-env"

mkdir -p "${KERNEL_DIR}"

python - <<EOF
import json
from pathlib import Path

kernel_path = Path("${KERNEL_DIR}") / "kernel.json"
payload = json.loads(kernel_path.read_text())

env = payload.setdefault("env", {})
env["PYTHONNOUSERSITE"] = "1"
env["LFM_REPO_ROOT"] = "${REPO_ROOT}"
env["PYTHONPATH"] = "${REPO_ROOT}"
env["PROJ_LIB"] = "/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/proj"
env["PROJ_DATA"] = "/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/proj"
env["GDAL_DATA"] = "/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/gdal"

kernel_path.write_text(json.dumps(payload, indent=2) + "\\n")
print(kernel_path)
EOF

mamba deactivate
module purge

echo "Done! Kernel should appear in JupyterHub as \"lfm-full-env\"."