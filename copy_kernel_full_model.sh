#!/usr/bin/env bash
set -euo pipefail

echo "Copying kernel info..."

REPO_ROOT="$(pwd)"
KERNEL_NAME="lfm-full-env"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
LAUNCHER="${KERNEL_DIR}/launch_lfm_full_env.sh"

module load miniforge
mamba activate /explore/nobackup/projects/lfm/lfm-full-env

python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "lfm-full-env"

mkdir -p "${KERNEL_DIR}"

cat > "${LAUNCHER}" <<EOF
#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
export LFM_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export PROJ_LIB="/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/proj"
export PROJ_DATA="\${PROJ_LIB}"
export GDAL_DATA="/panfs/ccds02/nobackup/projects/lfm/lfm-full-env/share/gdal"

exec /explore/nobackup/projects/lfm/lfm-full-env/bin/python -s -Xfrozen_modules=off -m ipykernel_launcher "\$@"
echo "LFM launcher ran at $(date)" >> "${HOME}/lfm_kernel_launcher.log"
echo "args: $*" >> "${HOME}/lfm_kernel_launcher.log"
EOF

chmod +x "${LAUNCHER}"

python - <<EOF
import json
from pathlib import Path

kernel_path = Path("${KERNEL_DIR}") / "kernel.json"
payload = json.loads(kernel_path.read_text())

payload["argv"] = [
    "${LAUNCHER}",
    "-f",
    "{connection_file}",
]
payload["display_name"] = "lfm-full-env"
payload["language"] = "python"

kernel_path.write_text(json.dumps(payload, indent=2) + "\\n")
print(kernel_path)
EOF

mamba deactivate
module purge

echo "Done! Restart JupyterHub or start a fresh notebook kernel."