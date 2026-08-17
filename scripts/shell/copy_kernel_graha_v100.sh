#!/usr/bin/env bash
set -euo pipefail

echo "Copying kernel info and clearing user site packages..."
rm -rf ~/.local/lib/python*
module load miniforge
mamba activate /explore/nobackup/projects/lfm/lfm-full-env
python -m ipykernel install --user --name lfm-full-env --display-name "lfm-full-env"
mamba deactivate
module purge
echo "Done! Kernel should appear in JupyterHub as \"lfm-full-env\"."