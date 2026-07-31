#!/usr/bin/env bash
set -euo pipefail

echo "Copying kernel info..."
module load miniforge
mamba activate /explore/nobackup/projects/lfm/lfm-full-env
export PYTHONNOUSERSITE=1
python -m pip install -e .
python -m ipykernel install --user --name lfm-full-env --display-name "lfm-full-env"
mamba deactivate
module purge
echo "Done! Kernel should appear in JupyterHub as \"lfm-full-env\"."