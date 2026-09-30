set -euo pipefail
echo "Copying kernel info and clearing old kernels..."
rm -rf ~/.local/lib/python*
rm -rf ~/.local/share/jupyter/kernels/*
KERNEL_PATH=~/.local/share/jupyter/kernels/lfm
mkdir -p $KERNEL_PATH
cp /panfs/ccds02/nobackup/projects/lfm/containers/kernel.json "$KERNEL_PATH/kernel.json"
echo "Done! Kernel should appear in JupyterHub as \"lfm_kernel\"."