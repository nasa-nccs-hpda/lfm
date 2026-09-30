set -euo pipefail
echo "Copying kernel info and clearing old kernels..."
rm -rf ~/.local/lib/python*

KERNELS_DIR=~/.local/share/jupyter/kernels
mkdir -p "$KERNELS_DIR"
find "$KERNELS_DIR" -mindepth 1 -maxdepth 1 ! -name 'lfm' -exec rm -rf {} +

KERNEL_PATH=~/.local/share/jupyter/kernels/lfm_ipyleaflet
mkdir -p "$KERNEL_PATH"
cp /panfs/ccds02/nobackup/projects/lfm/containers/kernel-v2.json "$KERNEL_PATH/kernel.json"
echo "Done! Kernel should appear in JupyterHub as \"lfm_kernel_ipyleaflet\"."