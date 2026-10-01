set -euo pipefail
echo "Copying kernel info and clearing old kernels..."
rm -rf ~/.local/lib/python*
rm -rf ~/.local/share/jupyter/kernels/lfm*
MAIN_KERNEL_PATH=~/.local/share/jupyter/kernels/lfm
IPY_KERNEL_PATH=~/.local/share/jupyter/kernels/lfm_ipyleaflet
mkdir -p $MAIN_KERNEL_PATH
mkdir -p $IPY_KERNEL_PATH
cp /panfs/ccds02/nobackup/projects/lfm/containers/kernel.json "$MAIN_KERNEL_PATH/kernel.json"
cp /panfs/ccds02/nobackup/projects/lfm/containers/kernel-v2.json "$IPY_KERNEL_PATH/kernel.json"
echo "Done! Kernels should appear in JupyterHub as \"lfm_kernel\" and \"lfm_kernel_ipyleaflet\"."
