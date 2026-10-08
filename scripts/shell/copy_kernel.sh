set -euo pipefail
echo "Copying kernel info and clearing old kernels..."
rm -rf ~/.local/lib/python*
rm -rf ~/.local/share/jupyter/kernels/lfm*
MAIN_KERNEL_PATH=~/.local/share/jupyter/kernels/lfm
mkdir -p $MAIN_KERNEL_PATH
cp /explore/nobackup/projects/lfm/containers/kernel-latest-again.json "$MAIN_KERNEL_PATH/kernel.json"
echo "Done! Kernel should appear as lfm_kernel."