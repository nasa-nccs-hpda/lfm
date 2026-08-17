echo "Copying kernel info..."
KERNEL_PATH=~/.local/share/jupyter/kernels/lfm
mkdir -p $KERNEL_PATH
cp -r /panfs/ccds02/nobackup/projects/lfm/kernel.json $KERNEL_PATH
echo "Done! Kernel should appear in JupyterHub as \"lfm_kernel\"."