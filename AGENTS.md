# HPC execution

The HPC system has no usable native Python runtime. Execute all project code on
HPC through Apptainer, including Python entry points, tests, linting, formatting,
and diagnostic snippets. Apply this rule to commands given to the user as well
as commands executed by an agent.

Login nodes provide Slurm submission tools but **do not provide Apptainer**.
Use host Bash and `sbatch` there to submit work to compute nodes. Start
Apptainer and container Python only inside a compute-node Slurm allocation.
Preparation, controller code, imports, and diagnostics need this allocation too;
they cannot bootstrap a container on a login node. Resolve compute-node
executable names such as `apptainer` on the compute node, rather than looking
them up on the login node.

Use this user-specified **Apptainer sandbox directory**:

```text
/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet-37938321
```

Use `apptainer exec --cleanenv` with that directory directly. Historical
container defaults and local Conda environments do not select the HPC runtime.

From the repository root in an appropriate Slurm allocation, a CPU test command
has this form:

```bash
LFM_CONTAINER=/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet-37938321
apptainer exec --cleanenv \
  --bind /panfs/ccds02/nobackup:/explore/nobackup \
  --bind "$PWD:/lfm-repo" --pwd /lfm-repo \
  --env PYTHONNOUSERSITE=1 --env MPLBACKEND=Agg \
  "$LFM_CONTAINER" \
  python -m pytest -q scripts/python/tests/test_shared_plotting.py
```

For allocated GPU work, add `--nv` and explicitly forward
`--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"` when the allocation sets
that variable. Preserve the assigned GPU mask. Bind other required inputs and
writable outputs explicitly, retaining their expected container paths.

Host shell tools may handle navigation, Git, and Slurm submission on login
nodes. Compute-node host Bash may also invoke Apptainer. A login-node launcher
must submit a compute-node controller job before executing container code.
`SLURM_JOB_ID` alone does not prove a shell is running on a compute node (for
example, an `salloc` shell can still be on the login node). The experiment's
submission wrapper always queues its controller through `sbatch`. Containerized
Python controllers must arrange access to the compute-node host's scheduler
and Apptainer commands. Host Python and host Conda are not execution paths.

If this sandbox is unavailable in the agent's workspace, continue file inspection
and editing with shell/file tools and report container verification as unavailable.
Use the specified sandbox for runtime verification; distinguish local historical
test results from results obtained in this HPC container.
