# HPC execution

The HPC system has no usable native Python runtime. Execute all project code on
HPC through Apptainer, including Python entry points, tests, linting, formatting,
and diagnostic snippets. Apply this rule to commands given to the user as well
as commands executed by an agent.

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

Host shell tools may handle navigation, Git, Slurm submission, and Apptainer
invocation. A launcher that invokes host `python`, `python3`, or host Conda needs
its execution path corrected before it is recommended. Containerized Python
controllers must also arrange access to any host scheduler/container tools they
invoke; wrapping a host-dependent submitter in `apptainer exec` alone does not
establish that access.

If this sandbox is unavailable in the agent's workspace, continue file inspection
and editing with shell/file tools and report container verification as unavailable.
Use the specified sandbox for runtime verification; distinguish local historical
test results from results obtained in this HPC container.
