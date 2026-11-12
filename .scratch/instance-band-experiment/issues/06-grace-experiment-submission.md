# 06: Submit the three-job Grace container experiment

**What to build:** One command submits independent Toy and GRAHA training jobs and a dependent checkpoint-comparison job, all running inside the selected container on the Slurm Grace partition. The same command supports an explicitly requested one-epoch end-to-end experiment.

**Blocked by:** 05 — Compare both best checkpoints on the test set.

**Status:** done

- [x] Create the resolved experiment configuration, shared split manifest, output layout, and Slurm log directories before submission. Submit two single-GPU training jobs followed by a single-GPU comparison job whose `afterok` dependency names both training IDs.
- [x] Capture parsable scheduler IDs, strip any cluster suffix before constructing dependencies, and persist all three IDs and the dependency. Preserve already-submitted job IDs and errors if a later submission fails; never silently lose the partial submission state.
- [x] Launch every GPU stage with Apptainer clean environment and NVIDIA support, preserve Slurm CUDA visibility explicitly, mount code and required artifacts, configure headless plotting, and avoid host Conda/GDAL/PROJ leakage. Keep inputs read-only and output/cache destinations writable.
- [x] Expose container, dataset, pretrained weights, cache/output/configuration paths, account/QoS, and resources. Default training requests to one GPU, 128 GB host memory, 10 CPUs, and 24 hours; comparison resources and inference batch are independently configurable.
- [x] Forward positive `--max-epochs` unchanged to both jobs/trainers and record the resolved value. Default to 50. An explicitly requested value of 1 preserves batch size 32, all 70 bands, full splits, and the dependent test-comparison stage.
- [x] Limit startup checks to necessary readable paths, ordinary writable output/log setup, and CUDA visibility. Do not add a probe job, mandatory smoke stage, benchmark, architecture/version gate, full pixel scan, large artifact hashing, or synthetic GPU tests. Ordinary configuration/loading correctness checks remain in place.
- [x] Record repository revision/dirty state, supplied container identity, resolved settings, job IDs, and lightweight assigned-device metadata without auditing the environment. Surface ordinary scheduling/loading/training errors and out-of-memory failures directly.
- [x] Verify submission and argument forwarding with a fake scheduler: exactly two training submissions and one comparison submission, both parent IDs in `afterok`, log creation before submission, resource overrides, suffix normalization, epoch 1 propagation, and partial-failure recording. These are local development tests, not cluster startup gates.


## Comments

Implemented `scripts/shell/instance_seg/submit_instance_grace_experiment.sh` with
shared orchestration in `lfm/all_models/inst_seg/grace_experiment.py`. Submission
uses system Python and the standard library; the selected container prepares the
existing configuration and filename-only split manifest before any scheduler
call. The shared preparation CLI definitions moved into `paired_experiment_cli.py`
while preserving their existing public imports and defaults.

The command reserves a fresh run, creates output/cache/log directories, and
submits exactly two independent single-GPU training jobs and one comparison job
with `afterok` naming both normalized parent IDs. `submission.json` is atomically
updated before and after each scheduler call and retains commands, responses,
returned IDs, errors, resource settings, repository revision/dirty state, supplied
container identity, and configuration/split identities. Partial submission never
automatically cancels or duplicates surviving jobs.

Every allocated GPU stage uses the same Apptainer launcher with clean environment,
NVIDIA support, explicit Slurm CUDA visibility, headless plotting, read-only
code/data/weights/configuration, and writable run/cache paths. Host environment
injection channels and implicit home/current-directory mounts are disabled.
Actual GPU processes record lightweight assigned-device metadata and status in
`runtime/<stage>.json`, require visible CUDA, and directly run the existing
training or fresh best-checkpoint comparison entry point. Loading, training, OOM,
and scheduler errors propagate without retries or batch reduction.

Container/data/weights, JSON configuration defaults, output/cache/log paths,
account/QoS, training resources, independent comparison resources, and inference
batch size are exposed. Positive epoch overrides reach both model commands
unchanged; default 50 and explicit 1 preserve the existing batch/band/split
protocol. The design document includes runnable fresh submission examples,
configuration semantics, actual output layout, partial-failure behavior, and the
boundary with the explicit recovery workflow in ticket 07.

Validation: 17 new local tests pass with scheduler/Apptainer and GPU API doubles.
They cover the three submissions, pre-existing log/manifests/output setup,
suffix normalization, both dependency parents, epoch 1 forwarding, default full
protocol, independent resource/inference overrides, JSON/CLI resolution, partial
failures at each submission, malformed IDs, ordinary preparation failure, clean
container bindings, Slurm visibility, standard-library host execution, device
metadata, and direct CUDA/OOM errors. Ruff, formatting, shell syntax, and scoped
whitespace checks pass. Another 59 existing preparation, checkpoint lifecycle,
and checkpoint-comparison checks pass (76 focused checks total). The first wider
run exposed missing local `pycocotools`; the existing `/tmp/lfm-mask-ap-deps`
copy supplied via `PYTHONPATH` allowed all affected checks to pass without changing
the repository environment or production startup. No real Slurm, Apptainer GPU, full pretrained backbone, or
HPC execution was performed. No development check was added as a startup gate.
