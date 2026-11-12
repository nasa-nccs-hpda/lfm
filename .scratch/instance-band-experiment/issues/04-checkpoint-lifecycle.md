# 04: Select best checkpoints and resume full training state

**What to build:** Each training run produces an authoritative best-validation checkpoint and a resumable latest checkpoint, with enough provenance for downstream comparison to reload the intended model. An explicitly selected one-epoch run produces the same required artifacts.

**Blocked by:** 03 — Measure shared mask AP during full validation.

**Status:** done

- [x] Select each model's best checkpoint using the shared full-validation mask AP. Save a latest full-state checkpoint containing optimizer/scheduler state and training progress; distinguish explicit resume from weights-only initialization.
- [x] Publish a result manifest recording experiment/configuration and split identities, model architecture, normalization and validity policies, resolved epoch budget, initialization, best checkpoint path/epoch/score, and latest checkpoint path. Include sufficient sampler/initialization information for consistent resume.
- [x] A positive `--max-epochs` reaches each actual trainer and relevant scheduling configuration unchanged. Default to 50; a value of 1 validates, saves both required checkpoints, and publishes a complete manifest after the first epoch.
- [x] Explicit resume restores training state and interprets the epoch budget as the total target, including completed epochs. A fresh run does not implicitly discover or load an earlier checkpoint.
- [x] Report success only after the required checkpoints and result manifest exist. Handle a missing/non-finite selection metric or incomplete artifact publication with a direct failure rather than an ambiguous successful result.
- [x] Verify best-versus-latest selection with a deliberately non-monotonic validation sequence, one-epoch artifact production, full-state restoration, and rejection of incompatible state or experiment identity. Keep these development checks outside production startup.

## Comments

Implemented the paired direct training entry point
`scripts/python/instance_seg/train_instance_experiment.py` and the shared lifecycle
in `lfm/all_models/inst_seg/training.py`. It uses the prepared experiment and split
manifest, matched data adapters, and existing architecture builders. Each model
saves its best full-validation `val_mask_ap` checkpoint, full-state `last.ckpt`,
and an atomically published `result.json`. The manifest includes architecture,
normalization/validity, initialization, budget, sampler, identities, and explicit
best/latest selection. Epoch indices are zero-based.

Explicit `--resume` restores model, optimizer, scheduler, loop, selection, and
Python/NumPy/Torch/CUDA RNG state. The epoch budget remains the original total
target; incompatible identities/budgets or already-completed targets fail directly.
`--initialize-weights` strictly loads weights into a fresh run with new training
state. Existing artifacts require explicit recovery or a fresh run root. Sampling
is addressable by seed and epoch, with deterministic preprocessing and separate
loader generators, including when invalid chips are skipped during loading.

The callback explicitly updates latest on declining AP: the installed Lightning
release otherwise only updated it alongside a top-k save. Pruning waits until
latest is durable so a failed write retains the previous latest checkpoint's
selected best. Missing/non-finite/undefined AP, missing checkpoint files,
incomplete epoch progress, and publication failures cannot report success.
Unrelated Transformers imports now occur only when a Mask2Former path uses them,
allowing the actual DINO builder to be checked without that optional dependency.
Legacy standalone notebook workflows retain their existing lifecycle.

Validation: 73 focused and related tests passed in `lfm-codex`, including 19 new
lifecycle checks. They cover non-monotonic selection, both real downstream wrappers'
one-epoch checkpoint/manifest production, production-builder epoch forwarding,
exact uninterrupted-versus-resumed state and sample order, both real eligible
loaders, strict incompatible-state/identity rejection, weights-only initialization,
and failed artifact publication. The suite includes all ticket 03 regression files;
the existing worker test ran with normal process/shared-memory permissions after
the sandboxed run was stopped. Local verification used
`PYTHONPATH=/tmp/lfm-mask-ap-deps:.`, `MPLBACKEND=Agg`, and one CPU thread, as in
ticket 03. No dependencies were added to the production requirements. Ruff passes
for the new lifecycle files and other touched files; `iseg_model.py` retains 13
pre-existing lint findings (down from 15), with no new findings. Relevant diff
whitespace checks pass. The source design documents commands and output semantics.
No pretrained backbone or GPU/HPC training ran; development checks are not startup
stages. Ticket 05 is now unblocked.
