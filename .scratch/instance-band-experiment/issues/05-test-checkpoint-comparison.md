# 05: Compare both best checkpoints on the test set

**What to build:** A standalone comparison reloads both best-validation checkpoints, evaluates them on the same full test split, and produces the requested four-row figure using identical chips across columns.

**Blocked by:** 04 — Select best checkpoints and resume full training state.

**Status:** done

- [x] Accept two explicit model-result manifests and resolve their exact best-validation checkpoints. Reject missing, incomplete, stale, ambiguous, or mismatched experiment/configuration/split artifacts during normal loading. Do not discover checkpoints by filename order, latest epoch, or test performance.
- [x] Reconstruct each recorded architecture, all 70 inputs, initialization/configuration, normalization, and validity treatment. Strictly load Toy, infer in evaluation mode without gradients, release its GPU state, then do the same for GRAHA. Both checkpoints are freshly loaded in this process.
- [x] Evaluate all eligible test chips with the shared mask AP evaluator and matched detection settings. Preserve independent overlapping masks/scores for metrics and record matching eligibility/skips. Default inference batch size to 32 with an explicit override and no silent reduction.
- [x] Select displayed chip identifiers deterministically before examining predictions; default to five with an override. Supply a common raw cropped RGB display using stored bands 3, 1, and 0; validate chip identity, geometry, ground-truth partitions, and validity across model exports.
- [x] Reuse the existing shared renderer for exactly four rows: input chip, Toy prediction, GRAHA prediction, ground truth. Preserve deterministic colors, red boxes, and gray NoData regions; identify test split and selected checkpoints. Reuse the existing fake-data preview without introducing model imports or measured metrics into it.
- [x] Write full-test metrics, comparison/provenance manifest, metric-capable prediction artifacts, visualization caches, and figure output. Include checkpoint paths/epochs, display chip IDs/bands, confidence settings, and validity policy. Aggregate metrics cover the whole test split rather than just displayed chips.
- [x] Verify both explicit checkpoint loads, independently normalized inputs, best-versus-latest selection, empty/NoData examples, same-chip four-row output, and rejection of mixed experiments. The comparison can run again without training or using training-time caches as its inference source.


## Comments

Implemented the standalone entry point
`scripts/python/instance_seg/compare_instance_experiment.py` and shared inference
workflow in `lfm/all_models/inst_seg/comparison.py`. Both explicit result manifests
are validated against their prepared experiment/configuration and split identity.
The retained latest checkpoint supplies selection/progress evidence; inference
strictly reloads the exact recorded best checkpoint, independently for Toy and
GRAHA. Checkpoint callback identity, initialization, selection score/epoch/path,
and full-state completeness must match. Model construction reuses the existing
builders and data adapters, including the recorded training epoch override and
independent normalization. Toy completes and releases GPU state before GRAHA is
constructed. Inference uses evaluation mode without gradients and the shared
validation detector/AP protocol across the entire eligible test split.

Default inference batch size is 32 with `--batch-size` for an explicit override;
OOM errors propagate. Display IDs are the sorted eligible prefix, fixed during
ordinary loading before the first prediction, defaulting to five and configurable
with `--n-samples`. Both exports must match identity, crop geometry, ground-truth
partitions, common/per-band validity, raw RGB crops, eligibility, and skip reasons.
The existing renderer supplies exactly four rows using raw cropped stored bands
3,1,0, deterministic colors, red boxes, gray NoData, test labels, and selected
checkpoint epochs. The synthetic preview remains unchanged and model-free.

The output includes full-test metrics, a success/provenance manifest, per-chip
metric artifacts retaining independent overlapping bit-packed masks and scores,
selected-chip visualization caches, and the figure. Display confidence filtering
does not affect AP. A rerun uses a fresh destination and reloads the checkpoints;
training and training-time prediction caches are never inference sources. The
source design documents the command, defaults, original asset paths, output
layout, and mask unpacking. Success is published only after inference and rendering
finish; failures cannot leave a complete comparison manifest.

Validation: 102 focused and related tests passed, including 29 new comparison
checks. Coverage includes both real downstream wrappers with small Mask R-CNNs,
non-monotonic best-versus-latest selection, strict checkpoint state and provenance,
recorded budget overrides, independently normalized 70-channel inputs, full-test
coverage beyond the displayed subset, AP replay from exported masks, overlapping
low-score detections, empty/NoData chips, matched four-row output, reruns, OOM
propagation, and incompatible/missing/stale/ambiguous artifact rejection. The
existing multiprocessing loader check ran separately with normal process/shared-
memory permissions; the remaining 101 checks ran in the sandbox. Ruff and format
checks pass for the three new Python files, as do compilation, CLI help, and
relevant whitespace checks. The generated real-inference test figure was visually
inspected. Unrelated pre-existing workspace edits were preserved. No full
pretrained backbone or GPU/HPC inference ran, and no development check was added
to production startup. Ticket 06 is now unblocked.
