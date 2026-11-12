# 02: Train both models with consistent NoData treatment

**What to build:** Both models can train on the matched dataset while retaining partially valid chips and supervising only valid instance-mask pixels. Toy keeps its own scaling instead of inheriting GRAHA statistics.

**Blocked by:** 01 — Resolve matched 70-band inputs for both models.

**Status:** done

- [x] Derive validity from finite raw values and declared NoData sentinels before normalization. Retain per-band image validity, and define the common target footprint as pixels valid in every selected band.
- [x] Toy min/max uses only each band's valid pixels. Invalid output pixels and entirely invalid bands become zeros; explicitly preserve the existing behavior for constant valid bands.
- [x] Carry the common validity mask through cropping, dataset adapters, target construction, and both Mask R-CNN training paths. Invalid pixels are excluded from instance mask loss rather than supervised as background.
- [x] Clip each ground-truth object to the valid footprint, retain its valid portion, remove wholly invalid objects, and recompute boxes. Keep empty-instance valid chips supported. Skip wholly invalid chips consistently during normal loading and record identifiers/reasons; fail clearly when a required split has no eligible samples.
- [x] Preserve GRAHA's own normalization and apply the same target policy to both models. Document that mask-loss exclusion does not provide a general pixelwise ignore mechanism for RPN or box losses.
- [x] Focused real-adapter and loss checks cover partial NoData, fully invalid bands/chips, constant bands, empty targets, boundary-clipped boxes, and finite forward/backward loss for both selected model routes. No separate cluster smoke job or silent batch-size fallback is introduced.

## Comments

Implemented shared raw validity, corrected Toy scaling before missing-pixel fill,
neutral normalized NoData values, cropped target clipping and box/count rebuilding,
and validity transport through both detection adapters and collators. Toy now uses
the same NoData-aware ROI mask loss as GRAHA, including its validation loss path.
RPN and box losses retain their existing semantics, as specified.

Wholly invalid crops are skipped during ordinary loading, with parent-process
JSON/JSONL reports. Eligible batches refill to the configured size; valid empty
targets remain supported. Entirely ineligible required splits raise during
iteration. The loader has no advertised length, avoiding skipped first-epoch
validation in Lightning. GRAHA's cosine budget is derived from nominal membership
when eligibility prevents an exact step estimate; it is logged and saved in
checkpoint hyperparameters. Explicit budgets and Toy's epoch scheduler remain.
Visualization cache round trips retain validity, chip identity, and ignored GT.

Validation: 40 tests passed in the requested lfm-codex environment, including both
real data adapters, real small TorchVision detector forward/backward passes through
both downstream wrappers, empty-only batches, zero invalid-bin gradients, worker
skip reporting, cache validity, and an actual one-epoch Toy Lightning fit with
validation. Existing matched-input, mask-loss, and synthetic-figure checks passed.
The multiprocessing test requires ordinary process/shared-memory permissions and
was run with those permissions. CPU test dependencies were installed in lfm-codex
with the user's authorization. No pretrained backbone or HPC training was run;
the GRAHA wrapper tests use a small detector-backed base rather than TerraTorch
initialization. No production preflight stages were added.

The source design documents runtime reports, scheduler budgeting, and verification
commands. Ticket 03 is now unblocked; shared AP and the final experiment runner
remain in their subsequent tickets.
