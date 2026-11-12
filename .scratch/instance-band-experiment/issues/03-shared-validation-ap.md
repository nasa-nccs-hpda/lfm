# 03: Measure shared mask AP during full validation

**What to build:** Both training routes report comparable COCO-style instance mask AP on the entire eligible validation split after each epoch, using the same evaluation rules and independent scored detections.

**Blocked by:** 02 — Train both models with consistent NoData treatment.

**Status:** done

- [x] Feed a shared COCO-style mask AP evaluator with independent per-instance masks and confidence scores; retain overlapping detections. Report aggregate AP over IoU 0.50:0.95 and document the common detection limits and postprocessing.
- [x] Apply the common validity footprint to prediction and ground-truth masks, remove predictions with no valid pixels, and handle empty-target/empty-prediction chips consistently across models.
- [x] Evaluate every eligible validation chip with no default sample cap. Publish the same metric key and aggregation semantics from both training paths, including at the end of epoch one.
- [x] Keep visualization confidence filtering separate from metric collection. Flattened instance-ID rasters and confidence-truncated plotting caches are never AP inputs.
- [x] Training and checkpoint-selection hooks use validation only; disable inherited per-epoch test evaluation in this experiment. The held-out test split remains for final checkpoint comparison.
- [x] Verify perfect detections, false positives, missed instances, overlapping masks, empty targets, NoData clipping, and evaluation independence from plotting thresholds. Check that both model routes reach the shared evaluator and cover the full eligible validation membership.

## Comments

Implemented `lfm/all_models/inst_seg/mask_ap.py` with COCO run-length mask encoding,
whole-split matching/accumulation, and shared validation lifecycle hooks. Both
selected downstream wrappers publish `val_mask_ap` and `val_mask_ap_chips` after
every validation epoch. The shared AP uses IoU 0.50:0.95, 101 recall points, all
areas, the crater class, and a documented 1,000-detection limit for dense chips.
It reads COCO's precision tensor directly because the upstream aggregate summary
hardcodes maxDets=100. Both checkpoint hooks now monitor this validation key;
best/latest retention remains ticket 04.

Metric inference temporarily applies identical detector/RPN limits and thresholds,
then restores the original settings. GRAHA bypasses inherited task-level score
filtering/NMS; Toy retains validation loss and adds unfiltered metric inference.
Both sides are clipped to the common validity mask. Overlapping detections remain
independent. Valid negative chips remain in the split; an entirely target-free
split retains COCO's undefined AP sentinel (-1). Plot caches never enter AP.
Paired configurations reject per-epoch test evaluation. GFFT opts out of the new
validation hooks to preserve its existing behavior.

Validation: 54 focused and related tests passed in `lfm-codex`, including 14 new
AP checks, both real data adapters, and actual one-epoch Lightning fits through
both downstream wrappers with small CPU Mask R-CNNs. They verify full eligible
validation membership, epoch-one AP/checkpoints, and no test traversal. The final
GRAHA monitor assignment also passed a targeted one-epoch rerun. Ruff passes for
all changed Python files. The existing worker regression needed normal process
and shared-memory permissions; the sandboxed attempt was stopped and the focused
suite rerun successfully with those permissions. `pycocotools` (already declared
in container requirements) was installed under `/tmp/lfm-mask-ap-deps` for local
verification, with that directory on `PYTHONPATH`; no production preflight stages
were added. No pretrained backbone or HPC training ran. The GRAHA tests use a
detector-backed base matching TerraTorch's output wrapper.

The source design documents the protocol and verification command. Ticket 04 is
now unblocked; checkpoint lifecycle, held-out comparison, and submission remain
in subsequent tickets.
