# 01: Resolve matched 70-band inputs for both models

**What to build:** Resolve one experiment into Toy and GRAHA configurations that load the same chips, targets, and crop geometry while preserving each model's input transformation. A developer can verify paired samples through the real data paths without training or adding a production probe stage.

**Blocked by:** None (can start immediately).

**Status:** done

- [x] Use the existing 70-band dataset, with an explicit dataset override, and all stored bands in order: five VIS, two UV, and 63 static. Select Toy DINO Mask R-CNN and GRAHA Mask R-CNN with native VIS/UV/static embeddings; reject a silent synthetic WAC substitution.
- [x] Build one persistent train/validation/test membership manifest through ordinary file pairing, with stable chip identifiers. Do not scan all chip pixels to determine eligibility at submission time.
- [x] Resolve normalization independently: Toy retains per-chip, per-band min/max scaling without z-score normalization; GRAHA retains its pretraining statistics. Record both policies and selected band identities.
- [x] Default to seed 42, 256-pixel center crops, batch size 32, unfrozen backbones, and 50 epochs. Expose a positive-integer `--max-epochs` override and record it for both models without changing unrelated training CLI defaults. Preserve documented model-specific optimizer and feature-pyramid choices.
- [x] Record model initialization, explicit pretrained-weight locations, dataset/configuration identity, and resolved settings. Do not claim that changing band count or crop size is supported by this baseline.
- [x] Focused fixture checks load matching chip identifiers, raw display crops, and target geometry through both model data paths while allowing different normalized tensors. Keep those checks in development verification rather than normal job startup.

## Comments

Implemented the paired configuration resolver, manifest preparation CLI, persisted
membership loading, and real Toy/GRAHA data-adapter factory. Both datasets and
detection collators retain chip identity and can optionally expose canonical raw
crops. Toy derives spectral initialization metadata from the first manifest chip,
so files appearing later cannot change its source. Public exports load lazily to
keep unrelated optional datacube-inference imports out of the data-only path.

Validation: 21 focused and related tests passed, covering the new paired-input
contract, existing comparison figures, and existing Mask R-CNN NoData helpers.
Fixtures use real 70-band GeoTIFF/NPZ/YAML data and compare both actual adapters;
GRAHA task arguments retain native embeddings. Manifest preparation succeeds
with intentionally unreadable raster bytes, confirming no pixel reads. CLI epoch
1 propagation, unchanged legacy defaults, frozen membership, and manifest identity
errors are covered. No HPC training or job submission was performed.

The source design contains runnable preparation and data-loading examples.
NoData training repairs remain in ticket 02; this ticket does not complete the
training or Slurm workflow. Ticket 02 is now unblocked.
