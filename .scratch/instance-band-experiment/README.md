# Toy versus GRAHA instance segmentation tickets

Source: [approved experiment design](../../docs/instance_band_experiment.md).

The user approved this seven-ticket breakdown. The tickets are published as
individual Markdown files under `issues/`. Tickets 01–06 are complete; ticket
07 is now unblocked and retains status `ready-for-agent` with its declared
predecessor dependency.

The user requested a strictly sequential chain: 01 → 02 → 03 → 04 → 05 → 06 → 07.
Each ticket lists only its immediate predecessor; earlier requirements are
transitive. This ordering deliberately serializes the work, even where isolated
implementation could otherwise proceed in parallel. Work only
the first unfinished ticket whose predecessor is complete.

1. [Resolve matched 70-band inputs for both models](issues/01-matched-model-inputs.md)
2. [Train both models with consistent NoData treatment](issues/02-nodata-safe-training.md)
3. [Measure shared mask AP during full validation](issues/03-shared-validation-ap.md)
4. [Select best checkpoints and resume full training state](issues/04-checkpoint-lifecycle.md)
5. [Compare both best checkpoints on the test set](issues/05-test-checkpoint-comparison.md)
6. [Submit the three-job Grace container experiment](issues/06-grace-experiment-submission.md)
7. [Recover and rerun an interrupted experiment](issues/07-experiment-recovery.md)

The synthetic four-row renderer and preview already exist. Reuse them rather
than assigning completed work to a new implementation ticket. An increasing-band
sweep, partial native modalities, Mask2Former changes, and unrelated checkpoint
sweep repairs are outside this baseline.

Each ticket includes its own verification. Development checks and optional
one-epoch experiments must not become mandatory production preflight stages.
