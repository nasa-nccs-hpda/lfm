# 07: Recover and rerun an interrupted experiment

**What to build:** A user can explicitly resume interrupted training or rerun only the checkpoint comparison while retaining the experiment's provenance and correct scheduler dependencies. The documented workflow covers both the full experiment and an optional one-epoch end-to-end smoke run.

**Blocked by:** 06 — Submit the three-job Grace container experiment.

**Status:** ready-for-agent

- [ ] Provide explicit recovery inputs selecting the experiment and each training checkpoint to resume. Restore full training state without silently restarting, changing batch size, or replacing initialization/normalization. Interpret `--max-epochs` as the total target budget.
- [ ] When training is resubmitted, persist replacement job IDs and rebuild the comparison dependency using the current two required training IDs. Retain previous submission history, and explicitly cancel or supersede stale comparison submissions so they cannot publish competing final results.
- [ ] Recover from partial submission without losing surviving job references or automatically duplicating successful training. Validate recovered state and required artifacts during normal loading; surface incompatible experiments and missing checkpoints clearly.
- [ ] Allow comparison-only reruns from the two authoritative completed result manifests without retraining. Fresh checkpoint inference still occurs and produces a complete, attributable metrics/figure result; reruns cannot silently mix stale comparison artifacts with new ones.
- [ ] Document runnable commands and expected outputs for a fresh 50-epoch run, an optional `--max-epochs 1` run, explicit resume, and comparison-only rerun. Explain model normalization, common NoData/AP rules, resource overrides, minimal startup, and where best/latest checkpoints and four-row figures are recorded.
- [ ] Mark an experiment complete only after its comparison has succeeded and its required metrics, manifests, and figure artifacts are present. Provide a consolidated Toy/GRAHA result table with provenance; repeated seeds, if explicitly configured, are reported separately with appropriate aggregation.
- [ ] Verify recovery with fake scheduler submissions and fixture manifests, including a failed parent, missing/mixed artifacts, replacement job IDs, superseded dependencies, and comparison-only execution. Exercise the one-epoch path through checkpoint production and comparison with focused development fixtures; leave real HPC execution clearly identified as unverified unless actually run. Never make a real smoke experiment a prerequisite for the full run.
