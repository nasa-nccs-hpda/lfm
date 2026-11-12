# Lunar Foundation Model experiments

Shared language for lunar raster inputs and comparisons of segmentation models.

## Language

**Toy model**:
The project's DINO-based model family used as a baseline for lunar segmentation
experiments. The name identifies a model family and does not imply a small model.

**GRAHA**:
The lunar foundation model family compared with the Toy model in downstream
segmentation experiments.

**Input band**:
One spatially aligned raster measurement available as an input to a lunar model.
_Avoid_: Modality when referring to a single raster measurement

**Modality**:
A named group of related lunar measurements, such as VIS, UV, or static, that can
contain one or more input bands.
_Avoid_: Band when referring to the measurement group

**70-band baseline**:
The initial Toy/GRAHA instance segmentation comparison using all five VIS, two UV,
and 63 static bands from the same dataset.

**Paired experiment**:
A Toy/GRAHA comparison with shared sample membership, target geometry, validity
rules, seed, and training budget, while retaining each model's normalization.

**Training attempt**:
One execution of training for one model in a paired experiment, either starting
fresh or resuming an earlier attempt.

**Training result**:
The completed training record for one model, identifying its paired experiment,
training budget, best validation checkpoint, and latest checkpoint.

**Prediction artifact**:
A saved prediction record retaining chip identity, independently scored instance
masks, targets, and validity, separately from its display representation.

**Experiment submission**:
The record linking a paired experiment's two training jobs and dependent
checkpoint-comparison job to their scheduler identities and execution settings.
