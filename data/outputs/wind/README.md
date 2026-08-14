# `data/outputs/wind/`

Empty. Wind generates directly through `enerdatasynth.model("wind", ...).sample(...)`
without writing intermediate output files the way PJM load's training/generation
CLI does — there's nothing for this package to write here today. Exists as the
resource-nested sibling of [`../pjm_load/`](../README.md) for structural
consistency.
