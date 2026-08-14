# `enerdatasynth.sampling`

**PJM load's batch-generation tooling specifically** (wind generates through
`enerdatasynth.model("wind", ...).sample(...)` directly, via
`generators/wind.py` — it has no separate research-tooling equivalent of this
subpackage, since it has no from-scratch training workflow to generate research
datasets against). Decoupled from training itself; closely related to
[`../training/`](../training/README.md) — see that folder's README for the shared
"train then generate" workflow.

## Files

- **`generate.py`** — `generate_for_run(...)`: generates one synthetic sample per real
  condition vector in a dataset split (train or validation). Required for the in-sample
  validation methodology used throughout this project (`docs/results.md` notes this
  explicitly) — generated and real sets must line up one-to-one by conditioning.

## Public API vs. internal

`generate_for_run` is re-exported at the top level (`enerdatasynth.generate_for_run`),
Tier 2/research use. For everyday generation, use `enerdatasynth.model(...).sample(...)`
(Tier 1, in `sdk.py`) instead — it doesn't require a real dataset's condition vectors to
generate from.

## Assumptions this makes

Historically didn't reseed RNGs before sampling — a gap noted in
`generators/load.py`'s own docstring. Fixed at the `LoadDiffusionModel.sample()` layer
(pass `seed=` explicitly there), not here — this module's `generate_for_run` doesn't
independently reseed either.

## What should not be placed here

Training logic (belongs in `training/`) or metric computation (belongs in
`evaluation/`) — this module is generation only.
