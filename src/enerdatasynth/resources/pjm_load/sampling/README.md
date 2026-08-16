# `enerdatasynth.resources.pjm_load.sampling`

PJM load's batch-generation tooling specifically. Wind generates through `enerdatasynth.model("wind", ...).sample(...)` directly via `generators/wind.py`, with no separate research-tooling equivalent of this subpackage — it has no from-scratch training workflow to generate research datasets against in the first place. This folder is decoupled from training itself but closely related to [`../training/`](../training/README.md); see that README for the shared "train then generate" workflow.

## Files

- **`generate.py`** — `generate_for_run(...)` generates one synthetic sample per real condition vector in a dataset split, train or validation. The in-sample validation methodology used throughout this project needs this (`docs/results.md` spells it out) — generated and real sets have to line up one-to-one by conditioning.

## Public API vs. internal

`generate_for_run` is re-exported at the top level as `enerdatasynth.generate_for_run`, for Tier 2/research use. For everyday generation, use `enerdatasynth.model(...).sample(...)` instead (Tier 1, in `sdk.py`) — it doesn't need a real dataset's condition vectors to generate from.

## Assumptions this makes

Historically this didn't reseed RNGs before sampling, a gap noted in `generators/load.py`'s own docstring. That got fixed at the `LoadDiffusionModel.sample()` layer (pass `seed=` there), not here — `generate_for_run` still doesn't reseed independently.

## What should not be placed here

Training logic belongs in `training/`; metric computation belongs in `evaluation/`. This module is generation only.
