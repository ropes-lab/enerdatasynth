# `enerdatasynth.resources.pjm_load.sampling`

Batch generation for PJM load. Wind generates through `enerdatasynth.model("wind", ...).sample(...)` and `generators/wind.py` and has nothing like this folder, because it has no from-scratch training workflow to generate research datasets for. This folder is separate from training but closely related to it; see [`../training/`](../training/README.md) for the shared train-then-generate workflow.

## Files

- **`generate.py`**: `generate_for_run(...)` makes one synthetic sample for every real condition vector in a dataset split (train or validation). The in-sample validation used throughout this project needs the generated and real sets to line up one-to-one by condition (`docs/results.md` explains). It reads the processed arrays, so it needs your own `enerdatasynth preprocess` output.

## Public API

`generate_for_run` is re-exported as `enerdatasynth.generate_for_run` for Tier 2 (research) use. For everyday generation use `enerdatasynth.model(...).sample(...)` (Tier 1, in `sdk.py`), which doesn't need a dataset's condition vectors.

## Reproducibility

The original version didn't reseed the random generators before sampling, which `generators/load.py` notes in its docstring. The fix is in `LoadDiffusionModel.sample()` (pass `seed=` there). `generate_for_run` still doesn't reseed on its own.

## What doesn't belong here

Training logic goes in `training/` and metric computation in `evaluation/`. This module only generates.
