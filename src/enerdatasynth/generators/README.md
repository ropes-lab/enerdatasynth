# `enerdatasynth.generators`

This is the layer that makes PJM load and wind feel like one product. Everything under `src/enerdatasynth/resources/` is specific to one resource; this package is what drives both the same way. `BaseGenerator` is the contract each resource's generator implements, so `ResourceModel` in `sdk.py` can use any of them without knowing what is underneath.

## Files

- **`base.py`**: `BaseGenerator`, an abstract base class (`domain` plus `.sample()`). Subclass it without implementing both and you get a `TypeError` when you instantiate it.
- **`load.py`**: `LoadDiffusionModel`. Handles checkpoint loading, architecture selection, the DDPM sampler and calibration in one object.
- **`load_conditioning.py`**: `LOAD_CONDITION_SPEC`, `_load_construct` and `_load_checkpoint_status`, plus the `register_resource(...)` call that adds `"pjm_load"`. A third resource would copy this file.
- **`wind.py`**: `WindDiffusionModel`, which wraps `resources/wind/`. Checkpoints for all three presets are bundled under `data/checkpoints/wind/`.
- **`wind_conditioning.py`**: wind's registration and its `condition_builder`, which exists because wind's conditions don't fit `ConditionField`. The module docstring explains why.

## Public API

`BaseGenerator` and `LoadDiffusionModel` are re-exported at the top level (Tier 2). `WindDiffusionModel` is available the same way, but for everyday use call `enerdatasynth.model("wind", zone=<preset>)` (Tier 1).

## Adding a resource

Copy the pattern in `load_conditioning.py`: a `ConditionField` list (or a `condition_builder` if the shape doesn't fit, as with wind), a `_construct(zone)` hook that returns a `ConstructResult`, a `register_resource(...)` call at import time, and one import line in this folder's `__init__.py`. `sdk.py` needs no changes. The full walkthrough is in `docs/extending.md`.

## Be careful with

`LOAD_CONDITION_SPEC` has a runtime assertion tying its width to `utils.config.COND_DIM`. If you change one without the other, generation breaks inside the model with a shape mismatch rather than a clear error at the boundary.

## Scientific status

Wind's checkpoints are only validated in the sense that they load and produce output of the right shape. `docs/research_status.md` explains the difference between "code-complete" and "scientifically validated". Wind results don't belong in `docs/results.md`, which covers load only.
