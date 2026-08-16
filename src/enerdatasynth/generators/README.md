# `enerdatasynth.generators`

This is where PJM load and wind become one product. Everywhere else under `src/enerdatasynth/resources/` is per-resource; this subpackage is the shared layer that drives both the same way. `BaseGenerator` is the common contract every resource-type generator implements, so `sdk.py`'s `ResourceModel` can drive any of them identically regardless of what's underneath.

## Files

- **`base.py`** — `BaseGenerator`, an enforced ABC (`domain` plus `.sample()`). Subclass it without implementing both and you get a `TypeError` at instantiation.
- **`load.py`** — `LoadDiffusionModel`. Wraps checkpoint loading, architecture selection, the DDPM sampler, and calibration behind one object.
- **`load_conditioning.py`** — `LOAD_CONDITION_SPEC`, `_load_construct`/`_load_checkpoint_status`, and the `register_resource(...)` call that adds `"pjm_load"`. This is the reference implementation a third resource would copy.
- **`wind.py`** — `WindDiffusionModel`, wrapping `resources/wind/`. Real checkpoints for all three presets are bundled under `data/checkpoints/wind/`.
- **`wind_conditioning.py`** — wind's registration, plus its `condition_builder`, an escape hatch from `ConditionField` (the module docstring explains why wind needs it).

## Public API vs. internal

`BaseGenerator` and `LoadDiffusionModel` are re-exported at the top level (Tier 2). `WindDiffusionModel` is reached the same way, or through `enerdatasynth.model("wind", zone=<preset>)` for everyday use (Tier 1).

## Adding a new resource type

Copy `load_conditioning.py`'s pattern: a `ConditionField` list (or a `condition_builder`, if the shape doesn't fit — wind needed one), a `_construct(zone)` hook returning a `ConstructResult`, a `register_resource(...)` call at import time, and one import line in this folder's `__init__.py`. `sdk.py` needs no edits. Full walkthrough in `docs/extending.md`.

## What should not be changed casually

`load_conditioning.py`'s `LOAD_CONDITION_SPEC` has a runtime assertion tying its width to `utils.config.COND_DIM`. Change one without the other and generation breaks silently — a shape mismatch inside the model, not a clear error at the boundary.

## Where the scientific provenance lives

Wind's checkpoints are validated only in the sense that they load and produce plausible-shaped output — see `docs/research_status.md` for the distinction between "code-complete" and "scientifically validated." No wind result belongs in `docs/results.md`, which is load-only.
