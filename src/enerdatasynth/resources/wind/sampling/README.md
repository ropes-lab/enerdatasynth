# `enerdatasynth.resources.wind.sampling`

Wind's batch-generation convenience tooling, added to mirror [`resources/pjm_load/sampling/`](../../pjm_load/sampling/README.md) after pjm_load's version already existed. The two aren't used identically, though: load's `generate_for_run` generates one sample per real validation condition for in-sample metric computation, while wind's `generate_for_regimes` just generates a fixed batch per wind-regime bucket — there's no equivalent "real condition vectors" dataset to line up against here.

## Files

- **`generate.py`** — `generate_for_regimes(preset="3ch", regimes=("low","mid","high"), n_per_regime=100, seed=None, **kwargs)`, a thin wrapper that loops `resources.wind.api.generate_wind_profile()` once per regime, offsetting `seed` by `+i` each time so outputs differ reproducibly instead of repeating.

## Public API vs. internal

Not re-exported at the top level. For everyday single-regime generation, use `enerdatasynth.model("wind", ...).sample(...)` (Tier 1) or `resources.wind.api.generate_wind_profile()` directly — this module is purely a multi-regime convenience layered on top of both.
