# `enerdatasynth.resources.wind.sampling`

**Wind's batch-generation convenience tooling** — mirrors
[`resources/pjm_load/sampling/`](../../pjm_load/sampling/README.md) for structural
parity, added after pjm_load's. The two aren't used identically: load's
`generate_for_run` generates one sample per real validation condition (for in-sample
metric computation); wind's `generate_for_regimes` generates a fixed batch per
wind-regime bucket instead — there's no equivalent "real condition vectors" dataset to
line up against here.

## Files

- **`generate.py`** — `generate_for_regimes(preset="3ch", regimes=("low","mid","high"),
  n_per_regime=100, seed=None, **kwargs)`: thin wrapper looping
  `resources.wind.api.generate_wind_profile()` once per regime, offsetting `seed` by
  `+i` per regime so outputs differ reproducibly instead of repeating.

## Public API vs. internal

Not re-exported at the top level. For everyday single-regime generation, use
`enerdatasynth.model("wind", ...).sample(...)` (Tier 1) or
`resources.wind.api.generate_wind_profile()` directly instead — this module is purely a
multi-regime convenience on top of both.
