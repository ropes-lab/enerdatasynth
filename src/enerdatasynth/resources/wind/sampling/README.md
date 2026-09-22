# `enerdatasynth.resources.wind.sampling`

A helper for generating batches, added to match
[`resources/pjm_load/sampling/`](../../pjm_load/sampling/README.md). The two work
differently. Load's `generate_for_run` generates one sample for each real validation
condition, so metrics can be computed against them. Wind's `generate_for_regimes`
generates a fixed batch for each wind regime, because there are no real conditions here
to match.

## Files

- **`generate.py`:** `generate_for_regimes(preset="3ch", regimes=("low","mid","high"), n_per_regime=100, seed=None, **kwargs)`
  calls `resources.wind.api.generate_wind_profile()` once per regime. It adds `i` to
  `seed` for each regime so the batches differ but stay reproducible.

## Public API vs. internal

Not exported at the top level. To generate a single regime, use
`enerdatasynth.model("wind", ...).sample(...)` or
`resources.wind.api.generate_wind_profile()`. This module only adds the multi-regime
loop.
