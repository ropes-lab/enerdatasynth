# `enerdatasynth.resources.wind.training`

**Not a runnable training workflow in this repository** — unlike
[`resources/pjm_load/training/`](../../pjm_load/training/README.md), wind ships
pre-trained, bundled checkpoints (`data/checkpoints/wind/{1ch,2ch,3ch}/`). This
subpackage exists for structural parity and to document the upstream recipe, not
because `enerdatasynth train --resource wind` exists (it doesn't — see
`enerdatasynth.cli`'s module docstring).

## Files

- **`train.py`** — the upstream `wind_diffusion` project's training entry point.
  Not registered as an installable console script by this package's `pyproject.toml`;
  describes the original project's intended invocation, not something this repository
  currently exposes. See `docs/models/wind.md` §6/§14 for the training recipe each
  bundled checkpoint actually used.

## Public API vs. internal

Not re-exported at the top level and not reachable through
`enerdatasynth.model("wind", ...)` at all — generation uses the bundled checkpoints
directly, never this module.
