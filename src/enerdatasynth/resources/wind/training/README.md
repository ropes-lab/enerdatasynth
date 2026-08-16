# `enerdatasynth.resources.wind.training`

Not a runnable training workflow in this repository. Unlike [`resources/pjm_load/training/`](../../pjm_load/training/README.md), wind ships pre-trained, bundled checkpoints (`data/checkpoints/wind/{1ch,2ch,3ch}/`). This subpackage exists for structural parity and to document the upstream recipe — `enerdatasynth train --resource wind` doesn't exist (see `enerdatasynth.cli`'s module docstring).

## Files

- **`train.py`** — the upstream `wind_diffusion` project's training entry point. It's not registered as an installable console script in this package's `pyproject.toml`; it describes the original project's intended invocation, not something this repository currently exposes. `docs/models/wind.md` §6/§14 covers the training recipe each bundled checkpoint actually used.

## Public API vs. internal

Not re-exported at the top level, and not reachable through `enerdatasynth.model("wind", ...)` at all. Generation uses the bundled checkpoints directly and never touches this module.
