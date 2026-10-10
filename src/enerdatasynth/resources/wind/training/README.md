# `enerdatasynth.resources.wind.training`

You can't run wind training from this repository. Unlike
[`resources/pjm_load/training/`](../../pjm_load/training/README.md), wind ships
pre-trained checkpoints (`data/checkpoints/wind/{1ch,2ch,3ch}/`). This folder mirrors
load's layout and keeps the upstream training code for reference. There is no
`enerdatasynth train --resource wind` command (see the `enerdatasynth.cli` module
docstring).

## Files

- **`train.py`:** the upstream `wind_diffusion` training entry point. It isn't installed
  as a command by this package's `pyproject.toml`. `docs/models/wind.md` §6 and §14
  give the recipe each bundled checkpoint used.

## Public API vs. internal

Not exported at the top level. Generation uses the bundled checkpoints and never calls
this module.
