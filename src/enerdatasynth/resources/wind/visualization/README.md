# `enerdatasynth.resources.wind.visualization`

Not runnable against real data in this repository. Unlike [`resources/pjm_load/visualization/`](../../pjm_load/visualization/README.md)'s real-vs-generated plots, there's no real wind data here to plot against (`docs/models/wind.md` §9). This subpackage exists for structural parity and to document what the vendored pipeline provides.

## Files

- **`plotting.py`** — real-vs-synthetic comparison plots, vendored from the upstream `wind_diffusion` project. Needs real wind data this repository doesn't ship.

## Public API vs. internal

Not re-exported at the top level, and not called anywhere in the `enerdatasynth` bridge or CLI. Nothing in this repository currently invokes it.
