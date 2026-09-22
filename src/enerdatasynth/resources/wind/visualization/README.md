# `enerdatasynth.resources.wind.visualization`

Can't be run against real data in this repository. Unlike
[`resources/pjm_load/visualization/`](../../pjm_load/visualization/README.md), there's
no real wind data here to plot against (`docs/models/wind.md` §9). The folder mirrors
load's layout and keeps the upstream plotting code.

## Files

- **`plotting.py`:** real-vs-synthetic comparison plots from the upstream
  `wind_diffusion` project, including `compare_to_real`, which made the plots in
  `docs/models/wind.md` §9 upstream. Needs real wind data.

## Public API vs. internal

Not exported at the top level. Nothing in the `enerdatasynth` wrapper or CLI calls it.
