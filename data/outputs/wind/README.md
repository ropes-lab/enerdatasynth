# `data/outputs/wind/`

Empty. For [`../pjm_load/`](../README.md), the `enerdatasynth train` and
`enerdatasynth generate` commands write files to `generated/` and `logs/`. Wind has no
CLI commands. `enerdatasynth.model("wind", ...).sample(...)` returns an `(N, 288, 2)`
array in memory and writes nothing to disk, so generated wind profiles are only saved if
you save them yourself.

Large batches are slow on CPU: `n=100` didn't finish within 2.5 minutes with default
settings (`docs/models/wind.md` §13), and an interrupted run leaves no partial output.
