# `data/outputs/wind/`

Empty. [`../pjm_load/`](../README.md)'s `outputs/` holds real files — `generated/` and `logs/` — written by `enerdatasynth train`/`enerdatasynth generate`'s CLI commands. Wind has no equivalent CLI surface: `enerdatasynth.model("wind", ...).sample(...)` returns its `(N, 288, 2)` array directly to the caller in memory, with nothing written to disk along the way, so there's no intermediate or logged output for this package to stage here.

Worth knowing if you're wondering where generated wind profiles "went": nowhere on disk unless you save the returned array yourself. Also worth knowing before generating a large batch: CPU generation is slow and scales poorly — `n=100` did not finish within 2.5 minutes on CPU with default settings, measured during the `docs/models/wind.md` documentation pass (§13) — and there's no checkpointing or partial-output mechanism to fall back on if you interrupt a large request.
