# `data/raw_data/`

**Not read by any code path automatically** — unlike `data/raw/` (which
`enerdatasynth preprocess` has a hardcoded default pointing at), this folder is a
convention-only placeholder location, currently just for wind:

- `wind_data/` — a placeholder (`add_raw_data_here.txt`) for wind's raw training data,
  which does not exist anywhere in this repository or its dependencies. See
  [`docs/models/wind.md`](../../docs/models/wind.md) §2 for the full explanation — wind's
  bundled checkpoints work for generation today, but nothing here can retrain or
  independently validate them against ground truth.

If you're looking for where to put **PJM load** raw data, that's
[`data/raw/`](../raw/README.md), not here — the two folders' similar names are a known,
slightly confusing naming inconsistency in this repository (flagged, not silently
renamed, since `data/raw/` is load-bearing config and `wind_data/`'s parent folder name
isn't referenced by any code either way).
