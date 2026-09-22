# `enerdatasynth.resources.wind.data`

Wind's data-handling code. There is no raw wind data in this repository to run it on
(`data/raw/wind/` is empty; see
[`docs/models/wind.md`](../../../../../docs/models/wind.md) §2). The folder mirrors
`resources/pjm_load/data/` and shows what input the pipeline expects.

## Files

- **`data.py`:** raw CSV loading, channel derivation for `1ch`/`2ch`/`3ch`, and splitting
  into 288-step (24-hour, 5-minute) windows. Needs `windspeed_100m` and
  `winddirection_100m` columns, which this repository doesn't include.
- **`transforms.py`:** wind-regime bucketing (low, mid, high) and the one-hot or ordinal
  regime encoding, which is set per checkpoint (`docs/models/wind.md` §4).

## Public API vs. internal

Not exported at the top level. Import `enerdatasynth.resources.wind.data.data` or
`.transforms` directly if you need them. Normal generation never uses this module,
because the checkpoints are already trained.

## What should not be placed here

Anything specific to `enerdatasynth`. Glue code belongs in `generators/wind*.py` (see
[`../README.md`](../README.md)).
