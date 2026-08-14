# `enerdatasynth.resources.wind.data`

**Wind's data-handling code — but there is no raw wind dataset in this repository to
run it against.** `data/raw/wind/` is empty (see
[`docs/models/wind.md`](../../../../../docs/models/wind.md) §2). This subpackage exists
for structural parity with `resources/pjm_load/data/` and to document what the vendored
pipeline expects, not because preprocessing is runnable here today.

## Files

- **`data.py`** — raw-CSV loading, channel derivation (`1ch`/`2ch`/`3ch`), and windowing
  into 288-step (24-hour, 5-minute) sequences. Needs `windspeed_100m` /
  `winddirection_100m` columns this repo doesn't ship.
- **`transforms.py`** — feature engineering: wind-regime bucketing (low/mid/high) and
  the onehot/ordinal regime encoding choice (a checkpoint-level setting, see
  `docs/models/wind.md` §4).

## Public API vs. internal

Not re-exported at the top level — reached via
`enerdatasynth.resources.wind.data.data`/`.transforms` directly if needed. Everyday
generation doesn't need this module at all; the bundled checkpoints already exist.

## What should not be placed here

Anything `enerdatasynth`-specific — resource-specific glue belongs in
`generators/wind*.py` instead, per [`../README.md`](../README.md).
