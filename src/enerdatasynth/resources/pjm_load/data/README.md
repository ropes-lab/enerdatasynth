# `enerdatasynth.resources.pjm_load.data`

**PJM load's raw-data pipeline specifically** — despite the generic name, this
subpackage only ever handles PJM/NOAA data. Wind has no equivalent preprocessing
pipeline in this package at all (it ships pre-trained; see
[`docs/models/wind.md`](../../../../../docs/models/wind.md) §2 for why there's nothing to
mirror here). This is one of the clearest examples of why the two resources'
subpackages aren't symmetric — see [`docs/architecture.md`](../../../../../docs/architecture.md)
for the full explanation.

Raw CSVs → cleaned, normalized daily tensors ready for training.

## Files

- **`preprocess.py`** — `run_preprocessing()`: the full pipeline (pivot to daily rows,
  detect/remove DST days, remove missing-load/missing-weather/COVID days, normalize,
  save `X_daily_profiles.npy` / `Z_conditions.npy` / `metadata.json` / train-val parquet
  files to `data/processed/pjm_load/`). See `docs/data_card.md` for the exact rules.
- **`temperature.py`** — NOAA ISD fetch, per-station-year Parquet caching
  (`data/cache/pjm_load/`), and zone-level equal-weighted averaging across each zone's
  constituent stations (`ZONE_STATIONS`).

## Public API vs. internal

`run_preprocessing` is re-exported at the top-level package (`enerdatasynth.run_preprocessing`).
Everything else here (`detect_dst_days`, the temperature-fetch internals) is
implementation detail, reached via `enerdatasynth.resources.pjm_load.data.preprocess`/`.temperature` if
needed directly.

## How to run it

```bash
enerdatasynth preprocess
```

Requires raw CSVs already present under `data/raw/pjm_load/` (see
`docs/data_card.md` — `pip install enerdatasynth` does not ship them).

## Assumptions this makes

- A fixed 24-hour daily profile — DST days are excluded rather than handled, because a
  23- or 25-hour day can't be represented in a fixed-width array.
- No imputation for missing load or weather — affected days are dropped, not filled.
- Zone temperature is an equal-weighted station average, not load- or population-
  weighted.

## What should not be changed casually

Changing the cleaning rules (DST/missing/COVID exclusion) changes what every downstream
checkpoint was actually trained on. If you change this, existing checkpoints' reported
metrics no longer describe the same dataset.
