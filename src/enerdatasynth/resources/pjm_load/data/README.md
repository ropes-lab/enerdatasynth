# `enerdatasynth.resources.pjm_load.data`

Despite the generic name, this subpackage only ever handles PJM and NOAA data — it's PJM load's raw-data pipeline specifically. Wind has no equivalent preprocessing here at all; it ships pre-trained (see [`docs/models/wind.md`](../../../../../docs/models/wind.md) §2 for why there's nothing to mirror). This is one of the clearest examples of why the two resources' subpackages aren't symmetric. [`docs/architecture.md`](../../../../../docs/architecture.md) has the full explanation.

In short: raw CSVs in, cleaned and normalized daily tensors out, ready for training.

## Files

- **`preprocess.py`** — `run_preprocessing()` runs the full pipeline: pivot to daily rows, detect and remove DST days, drop missing-load/missing-weather/COVID days, normalize, then save `X_daily_profiles.npy`, `Z_conditions.npy`, `metadata.json`, and train/val parquet files to `data/processed/pjm_load/`. `docs/data_card.md` has the exact rules.
- **`temperature.py`** — NOAA ISD fetch, per-station-year Parquet caching (`data/cache/pjm_load/`), and zone-level averaging across each zone's constituent stations (`ZONE_STATIONS`), weighted equally.

## Public API vs. internal

`run_preprocessing` is re-exported at the top level as `enerdatasynth.run_preprocessing`. Everything else — `detect_dst_days`, the temperature-fetch internals — is implementation detail; reach it through `enerdatasynth.resources.pjm_load.data.preprocess`/`.temperature` if you need it directly.

## How to run it

```bash
enerdatasynth preprocess
```

Needs raw CSVs already sitting under `data/raw/pjm_load/` — `pip install enerdatasynth` doesn't ship them (see `docs/data_card.md`).

## Assumptions this makes

- A fixed 24-hour daily profile. DST days get excluded rather than handled, because a 23- or 25-hour day just doesn't fit a fixed-width array.
- No imputation for missing load or weather. Affected days are dropped, not filled in.
- Zone temperature is an equal-weighted station average — not load-weighted, not population-weighted.

## What should not be changed casually

The cleaning rules (DST/missing/COVID exclusion) define what every downstream checkpoint was actually trained on. Change them and existing checkpoints' reported metrics stop describing the same dataset.
