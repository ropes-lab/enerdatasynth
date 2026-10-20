# `enerdatasynth.resources.pjm_load.data`

The name is generic, but this package only handles PJM and NOAA data. It is the raw-data pipeline for load. Wind has nothing equivalent because it ships pre-trained ([`docs/models/wind.md`](../../../../../docs/models/wind.md) §2), which is one of the main reasons the two resource packages aren't symmetric. [`docs/architecture.md`](../../../../../docs/architecture.md) has the longer explanation.

In short: raw CSVs go in, and cleaned, normalized daily tensors come out, ready for training.

## Files

- **`preprocess.py`**: `run_preprocessing()` runs the whole pipeline. It pivots to daily rows, removes DST days, drops days with missing load, missing weather or COVID, normalizes, and saves `X_daily_profiles.npy`, `Z_conditions.npy`, `metadata.json` and the train/val parquet files to `data/processed/pjm_load/`. Git ignores everything there except `metadata.json`. `docs/data_card.md` lists the exact rules.
- **`temperature.py`**: NOAA ISD fetching, per-station-year parquet caching in `data/cache/pjm_load/`, and averaging across each zone's weather stations (`ZONE_STATIONS`), equally weighted.

## Public API

`run_preprocessing` is re-exported as `enerdatasynth.run_preprocessing`. Everything else, including `detect_dst_days` and the temperature-fetch internals, is an implementation detail. Import from `enerdatasynth.resources.pjm_load.data.preprocess` or `.temperature` if you need it.

## Running it

```bash
enerdatasynth preprocess
```

It needs the raw PJM and NOAA files in `data/raw/pjm_load/`. Neither the package nor the repository includes them; `docs/data_card.md` says where to get them.

## Assumptions

- Each day is a fixed 24-hour profile. DST days are dropped instead of handled, because a 23- or 25-hour day doesn't fit a fixed-width array.
- Nothing is imputed. A day with missing load or weather is removed, not filled in.
- Zone temperature is a plain average over stations, not weighted by load or population.

## Be careful with

The cleaning rules (DST, missing data, COVID exclusion) define what every checkpoint was trained on. Change them and the reported metrics no longer describe the same dataset.
