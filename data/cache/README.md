# `data/cache/`

Empty by design — not distributed with this repository.

`src/enerdatasynth/data/temperature.py` populates this folder itself, one
`{station}_{year}.parquet` file at a time, the first time NOAA ISD temperature data for
that station-year is needed. It exists purely so repeated preprocessing runs don't
re-download the same station-year twice — deleting everything in this folder is always
safe; it will be rebuilt automatically (given network access) the next time
`enerdatasynth preprocess` runs.

Nothing here is source data specific to this project — it's a generic NOAA ISD cache
keyed by station and year. See `docs/data_card.md` for where the actual data comes from.
