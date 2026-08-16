# `data/cache/pjm_load/`

Empty by design, not distributed with this repository.

`src/enerdatasynth/resources/pjm_load/data/temperature.py` populates this folder itself, one `{station}_{year}.parquet` file at a time, the first time NOAA ISD temperature data for that station-year is needed. It exists so repeated preprocessing runs don't re-download the same station-year twice. Deleting everything here is always safe — it rebuilds automatically (given network access) the next time `enerdatasynth preprocess` runs.

Nothing here is specific to this project — it's a generic NOAA ISD cache keyed by station and year. See `docs/data_card.md` for where the actual data comes from.
