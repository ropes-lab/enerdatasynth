# `data/raw/pjm_load/`

Empty by default. This is the folder `enerdatasynth preprocess` actually reads from — `RAW_DIR` in `src/enerdatasynth/utils/config.py`, consumed by `src/enerdatasynth/resources/pjm_load/data/preprocess.py`'s `load_raw_load_data`/`load_temperature_data`.

This repository doesn't distribute raw data. To reproduce preprocessing from scratch, place these files directly in this folder (not a subfolder):

| File(s) | What it is | Source |
|---|---|---|
| `hrl_load_metered_2016.csv` … `hrl_load_metered_2025.csv` | PJM hourly metered load | [PJM Data Miner 2](https://dataminer2.pjm.com/) |
| `pjm_hourly_temps_wide.csv` | Hourly temperature per PJM zone, built from NOAA ISD | Run `python -m enerdatasynth.resources.pjm_load.data.temperature` (given network access) to fetch and build this file automatically, instead of sourcing it by hand |

See [`docs/data_card.md`](../../../docs/data_card.md) and [`docs/models/pjm_load.md`](../../../docs/models/pjm_load.md) for what preprocessing does to these files, and why they aren't bundled here.

## What should not be placed here

Anything other than the exact files above. `preprocess.py` reads this folder by a fixed naming convention, not a directory scan.
