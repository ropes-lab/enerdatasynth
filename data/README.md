# `data/`

Everything the package reads from or writes into at runtime, plus the frozen research record. **This repository contains no real data.** Where the data comes from is listed below; you download it yourself under those sources' own terms. Every category is resource-nested, with one `pjm_load/` and one `wind/` subfolder each. See [`docs/architecture.md`](../docs/architecture.md) for why, and what adding a third resource here would look like.

## Data sources

| Data | Where to get it | Used for |
|---|---|---|
| PJM hourly metered load (`hrl_load_metered`, 2016–2025) | [PJM Data Miner 2](https://dataminer2.pjm.com/) | PJM load training, evaluation, calibration |
| Hourly temperature per PJM zone | NOAA Integrated Surface Database, fetched per weather station by `python -m enerdatasynth.resources.pjm_load.data.temperature` | PJM load conditioning |
| Wind speed and direction at 100 m, 5-minute | Not confirmed. The dependencies point to an NREL WIND Toolkit (HSDS) source; see [`docs/models/wind.md`](../docs/models/wind.md) §2 | Wind (upstream training only) |

With your own copy of the PJM and NOAA files in `raw/pjm_load/`, `enerdatasynth preprocess` rebuilds everything the training, evaluation and calibration commands need. Generating from the shipped checkpoints needs none of it.

## What is in this folder

| Folder | What's in it | README |
|---|---|---|
| `raw/{pjm_load,wind}/` | Where you place the source files above to run `enerdatasynth preprocess` yourself. Empty, and ignored by git | [`raw/README.md`](raw/README.md) |
| `cache/{pjm_load,wind}/` | Regenerable NOAA temperature fetch cache. Empty, and ignored by git | [`cache/README.md`](cache/README.md) |
| `processed/{pjm_load,wind}/` | Built by `enerdatasynth preprocess` from your own data and ignored by git. Only `pjm_load/metadata.json` (zone PS) and `pjm_load/zones/<ZONE>/metadata.json` (the other 16 zones) are tracked: the scalers the shipped checkpoints need to turn conditions into model inputs and outputs back into MW | [`processed/README.md`](processed/README.md) |
| `configs/{pjm_load,wind}/` | YAML mirrors of `RunConfig` hyperparameters (load only, since wind's config lives in code) | [`configs/README.md`](configs/README.md) |
| `checkpoints/{pjm_load,wind}/` | The trained models this package ships (weights and normalization statistics, not data) | [`checkpoints/README.md`](checkpoints/README.md) |
| `outputs/{pjm_load,wind}/` | Training logs and synthetic samples written by the CLI | [`outputs/README.md`](outputs/README.md) |
| `reports/{pjm_load,wind}/` | Current-package benchmark and evaluation reports, distinct from the historical ones | [`reports/README.md`](reports/README.md). See also `docs/results.md`, the canonical cross-checked read |
| `research_history/{pjm_load,wind}/` | Frozen research provenance: scripts, reports, trained weights. Not the product, and no data | [`research_history/README.md`](research_history/README.md) |

## What a developer should look at first

`checkpoints/` if you're wondering what the package ships and runs on. `research_history/` only if you're trying to understand why the package looks the way it does, not to run anything from it.

## What should not be placed here

Real data in any form: raw or cached PJM and NOAA files, and the arrays and tables built from them (`X_*.npy`, `Z_*.npy`, `*.parquet`). `.gitignore` blocks them. See "Licensing / Redistribution" in [`docs/data_card.md`](../docs/data_card.md).
