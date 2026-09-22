# `data/raw/wind/`

Empty. This repository has no step for downloading raw wind data, unlike
[`../pjm_load/`](../pjm_load/README.md), which `enerdatasynth preprocess` reads directly.

What we can tell about the original data source:

| Clue | What it suggests |
|---|---|
| `h5py`, `h5pyd` and `python-dotenv` are listed in `pyproject.toml` | Data from an HSDS service such as the NREL WIND Toolkit. `h5pyd` is the client for NREL's HSDS API, which needs an API key (hence `python-dotenv`) |
| None of the three are imported anywhere in `src/enerdatasynth/resources/wind/` | The download code was never vendored; these dependencies were carried over from the upstream project |
| `feature_cols` in `resources/wind/manifest.json` | `windspeed_100m` and `winddirection_100m`: speed and direction at 100 m, every 5 minutes, 288 steps per day |

The bundled checkpoints (`data/checkpoints/wind/`) were trained on this data in the
upstream `wind_diffusion` project. This repository can generate from them, but it can't
reproduce the preprocessing because it has neither the data nor the download code
([`docs/models/wind.md`](../../../docs/models/wind.md) §2).

If raw wind data is ever added, put it here following `../pjm_load/`'s convention: a flat
folder of source files that preprocessing reads by fixed file names.
