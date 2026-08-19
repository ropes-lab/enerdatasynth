# `data/raw/wind/`

Empty, and will stay that way for this package. There is no raw-data acquisition step for wind anywhere in this repository — unlike [`../pjm_load/`](../pjm_load/README.md), which `enerdatasynth preprocess` reads directly.

What's known about the original source, from the vendored code and its declared-but-unused dependencies:

| Signal | What it suggests |
|---|---|
| `h5py`, `h5pyd`, `python-dotenv` in `pyproject.toml` | An HSDS/NREL WIND Toolkit-style original data source — `h5pyd` specifically is the client library for NREL's HSDS API, which needs an API key (hence `python-dotenv`) |
| Zero actual imports of any of the three, anywhere in `src/enerdatasynth/resources/wind/` | The acquisition step was never vendored in — these are leftover dependencies from the upstream project, not code this package runs |
| `resources/wind/manifest.json`'s `feature_cols` | `windspeed_100m`, `winddirection_100m` — wind speed and direction at 100m, 5-minute cadence, 288 steps/day |

The bundled checkpoints (`data/checkpoints/wind/`) were trained on this data elsewhere, in the separate upstream `wind_diffusion` project — this repository can generate from them but cannot reproduce the preprocessing that produced them, since neither the raw data nor the fetch code for it lives here. See [`docs/models/wind.md`](../../../docs/models/wind.md) §2 for the full accounting.

If raw wind data (and the HSDS fetch code to acquire it) is ever vendored into this repository, it goes here, following `../pjm_load/`'s convention: a flat folder of source files this package's own preprocessing reads by a fixed naming convention, not a directory scan.
