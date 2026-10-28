<div align="center">

# ⚡ enerdatasynth

[![PyPI](https://img.shields.io/pypi/v/enerdatasynth.svg)](https://pypi.org/project/enerdatasynth/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Tests: 283 passing](https://img.shields.io/badge/tests-283%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyPI: <https://pypi.org/project/enerdatasynth/>**

</div>

enerdatasynth generates synthetic energy profiles for electric load and wind. Trained conditional models sit behind one Python interface: you describe the conditions you want (season, temperature, day type) and get a profile back.

## Generators

Two resources share one interface, `enerdatasynth.model(<name>, zone=...)`.

**PJM electric load.** A conditional diffusion model for 24-hour load profiles in the PJM `PS` zone, conditioned on month, day of week, holiday and temperature. Report: [docs/models/pjm_load.md](docs/models/pjm_load.md).

**Wind.** A conditional diffusion model for 24-hour wind profiles at 5-minute resolution, with three channel presets (`1ch`, `2ch`, `3ch`), conditioned on calendar, temperature and wind regime. Report: [docs/models/wind.md](docs/models/wind.md).

## Quick start

```bash
pip install enerdatasynth
```

Then copy a whole block below into your terminal and press Enter. Each one imports the package, sets conditions, generates a day and prints its 24-hour profile.

**PJM electric load** (temperatures in °F):

```bash
python3 - <<'EOF'
import enerdatasynth
load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True)
samples = load.sample(n=1, seed=0)
enerdatasynth.show(samples)
EOF
```

**Wind** (temperatures in °C; `zone` is the preset: `"1ch"` speed only, `"2ch"` or `"3ch"` speed and direction):

```bash
python3 - <<'EOF'
import enerdatasynth
wind = enerdatasynth.model("wind", zone="1ch")
wind.set(wind_regime="high", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=1, seed=0)
enerdatasynth.show(samples)
EOF
```

You get the 24-hour profile: one row per hour, in MW for load and m/s for wind (wind is the average of each hour's twelve 5-minute values; `"2ch"` and `"3ch"` also show direction). Change the numbers in `.set(...)` to ask for a different day. Use `n=` to generate more days; `show` prints the first, and `samples` holds them all as a plain numpy array (`(n, 24)` for load, `(n, 288, 2)` for wind). The trained models ship inside the package, so this works from any folder. The first run takes about a minute while torch loads.

It works on Python 3.9 and newer. If `pip install` gives you version `0.4.1`, your Python is older than 3.9. Wind sampling is slow on CPU for larger `n`; see [docs/models/wind.md](docs/models/wind.md). Typing the same lines inside `python3` works too. To train your own checkpoints you need a clone of this repository and your own copy of the data; see [docs/installation.md](docs/installation.md) and [docs/quickstart.md](docs/quickstart.md).

## How it works

```
data -> conditioning -> generative model -> synthetic profile -> evaluation
```

Source data is cleaned and reduced to a conditioning vector (calendar plus weather). A conditional diffusion model turns pure noise into a profile that matches that vector. Load and wind each have their own pipeline, because the two problems are different enough that forcing them into one would not help. Generated profiles are scored against real ones with a metric suite (a different one per resource), and load gets a small bias correction on top. The interface is the same for both; what runs underneath is not. Details are in the [architecture docs](docs/architecture.md) and the two model reports.

## Model reports

Each resource has a full report covering the input data, conditioning, architecture, training, evaluation, results and known limitations. They are the main scientific documentation for the project, so start there if you want more than basic usage:

- [PJM load model report](docs/models/pjm_load.md)
- [Wind model report](docs/models/wind.md)

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | [Quick start](docs/quickstart.md) |
| [API reference](docs/api_reference.md) | [Architecture](docs/architecture.md) |
| [Evaluation](docs/evaluation.md) | [Extending (add a resource)](docs/extending.md) |
| [PJM load model report](docs/models/pjm_load.md) | [Wind model report](docs/models/wind.md) |
| [Model card](MODEL_CARD.md) | [Data card](docs/data_card.md) |
| [Developer guide](docs/developer_guide.md) | [FAQ](docs/faq.md) |

The full map is in [docs/README.md](docs/README.md).

## Data and reproducibility

This repository contains no real data, raw or processed. You get trained checkpoints and the generation interface, not the datasets behind them.

| Resource | Where the data comes from |
|---|---|
| PJM load | PJM hourly metered load (`hrl_load_metered`) from [PJM Data Miner 2](https://dataminer2.pjm.com/) |
| Temperature (for load conditioning) | NOAA Integrated Surface Database, fetched for each zone's weather stations |
| Wind | A separate research pipeline (`wind_diffusion`). Its [model report](docs/models/wind.md) says what is and isn't known about the original data source |

To reproduce load preprocessing end to end, download the PJM and NOAA files yourself, put them in `data/raw/pjm_load/` (the README there explains what is needed), and run `enerdatasynth preprocess`. Wind has no reproduction path here: the raw wind data was never part of this repository, and the wind report says so.

## Citation

No paper yet. If you use this software, please cite the repository:

```bibtex
@software{enerdatasynth,
  title  = {enerdatasynth: A Research SDK for Synthetic PJM Energy Resource Profiles},
  author = {Sharma, Sonakshi and Tikkawar, Pranav and Mieth, Robert},
  year   = {2026},
  note   = {Software}
}
```

## License

[MIT](LICENSE) © 2026 Sonakshi Sharma
