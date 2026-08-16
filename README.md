<div align="center">

# ⚡ enerdatasynth

[![PyPI](https://img.shields.io/pypi/v/enerdatasynth.svg)](https://pypi.org/project/enerdatasynth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests: 252 passing](https://img.shields.io/badge/tests-252%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

**enerdatasynth generates synthetic energy-resource profiles for electric load and wind, using trained conditional generative models behind one Python interface.** Describe the conditions you want (season, temperature, calendar type) and it returns a realistic profile.

## Generators

Two resources share one interface. `enerdatasynth.model(<name>, zone=...)` works the same way regardless of what's actually running underneath.

### PJM Electric Load
Conditional diffusion model for 24-hour PJM load profiles (zone `PS`), conditioned on month, day-of-week, holiday, and temperature.
→ [PJM Load model report](docs/models/pjm_load.md)

### Wind
Conditional diffusion model for 24-hour wind profiles at 5-minute resolution, across three channel presets (`1ch`/`2ch`/`3ch`), conditioned on calendar, temperature, and wind regime.
→ [Wind model report](docs/models/wind.md)

## Quick Start

```bash
pip install enerdatasynth
```

```python
import enerdatasynth

load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True)
samples = load.sample(n=100, seed=0)     # (100, 24) ndarray, MW

wind = enerdatasynth.model("wind", zone="1ch")
wind.set(wind_regime="mid", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=10, seed=0)      # (10, 288, 2) ndarray, m/s + degrees
# wind sampling is slow on CPU at larger n — see docs/models/wind.md
```

`pip install enerdatasynth` gives you the library only, not trained checkpoints or data. The snippet above runs as-is if you clone this repo and work from its root, since the checkpoints are bundled there. See [`docs/installation.md`](docs/installation.md) for other install paths and [`docs/quickstart.md`](docs/quickstart.md) if you want to train your own checkpoint from scratch. PyPI is up to date as of `0.7.0`, which has full wind support, so a plain `pip install` gets you both resources' code — you'll still need `data/checkpoints/` from this repo to actually generate anything.

## How It Works

```
data → conditioning → generative model → synthetic profile → evaluation
```

Raw data gets cleaned and reduced to a conditioning vector — calendar plus weather. A conditional diffusion model denoises pure noise into a profile matching that conditioning. Load and wind each have their own pipeline here, built independently, because the two are different enough scientifically that sharing one wouldn't make sense. Generated profiles get checked against real ones with a metric suite (again, a different suite per resource), plus a small bias correction for load. The interface stays the same across both; what happens underneath doesn't. Architecture, sampling, and conditioning details all live in the [architecture docs](docs/architecture.md) and the two model reports below.

## Model Reports

Each resource has a full report covering input data, representation, conditioning, architecture, training, evaluation, results, real-vs-generated comparisons, and known limitations:

- [**PJM Load Model Report**](docs/models/pjm_load.md)
- [**Wind Model Report**](docs/models/wind.md)

Start there if you want more than basic usage — these are the primary scientific documentation for the project.

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | [Quick Start](docs/quickstart.md) |
| [API Reference](docs/api_reference.md) | [Architecture](docs/architecture.md) |
| [Evaluation](docs/evaluation.md) | [Extending (add a resource)](docs/extending.md) |
| [PJM Load Model Report](docs/models/pjm_load.md) | [Wind Model Report](docs/models/wind.md) |
| [Model Card](MODEL_CARD.md) | [Data Card](docs/data_card.md) |
| [Developer Guide](docs/developer_guide.md) | [FAQ](docs/faq.md) |

Full documentation map: [`docs/README.md`](docs/README.md).

## Data & Reproducibility

No raw or source data ships with this package or repository. What you get is trained checkpoints and the generation interface, not the underlying datasets.

| Resource | Source |
|---|---|
| PJM load | PJM hourly metered load (`hrl_load_metered`), via [PJM Data Miner 2](https://dataminer2.pjm.com/) |
| Temperature (PJM load conditioning) | NOAA Integrated Surface Database (ISD), fetched per zone's constituent weather stations |
| Wind | A separately-developed research pipeline (`wind_diffusion`) — see its [model report](docs/models/wind.md) for what's known (and not known) about its original data source |

PJM load's cleaning and conditioning steps are documented in its [model report](docs/models/pjm_load.md) and [data card](docs/data_card.md); bring your own copy of the source data to `data/raw/pjm_load/` if you want to reproduce preprocessing end to end (there's a placeholder file there explaining what's needed). Wind doesn't have a reproduction path in this repository at all right now — no raw data exists here for it, and its report says so directly. There's no separate public data archive for this project either.

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
