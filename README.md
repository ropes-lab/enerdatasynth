<div align="center">

# ⚡ enerdatasynth

[![PyPI](https://img.shields.io/pypi/v/enerdatasynth.svg)](https://pypi.org/project/enerdatasynth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests: 230 passing](https://img.shields.io/badge/tests-230%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

**enerdatasynth generates synthetic energy-resource profiles — electric load and wind
— from trained conditional generative models, behind one shared Python interface.** You
describe the conditions (season, temperature, calendar type); it returns a realistic
profile. No ML expertise required to use it.

## Generators

### PJM Electric Load
Conditional diffusion model for 24-hour PJM load profiles (zone `PS`), conditioned on
month, day-of-week, holiday, and temperature.
→ [PJM Load model report](docs/models/pjm_load.md)

### Wind
Conditional diffusion model for 24-hour (5-minute resolution) wind profiles across
three channel presets (`1ch`/`2ch`/`3ch`), conditioned on calendar, temperature, and
wind regime.
→ [Wind model report](docs/models/wind.md)

Both share the same `model()` → `.set()` → `.sample()` interface. The underlying data,
conditioning, and architectures are resource-specific — the product is unified, the
science is not forced to match.

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
samples = wind.sample(n=10, seed=0)      # (10, 288, 2) ndarray, m/s + degrees — wind sampling is CPU-slow at large n, see docs/models/wind.md
```

`pip install enerdatasynth` installs the library only — it does not ship trained
checkpoints or data. The example above works as-is if you clone this repository and run
from its root (checkpoints are bundled); see [`docs/installation.md`](docs/installation.md)
for every install path and [`docs/quickstart.md`](docs/quickstart.md) for training your
own checkpoint from scratch. **Note:** the version on PyPI today (`0.4.1`) predates this
repository's wind support; install from source for wind.

## How It Works

```
data → conditioning → generative model → synthetic profile → evaluation
```

Raw data is cleaned and reduced to a conditioning vector (calendar + weather). A
conditional diffusion model, trained per resource, denoises pure noise into a
profile matching that conditioning. Generated profiles are compared against real ones
using a fixed metric suite, and — for load — a small post-hoc bias correction. Full
technical detail (architectures, sampling, conditioning schemas) lives in the
[architecture docs](docs/architecture.md) and the two model reports, not here.

## Model Reports

Each resource has a model report documenting its input data, representation,
conditioning, architecture, generative model, training, evaluation, quantitative
results, real-vs-generated examples, and known limitations:

- [**PJM Load Model Report**](docs/models/pjm_load.md)
- [**Wind Model Report**](docs/models/wind.md)

These reports are the primary scientific documentation for this project — start there
for anything beyond basic usage.

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

**No raw or source data is distributed with this package or repository** — it ships
trained checkpoints and the generation interface, not the datasets behind them.

| Resource | Source |
|---|---|
| PJM load | PJM hourly metered load (`hrl_load_metered`), via [PJM Data Miner 2](https://dataminer2.pjm.com/) |
| Temperature (PJM load conditioning) | NOAA Integrated Surface Database (ISD), fetched per zone's constituent weather stations |
| Wind | A separately-developed, previously-published research pipeline (`wind_diffusion`) — see its [model report](docs/models/wind.md) for what is and isn't known about its original data source |

PJM load's cleaning, conditioning, and reproduction steps are fully documented in its
[model report](docs/models/pjm_load.md) and [data card](docs/data_card.md); bring your
own copy of the source data to `data/raw_data/pjm_load_data/` to reproduce preprocessing
end-to-end (see the placeholder file there). Wind currently has **no raw data or
reproduction path in this repository at all** — its report says so directly. There is
no separate public data archive (e.g. Zenodo) for this project today.

## Citation

No accompanying paper has been published yet. If you use this software, please cite the
repository:

```bibtex
@software{enerdatasynth,
  title  = {enerdatasynth: A Research SDK for Synthetic PJM Energy Resource Profiles},
  author = {Sharma, Sonakshi},
  year   = {2026},
  note   = {Software}
}
```

## License

[MIT](LICENSE) © 2026 Sonakshi Sharma
