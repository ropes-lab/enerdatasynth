<div align="center">

# ⚡ enerdatasynth

[![PyPI](https://img.shields.io/pypi/v/enerdatasynth.svg)](https://pypi.org/project/enerdatasynth/)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10%20to%203.14-blue.svg)](pyproject.toml)
[![Tests: 272 passing](https://img.shields.io/badge/tests-272%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyPI: <https://pypi.org/project/enerdatasynth/>**

</div>

enerdatasynth generates synthetic energy profiles for electric load and wind. Trained conditional models sit behind one Python interface: you describe the conditions you want (season, temperature, day type) and get a profile back.

## Generators

Two resources share one interface, `enerdatasynth.model(<name>, zone=...)`.

**PJM electric load.** A conditional diffusion model for 24-hour load profiles in the PJM `PS` zone, conditioned on month, day of week, holiday and temperature. Report: [docs/models/pjm_load.md](docs/models/pjm_load.md).

**Wind.** A conditional diffusion model for 24-hour wind profiles at 5-minute resolution, with three channel presets (`1ch`, `2ch`, `3ch`), conditioned on calendar, temperature and wind regime. Report: [docs/models/wind.md](docs/models/wind.md).

## Quick start

**1. Check your Python.** In a terminal, run `python3 --version`. It must say 3.10, 3.11, 3.12, 3.13 or 3.14. If it says 3.9 or lower (the default on many Macs), install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) and open a new terminal. Without this, pip installs an old 0.4.1 release that doesn't work as described here. Don't pick a brand-new release such as 3.15; PyTorch doesn't publish builds for it yet and the install fails.

**2. Install.**

```bash
python3 -m pip install --upgrade pip
python3 -m pip install enerdatasynth
```

We write `python3 -m pip` because many machines have no plain `pip` command. The first line is there because the pip that comes with a fresh Python is often too old to install PyTorch (you'd see "conflicting dependencies" and a `torch` error).

**3. Use it.** Type `python3` and press Enter. The prompt changes to `>>>`. Then type these lines, pressing Enter after each:

```python
import enerdatasynth
enerdatasynth.load(month=7, Tmax=90, Tmin=74)
enerdatasynth.wind(month=7, wind_regime="high")
```

`load` prints one generated day of PJM load, hour by hour, in MW. `wind` prints one generated day of wind speed in m/s. Call either one on its own with the conditions you want. The models ship inside the package, so this works from any folder. The first call takes about a minute while torch loads. Type `exit()` to leave Python.

Conditions for `load`: `month` (1 to 12), `Tmax` and `Tmin` in degrees Fahrenheit, and optionally `weekday=True/False`, `holiday=True/False` or `day_of_week` (0 is Monday). Conditions for `wind`: `preset` (`"1ch"` for speed only, `"2ch"` or `"3ch"` for speed and direction), `wind_regime` (`"low"`, `"mid"` or `"high"`), `month`, and `temp_min` and `temp_max` in degrees Celsius. Both return the numbers as a numpy array, and the same `seed=` gives the same day.

Type the lines one at a time. Pasting several lines at once into `>>>` can run them together and give a `SyntaxError` on some Python versions.

## More control

For many samples at once, use the fuller interface. Save this as `try_it.py` and run `python3 try_it.py`:

```python
import enerdatasynth

load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True)
samples = load.sample(n=100, seed=0)
print("load:", samples.shape)            # (100, 24), MW

wind = enerdatasynth.model("wind", zone="1ch")
wind.set(wind_regime="high", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=10, seed=0)
print("wind:", samples.shape)            # (10, 288, 2), m/s and degrees
```

If you end up with version `0.4.1`, your `python3` is older than 3.10. If the install fails on a `torch` error, your Python is newer than 3.14. Wind sampling is slow on CPU for larger `n`; see [docs/models/wind.md](docs/models/wind.md). To train your own checkpoints you need a clone of this repository and your own copy of the data; see [docs/installation.md](docs/installation.md) and [docs/quickstart.md](docs/quickstart.md).

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
