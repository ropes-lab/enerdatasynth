# Installation

## Requirements

- Python 3.10 to 3.14 (3.12 is a safe choice). Newer releases fail until PyTorch publishes builds for them. On Python 3.9 or older, pip silently installs the old 0.4.1 release.
- Disk for your own raw and processed data if you plan to train (the repository ships none), plus the checkpoints

## Using the package

```bash
python3 -m pip install enerdatasynth
```

Check first that `python3 --version` says 3.10 to 3.14. That is enough to generate: the wheel includes the trained load Transformer and the three wind presets (about 13 MB). Use it from any folder. To see it work, run `python3 -m enerdatasynth demo` (or `enerdatasynth demo` if pip's script folder is on your PATH).

## Editable install (development)

You need a clone only to train your own models, run the tests, or change the code.

```bash
git clone https://github.com/ropes-lab/enerdatasynth.git
cd enerdatasynth
pip install -e .
```

This installs the core dependencies (numpy, pandas, torch, scikit-learn,
statsmodels, holidays, matplotlib, requests, tqdm, pyarrow, pyyaml) and
the `enerdatasynth` console command.

## Optional extras

```bash
pip install -e ".[viz]"   # seaborn, for data/research_history/pjm_load/eda_exploration.py
pip install -e ".[dev]"   # pytest, pytest-cov, ruff, mypy, build, twine
pip install -e ".[viz,dev]"
```

## Verifying the install

```bash
enerdatasynth --help
python -c "import enerdatasynth; print(enerdatasynth.__version__)"
pytest tests/ -v
```

## Reproducing exact results

`requirements.txt` pins the exact dependency versions that produced
the checkpoints and reports currently committed to this repository.
Use it instead of the looser `pyproject.toml` ranges if you need
bit-for-bit reproduction:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Where does it look for data and checkpoints?

Wherever you run it from: `data/`, `data/checkpoints/pjm_load/`, `data/outputs/pjm_load/`, `data/reports/pjm_load/`, and `data/configs/pjm_load/` are resolved relative to your current working directory (or `$ENERDATASYNTH_PROJECT_ROOT` if set) — **not** relative to wherever pip installed the code. If your project folder has none of these, generation falls back to the checkpoints bundled in the installed package, so a plain `pip install` works anywhere. Training, preprocessing and reports write into your project folder, so run those from inside your project directory (e.g. a clone of this repo), or set `ENERDATASYNTH_PROJECT_ROOT=/path/to/project` explicitly. See [`docs/packaging.md`](packaging.md) for why this matters for a real (non-editable) install specifically.

## GPU

`enerdatasynth.utils.config.DEVICE` resolves to `"cuda"` automatically
if `torch.cuda.is_available()`, otherwise `"cpu"`. No separate GPU
extra is needed — install a CUDA-enabled `torch` build yourself if you
want GPU training (`pip install -e .` will not do this for you; it
takes whatever `torch` build is already on your system's index).
