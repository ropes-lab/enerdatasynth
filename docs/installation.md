# Installation

## Requirements

- Python >= 3.10
- ~200MB disk for raw + processed data, more if you keep multiple checkpoints

## Editable install (development)

```bash
git clone <this-repo>
cd dfe_summer2026
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

Wherever you run it from: `data/`, `data/checkpoints/pjm_load/`, `data/outputs/pjm_load/`, `data/reports/pjm_load/`, and `data/configs/pjm_load/` are resolved relative to your current working directory (or `$ENERDATASYNTH_PROJECT_ROOT` if set) — **not** relative to wherever pip installed the code. Run `enerdatasynth ...` / `import enerdatasynth` from inside your project directory (e.g. this repo's root, which already has data and checkpoints committed), or set `ENERDATASYNTH_PROJECT_ROOT=/path/to/project` explicitly. See [`docs/packaging.md`](packaging.md) for why this matters for a real (non-editable) install specifically.

## GPU

`enerdatasynth.utils.config.DEVICE` resolves to `"cuda"` automatically
if `torch.cuda.is_available()`, otherwise `"cpu"`. No separate GPU
extra is needed — install a CUDA-enabled `torch` build yourself if you
want GPU training (`pip install -e .` will not do this for you; it
takes whatever `torch` build is already on your system's index).
