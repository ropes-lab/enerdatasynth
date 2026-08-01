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
the `pjm-diffusion` console command.

## Optional extras

```bash
pip install -e ".[viz]"   # seaborn, for research_history/eda_exploration.py
pip install -e ".[dev]"   # pytest, pytest-cov, ruff, mypy, build, twine
pip install -e ".[viz,dev]"
```

## Verifying the install

```bash
pjm-diffusion --help
python -c "import pjm_diffusion; print(pjm_diffusion.__version__)"
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

## GPU

`pjm_diffusion.utils.config.DEVICE` resolves to `"cuda"` automatically
if `torch.cuda.is_available()`, otherwise `"cpu"`. No separate GPU
extra is needed — install a CUDA-enabled `torch` build yourself if you
want GPU training (`pip install -e .` will not do this for you; it
takes whatever `torch` build is already on your system's index).
