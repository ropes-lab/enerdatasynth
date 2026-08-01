# PJM Conditional Diffusion Baseline

A minimal, reproducible conditional diffusion pipeline for generating
synthetic daily PJM electricity load profiles, built exactly to the
PI's specification: simple architecture, fixed conditioning set,
standard diffusion loss, in-sample validation against real data.

## What this is NOT

Per the PI's instruction, this baseline explicitly excludes:
MW anchor, Fourier modes, custom losses, higher moments, user-facing
abstractions, a unified PJM-wide model, attention, and transformers.
These are documented only as future work in `reports/future_experiments.md`.

## Quick Start

```bash
# Editable install (pulls in numpy, pandas, torch, scikit-learn,
# statsmodels, holidays, matplotlib, requests, tqdm, pyarrow)
pip install -e .

# Optional extras: seaborn for research_history/eda_exploration.py,
# or dev tooling (pytest, ruff, mypy, build, twine)
pip install -e ".[viz,dev]"

# Run everything: preprocess -> sweep -> generate -> validate -> report
powerdiffusion pipeline

# Or run stages individually via the CLI:
powerdiffusion preprocess
powerdiffusion sweep
powerdiffusion generate --diffusion-steps 100 --hidden-dim 256 --n-layers 4
powerdiffusion evaluate --run-id T100_H256_L4
powerdiffusion report

# Or equivalently, call the library functions directly:
python -m powerdiffusion.data.preprocess
python -m powerdiffusion.training.sweep
python -m powerdiffusion.sampling.generate
```

## Folder Structure

```
src/powerdiffusion/
├── data/
│   ├── preprocess.py             full cleaning + tensor-building pipeline
│   └── temperature.py            NOAA hourly temperature builder
├── models/denoiser.py            simple MLP denoiser
├── diffusion/
│   ├── ddpm.py                   DDPM forward/reverse process (corrected ancestral sampler)
│   └── schedules.py              linear/cosine/quadratic beta schedules
├── training/
│   ├── train.py                  single training run, checkpointing, early stop
│   └── sweep.py                  hyperparameter sweep runner
├── sampling/generate.py          in-sample synthetic dataset generation
├── evaluation/
│   ├── metrics.py                canonical 7-metric real-vs-generated comparison
│   ├── baselines.py              classical AR/ARMA/ARIMA/SARIMA baselines
│   ├── failure_analysis.py       automatic failure mode detection
│   └── results_report.py         auto-generates results_summary.md
├── visualization/plots.py        all plotting (validation plots + training loss curve)
├── utils/config.py               all paths, conditioning spec, sweep grid
└── cli.py                        `powerdiffusion` console-script entry point

data/
├── raw/                          raw CSVs (load + temperature)
├── processed/                    X, Z tensors + metadata + README (working copy)
└── processed_backup/             FROZEN duplicate — restore from here if corrupted

checkpoints/<run_id>/             best.pt, latest.pt, loss_curve.png, result.json
outputs/
├── generated/                    X_generated_<run_id>.npy
└── logs/                         per-epoch CSV training logs

reports/
├── figures/                      validation plots per run
├── results_summary.md            auto-generated
├── hyperparameter_results.csv
├── hyperparameter_summary.md
├── failure_analysis.md
└── future_experiments.md         NOT implemented — recommendations only

scripts/run_all.py                end-to-end orchestrator (same stages as `powerdiffusion pipeline`)
examples/                         small demonstration scripts
research_history/                 frozen historical research scripts (weekN.py lineage) — see Phase 1 audit
```

## Conditioning Vector (23-dim, fixed per PI spec)

| Variable | Encoding | Dim |
|----------|----------|-----|
| month | one-hot | 12 |
| day of week | one-hot | 7 |
| holiday | binary | 1 |
| Tmax | z-score | 1 |
| Tmin | z-score | 1 |
| spread = max(load) − min(load) | z-score | 1 |

## Restoring from Backup

If `data/processed/` ever gets corrupted by a bad re-run:

```bash
rm -rf data/processed
cp -r data/processed_backup data/processed
```

The backup is written once, immediately after preprocessing completes,
and is never touched again by any other script.
