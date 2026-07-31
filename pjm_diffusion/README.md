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
pip install torch numpy pandas scikit-learn matplotlib scipy statsmodels holidays pyarrow tzdata

# Run everything: preprocess -> sweep -> generate -> validate -> report
python run_all.py

# Or run stages individually:
python -m src.data.preprocess          # build X, Z tensors
python -m src.training.sweep           # hyperparameter sweep
python -m src.training.generate        # generate synthetic dataset
python -m src.validation.metrics       # 3 validation plots
python -m src.validation.failure_analysis
python -m src.validation.results_report
```

## Folder Structure

```
pjm_diffusion/
├── src/
│   ├── data/preprocess.py       full cleaning + tensor-building pipeline
│   ├── models/
│   │   ├── denoiser.py           simple MLP denoiser
│   │   └── diffusion.py          DDPM forward/reverse process
│   ├── training/
│   │   ├── train.py              single training run, checkpointing, early stop
│   │   ├── sweep.py              hyperparameter sweep runner
│   │   └── generate.py           in-sample synthetic dataset generation
│   ├── validation/
│   │   ├── metrics.py            3 validation plots (hourly/diff/ACF)
│   │   ├── failure_analysis.py   automatic failure mode detection
│   │   └── results_report.py     auto-generates results_summary.md
│   └── utils/config.py           all paths, conditioning spec, sweep grid
├── data/
│   ├── raw/                      symlinked source CSVs
│   ├── processed/                X, Z tensors + metadata + README (working copy)
│   └── processed_backup/         FROZEN duplicate — restore from here if corrupted
├── outputs/
│   ├── checkpoints/<run_id>/     best.pt, latest.pt, loss_curve.png, result.json
│   ├── generated/                X_generated_<run_id>.npy
│   └── logs/                     per-epoch CSV training logs
├── reports/
│   ├── figures/                  validation plots per run
│   ├── results_summary.md        auto-generated
│   ├── hyperparameter_results.csv
│   ├── hyperparameter_summary.md
│   ├── failure_analysis.md
│   └── future_experiments.md     NOT implemented — recommendations only
└── run_all.py                    end-to-end orchestrator
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
