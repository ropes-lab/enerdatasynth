# PJM Conditional Diffusion

A reproducible conditional diffusion pipeline for generating synthetic
daily PJM electricity load profiles: a transformer denoiser (the
architecture research found best), a fixed conditioning set, a
standard diffusion loss, scalar bias calibration, and in-sample
validation against real data.

## Architecture

The default and recommended architecture is `TransformerDenoiser` —
promoted from an architecture comparison across MLP/CNN/attention/
transformer backbones (`research_history/reports/week6/arch/`), then
trained to convergence and re-validated head-to-head against the
original MLP baseline and four classical time-series models
(`research_history/reports/week7/summary_report.md`):

- Beats the MLP baseline on 6 of 7 generation metrics at ~6x fewer
  parameters (110K vs 688K).
- Beats every classical baseline (AR/ARMA/ARIMA/SARIMA) on 7 of 7.

The original simple feedforward MLP denoiser remains available as
`architecture="mlp"` (`powerdiffusion train --architecture mlp`) — a
legacy/research baseline, not the default.

A scalar per-hour bias correction ("calibration") is applied at
generation time by default: the denoiser reproduces daily *shape*
well but carries a small systematic level offset that compounds into
a large daily-energy error when summed over 24 hours. Correcting it
cuts energy error by ~66% at zero cost to shape metrics — see
`research_history/reports/week7/calibration_report.md`. Run
`powerdiffusion calibrate` after training to fit it for a given
checkpoint.

**Known gap, not silently resolved:** the transformer/calibration
results above were validated in `research_history/` using a
6-dimensional raw conditioning vector (month/day-of-week as integers).
This package's own preprocessing (`powerdiffusion.data.preprocess`)
produces a different, 23-dimensional one-hot conditioning vector — a
deliberate, documented choice, but one that has not itself been
re-validated against these transformer/calibration results. No
pretrained checkpoint compatible with the current preprocessing exists
yet; train one with `powerdiffusion train`. See `MIGRATION.md` for
the full detail.

This baseline still excludes MW anchor, Fourier modes, custom losses,
higher moments, and a unified multi-zone model — documented as future
work in `reports/future_experiments.md`.

## Quick Start

```bash
# Editable install (pulls in numpy, pandas, torch, scikit-learn,
# statsmodels, holidays, matplotlib, requests, tqdm, pyarrow)
pip install -e .

# Optional extras: seaborn for research_history/eda_exploration.py,
# or dev tooling (pytest, ruff, mypy, build, twine)
pip install -e ".[viz,dev]"

# Run everything: preprocess -> sweep -> generate -> calibrate -> validate -> report
powerdiffusion pipeline

# Or run stages individually via the CLI (architecture defaults to "transformer"):
powerdiffusion preprocess
powerdiffusion sweep
powerdiffusion generate --diffusion-steps 1000 --hidden-dim 64 --n-layers 2
powerdiffusion calibrate --run-id transformer_T1000_H64_L2
powerdiffusion evaluate --run-id transformer_T1000_H64_L2
powerdiffusion report
```

```python
# Or use the public API directly:
from powerdiffusion import LoadDiffusionModel

model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")
samples = model.sample(z, seed=0)   # calibration applied automatically if fitted
```

## Folder Structure

```
src/powerdiffusion/
├── api.py                        LoadDiffusionModel — the public, high-level interface
├── data/
│   ├── preprocess.py             full cleaning + tensor-building pipeline
│   └── temperature.py            NOAA hourly temperature builder
├── models/
│   ├── denoiser.py               simple MLP denoiser (architecture="mlp", legacy baseline)
│   ├── transformer.py            transformer denoiser (architecture="transformer", default)
│   └── __init__.py               build_denoiser() factory — selects architecture by name
├── diffusion/
│   ├── ddpm.py                   DDPM forward/reverse process (corrected ancestral sampler)
│   └── schedules.py              linear/cosine/quadratic beta schedules (linear validated best)
├── training/
│   ├── train.py                  single training run, checkpointing, early stop
│   └── sweep.py                  hyperparameter sweep runner
├── sampling/generate.py          in-sample synthetic dataset generation
├── evaluation/
│   ├── metrics.py                canonical 7-metric real-vs-generated comparison
│   ├── baselines.py              classical AR/ARMA/ARIMA/SARIMA baselines
│   ├── calibration.py            scalar bias correction, applied at generation time
│   ├── failure_analysis.py       automatic failure mode detection
│   └── results_report.py         auto-generates results_summary.md
├── visualization/plots.py        all plotting (validation plots + training loss curve)
├── utils/config.py               all paths, conditioning spec, architecture + sweep defaults
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
research_history/                 frozen historical research scripts, kept for provenance and reproducibility
```

## Conditioning Vector (23 dimensions)

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
