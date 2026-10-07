# `examples/`

Runnable demonstrations of the package's public API, in increasing order of scope. Each script's own header explains exactly what it does — this file is just the map.

| Script | What it demonstrates | Runtime |
|---|---|---|
| `exploratory_data_analysis.py` | Inspecting the processed dataset (`metadata.json`, mean daily shape) — a lightweight companion to the full historical EDA in `data/research_history/pjm_load/scripts/eda_exploration.py` | Seconds |
| `generate_samples.py` | Loading a trained checkpoint via `LoadDiffusionModel` and generating profiles — the minimal Tier-2 usage pattern | Seconds |
| `run_full_pipeline.py` | End-to-end walkthrough (preprocess, train one small model, generate, validate) in a couple of minutes, not the full sweep | Minutes |
| `run_all.py` | The real top-level orchestrator: preprocess, full hyperparameter sweep, generate for the best config, validation plots, failure analysis, results summary | Hours (full sweep) |
| `compare_checkpoints.py` | Multi-seed comparison of trained checkpoints on the validation set (the evidence for which one ships as the default); writes `data/reports/pjm_load/checkpoint_comparison.csv` | A few minutes |
| `plot_real_vs_generated.py` | The six-day real-vs-generated figure in `docs/assets/`, for any checkpoint, with a fixed seed and a rule-based choice of days | Under a minute |
| `run_benchmark.py` | Classical-vs-diffusion comparison (AR/ARMA/ARIMA/SARIMA vs. a trained checkpoint), built entirely from library code with no duplicated logic | Depends on data size |

One naming quirk found during an earlier audit: `run_benchmark.py`'s own module docstring calls itself `benchmarks/run_benchmark.py` and describes itself as the "move to benchmarks/" successor to the research-history benchmark scripts. There's no top-level `benchmarks/` directory in this repository, though — the file just lives here in `examples/`. It wasn't moved, since that would change every path any external doc or script references; it's flagged in `docs/release_audit.md` as a low-priority inconsistency to fix deliberately rather than as a drive-by change.

## What a researcher should look at first

`generate_samples.py` if you just want to see output. `run_full_pipeline.py` if you want to see the whole pipeline connect without waiting hours.

## Assumptions these make

All of these assume raw data is already present (`data/raw/pjm_load/`) or that preprocessing has already been run. None of them ship or download data themselves.
