# `examples/`

Runnable demos of the public API, from smallest to largest. Each script's header explains what it does; this file is the map.

| Script | What it shows | Runtime |
|---|---|---|
| `exploratory_data_analysis.py` | Inspects the processed dataset (`metadata.json`, mean daily shape). Needs your own `enerdatasynth preprocess` output. A light companion to the full historical EDA in `data/research_history/pjm_load/scripts/eda_exploration.py` | Seconds |
| `generate_samples.py` | Loads a trained checkpoint through `LoadDiffusionModel` and generates profiles, the smallest Tier 2 example | Seconds |
| `run_full_pipeline.py` | A quick end-to-end run: preprocess, train one small model, generate, validate. Not the full sweep | Minutes |
| `run_all.py` | The full orchestrator: preprocess, hyperparameter sweep, generate for the best config, validation plots, failure analysis, results summary | Hours (full sweep) |
| `compare_checkpoints.py` | Compares trained checkpoints over several seeds on the validation set (the evidence for which one ships as the default). Writes `data/reports/pjm_load/checkpoint_comparison.csv` | A few minutes |
| `plot_real_vs_generated.py` | The six-day real vs. generated figure in `docs/assets/`, for any checkpoint, with a fixed seed and rule-based day selection | Under a minute |
| `run_benchmark.py` | Classical vs. diffusion comparison (AR, ARMA, ARIMA, SARIMA vs. a trained checkpoint), built only from library code | Depends on data size |

One oddity: the docstring in `run_benchmark.py` calls the file `benchmarks/run_benchmark.py` and describes it as the successor to the research-history benchmark scripts. There is no top-level `benchmarks/` folder; the file lives here. It wasn't moved because that would change every path other docs and scripts point to. `docs/release_audit.md` lists it as a low-priority inconsistency to fix on purpose, not in passing.

## Where to start

`generate_samples.py` if you just want to see output, `run_full_pipeline.py` if you want to watch the whole pipeline connect without waiting hours.

## Assumptions

All of them assume `enerdatasynth preprocess` has already run on your own copy of the data (raw files in `data/raw/pjm_load/`); even `generate_samples.py` reads the validation condition vectors from `data/processed/pjm_load/`. None of these scripts ship or download data. This repository has none, so get it from the sources in [`data/README.md`](../data/README.md#data-sources).
