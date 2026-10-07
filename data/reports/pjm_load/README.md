# `data/reports/pjm_load/`

Reports from PJM load's own sweep and evaluation commands:

- `results_summary.md` — `enerdatasynth report`
- `hyperparameter_results.csv`, `hyperparameter_summary.md` — `enerdatasynth sweep` (the 7 bundled runs)
- `failure_analysis.md` — `enerdatasynth evaluate`, for the shipped default (`transformer_T1000_H64_L2`, uncalibrated samples)
- `failure_analysis_legacy_mlp_T100_H256_L4.md` — the same analysis for the legacy MLP, preserved from before the default changed
- `benchmark_comparison.csv` — `examples/run_benchmark.py`: AR/ARMA/ARIMA/SARIMA against the shipped Transformer
- `checkpoint_comparison.csv` — `examples/compare_checkpoints.py`: legacy MLP vs. Transformer (uncalibrated and calibrated), mean and std over 5 generation seeds
- `future_experiments.md` — hand-written list of next experiments
- `figures/` — validation plots (hourly distribution, first difference, autocorrelation) for the shipped default and, from the earlier sweep, the legacy MLP `T100_H256_L4`

`hyperparameter_results.csv` and `hyperparameter_summary.md` describe the legacy MLP sweep; the other files describe the shipped Transformer unless the name says otherwise. `docs/results.md` is the cross-checked reference; the [category README](../README.md) explains how the two relate. Wind's reports are in [`../wind/`](../wind/README.md).
