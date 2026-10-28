# `data/reports/pjm_load/`

Reports written by the load sweep and evaluation commands.

- `results_summary.md`: from `enerdatasynth report`.
- `hyperparameter_results.csv`, `hyperparameter_summary.md`: from `enerdatasynth sweep` (the 7 bundled runs).
- `failure_analysis.md`: from `enerdatasynth evaluate`, for the shipped default (`transformer_T1000_H64_L2`, uncalibrated samples).
- `failure_analysis_legacy_mlp_T100_H256_L4.md`: the same analysis for the legacy MLP, kept from before the default changed.
- `benchmark_comparison.csv`: from `examples/pjm_load/run_benchmark.py`. AR, ARMA, ARIMA and SARIMA against the shipped Transformer.
- `checkpoint_comparison.csv`: from `examples/pjm_load/compare_checkpoints.py`. Legacy MLP vs. Transformer, uncalibrated and calibrated, mean and std over 5 generation seeds.
- `future_experiments.md`: a hand-written list of experiments to try next.
- `figures/`: validation plots (hourly distribution, first difference, autocorrelation) for the shipped default and, from the earlier sweep, the legacy MLP `T100_H256_L4`.

The two hyperparameter files describe the legacy MLP sweep. Everything else describes the shipped Transformer unless the filename says otherwise. `docs/results.md` is the cross-checked reference, and the [category README](../README.md) explains how the two relate. Wind's reports are in [`../wind/`](../wind/README.md).
