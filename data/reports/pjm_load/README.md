# `data/reports/pjm_load/`

Reports from PJM load's own sweep and evaluation commands:

- `results_summary.md` — `enerdatasynth report`
- `hyperparameter_results.csv`, `hyperparameter_summary.md` — `enerdatasynth sweep` (the 7 bundled runs)
- `failure_analysis.md` — `enerdatasynth evaluate`
- `benchmark_comparison.csv` — `examples/run_benchmark.py`
- `future_experiments.md` — hand-written list of next experiments
- `figures/` — validation plots for the best run (`T100_H256_L4`): hourly distribution, first difference, autocorrelation

These describe the legacy MLP sweep, not the Transformer. `docs/results.md` is the cross-checked reference; the [category README](../README.md) explains how the two relate. Wind's reports are in [`../wind/`](../wind/README.md).
