# `enerdatasynth.resources.pjm_load.evaluation`

Scoring and calibration for PJM load, and nothing else. Wind uses different code for its fidelity metrics (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy` and so on, in [`resources/wind/metrics.py`](../../wind/README.md)) and has no calibration step. [`docs/models/wind.md`](../../../../../docs/models/wind.md) §7 and §12 explain why.

## Files

- **`metrics.py`**: `compute_metrics(real, generated)`, the one implementation of the 7-metric suite (Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp). It replaced four slightly different versions scattered through `data/research_history/pjm_load/scripts/`. The math is written out in `docs/research_status.md` §G.
- **`calibration.py`**: `compute_bias`, `calibrate_from_samples`, `apply_calibration`, `calibrate_checkpoint`. This is the scalar bias correction from Week 7, fit on training days only, never on validation. `docs/research_status.md` §F explains why it is a scalar and not per-hour.
- **`baselines.py`**: classical baselines (AR, ARMA, ARIMA, SARIMA via `statsmodels`) used as the lower bound in `docs/results.md` §3. It needs the `research` extra (`pip install enerdatasynth[research]`) and isn't on the `model().sample()` path.
- **`failure_analysis.py`**: `analyze(run_id)` and `write_failure_analysis_md(result)`. Detects failure modes (ramp-rate mismatch, seasonal mismatch) for a trained run and writes a markdown report. Behind `enerdatasynth evaluate`.
- **`results_report.py`**: `generate_results_summary()`, which rebuilds `data/reports/pjm_load/results_summary.md`. Behind `enerdatasynth report`.

`failure_analysis.py`, `calibration.py` and the comparison scripts read the processed arrays, which the repository doesn't include. Build them first with `enerdatasynth preprocess` on your own copy of the data.

## Public API

`compute_metrics` and `calibrate_checkpoint` are re-exported at the top level. `baselines`, `failure_analysis` and `results_report` exist to support the CLI; import the submodule directly if you need one elsewhere.

## Be careful with

The formulas in `metrics.py`. Every number in `docs/results.md` and every report under `data/research_history/pjm_load/reports/` was computed with this math, which the module docstring says is copied verbatim from the identical `metrics()`/`_acf()` pair in `week6.py`, `week7.py` and `arch_compare.py`. Change a formula and old numbers stop being comparable to new ones.
