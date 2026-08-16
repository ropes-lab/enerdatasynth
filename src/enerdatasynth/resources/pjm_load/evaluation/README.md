# `enerdatasynth.resources.pjm_load.evaluation`

PJM load's scoring and calibration tooling specifically. Wind's fidelity-gate metrics (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, and so on) are separate code in [`src/enerdatasynth/resources/wind/metrics.py`](../../wind/README.md), and wind has no calibration step at all — see [`docs/models/wind.md`](../../../../../docs/models/wind.md) §7/§12 for why both differ. Everything below is load-specific.

This is everything involved in scoring a trained load checkpoint's output and correcting it.

## Files

- **`metrics.py`** — `compute_metrics(real, generated)`, the single canonical implementation of the 7-metric suite (Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp), replacing four inconsistent versions that used to be scattered across `data/research_history/pjm_load/scripts/`. Exact math is in `docs/research_status.md` §G.
- **`calibration.py`** — `compute_bias`, `calibrate_from_samples`, `apply_calibration`, `calibrate_checkpoint`. The scalar post-hoc bias correction validated in Week 7, fit on training days only, never validation. `docs/research_status.md` §F explains why scalar rather than per-hour.
- **`baselines.py`** — classical time-series baselines (AR/ARMA/ARIMA/SARIMA via `statsmodels`), used as the lower-bound comparison in `docs/results.md` §3. Needs the optional `research` extra (`pip install enerdatasynth[research]`); not on the `model().sample()` path.
- **`failure_analysis.py`** — `analyze(run_id)` and `write_failure_analysis_md(result)`, automated failure-mode detection (ramp-rate mismatch, seasonal mismatch) against a trained run, writing a markdown report. Backs `enerdatasynth evaluate`.
- **`results_report.py`** — `generate_results_summary()`, which regenerates `data/reports/pjm_load/results_summary.md`. Backs `enerdatasynth report`.

## Public API vs. internal

`compute_metrics` and `calibrate_checkpoint` are re-exported at the top level. `baselines`, `failure_analysis`, and `results_report` are CLI-support internals — import the submodule directly if you need one outside the CLI.

## What should not be changed casually

`metrics.py`'s formulas. Every number in `docs/results.md`, and every report under `data/research_history/pjm_load/reports/`, was computed with this exact math — the module's own docstring notes it reproduces the math verbatim from the mutually-identical `metrics()`/`_acf()` pair in `week6.py`, `week7.py`, and `arch_compare.py`. Change a formula here and every historical number stops being comparable to a newly computed one.
