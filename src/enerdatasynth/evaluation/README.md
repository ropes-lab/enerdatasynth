# `enerdatasynth.evaluation`

**PJM load's scoring/calibration tooling specifically** — wind's fidelity-gate metrics
(`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, ...) are independent code in
[`src/wind_diffusion/metrics.py`](../../wind_diffusion/README.md), and wind has no
calibration step at all (see [`docs/models/wind.md`](../../../docs/models/wind.md) §7/§12
for why both differ). Everything below is load-specific.

Everything involved in scoring and correcting a trained load checkpoint's generated output.

## Files

- **`metrics.py`** — `compute_metrics(real, generated)`: the single canonical
  implementation of the 7-metric suite (Hourly, FirstDiff, ACF, RMSE, Peak, Energy,
  Ramp), replacing four previously-inconsistent versions scattered across
  `data/research_history/scripts/`. Exact math in `docs/research_status.md` §G.
- **`calibration.py`** — `compute_bias`, `calibrate_from_samples`,
  `apply_calibration`, `calibrate_checkpoint`: the scalar post-hoc bias correction
  validated in Week 7. Fits on training days only, never validation. See
  `docs/research_status.md` §F for why scalar (not per-hour) is the default.
- **`baselines.py`** — Classical time-series baselines (AR/ARMA/ARIMA/SARIMA via
  `statsmodels`) used as the lower-bound comparison in `docs/results.md` §3. Needs the
  optional `research` extra (`pip install enerdatasynth[research]`) — not on the
  `model().sample()` path.
- **`failure_analysis.py`** — `analyze(run_id)` / `write_failure_analysis_md(result)`:
  automated failure-mode detection (ramp-rate mismatch, seasonal mismatch) against a
  trained run, writing a markdown report. Backs `enerdatasynth evaluate`.
- **`results_report.py`** — `generate_results_summary()`: regenerates
  `data/reports/results_summary.md`. Backs `enerdatasynth report`.

## Public API vs. internal

`compute_metrics` and `calibrate_checkpoint` are re-exported at the top level.
`baselines`, `failure_analysis`, and `results_report` are CLI-support internals — import
the submodule directly if needed outside the CLI.

## What should not be changed casually

`metrics.py`'s formulas — every number in `docs/results.md` and every report under
`data/research_history/reports/` was computed with this exact math (verified: it
"reproduces the math verbatim from the mutually-identical `metrics()`/`_acf()` pair in
week6.py / week7.py / arch_compare.py" per the module's own docstring). Changing a
formula here silently invalidates every historical number's comparability to a newly
computed one.
