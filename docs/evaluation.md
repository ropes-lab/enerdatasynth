# Evaluation

**This page covers PJM load's 7-metric evaluation suite specifically.** Wind uses a
different metric suite entirely (fidelity gates: `marginal_wasserstein`, `acf_rmse`,
`regime_accuracy`, and others) — see [`docs/models/wind.md`](models/wind.md) §7 for
those. Both are documented in full, resource-specific detail in their respective model
reports ([`docs/models/pjm_load.md`](models/pjm_load.md) §7,
[`docs/models/wind.md`](models/wind.md) §7); this page is the deeper reference for
load's suite specifically.

## The 7-metric suite

`enerdatasynth.resources.pjm_load.evaluation.metrics.compute_metrics(real, generated)`
returns a dict of 7 scalars, all computed on normalized `[0,1]` load
(lower is better for all of them):

| Metric | Definition |
|---|---|
| `Hourly` | Mean absolute difference in per-hour mean, plus mean absolute difference in per-hour IQR (a composite distribution-shape metric — not a standard named distance, defined here explicitly since it's project-specific) |
| `FirstDiff` | `\|std(diff(real)) - std(diff(generated))\|`, hour-over-hour |
| `ACF` | Mean absolute difference in autocorrelation, lags 0-12 (`autocorrelation()` in the same module) |
| `RMSE` | RMSE between the real and generated mean profiles |
| `Peak` | `\|mean(daily max, real) - mean(daily max, generated)\|` |
| `Energy` | `\|mean(daily sum, real) - mean(daily sum, generated)\|` |
| `Ramp` | `\|mean(\|hour-over-hour delta\|, real) - mean(\|...\|, generated)\|` |

This is the exact suite used throughout `data/research_history/pjm_load/` (verified
identical across week6/week7/arch_compare — see MIGRATION.md §3), so
numbers here are directly comparable to the historical reports.

## Classical baselines

`enerdatasynth.resources.pjm_load.evaluation.baselines` fits AR/ARMA/ARIMA/SARIMA models
to the mean training profile and simulates synthetic days from each,
for benchmarking the diffusion model against non-deep-learning
alternatives. See `examples/run_benchmark.py` for a runnable
comparison script.

## Calibration

`enerdatasynth.resources.pjm_load.evaluation.calibration` measures a per-hour bias
between real and generated profiles **on the training set only**
(never validation — that would be fitting to the test data) and
provides two correction modes:

- `scalar` (default, recommended): one number added to every hour.
- `per_hour`: a 24-value vector, one correction per hour.

The research evidence is explicit that `scalar` is the better choice
despite being the simpler one — `per_hour` actually degrades ACF
(0.0317→0.0501) and RMSE (0.0210→0.0249) relative to no calibration at
all, while `scalar` matches or improves every metric
(`data/research_history/pjm_load/reports/week7/calibration_report.md`). Don't assume
"more sophisticated" means "better" here; the numbers say otherwise.

## Conditional / slice evaluation

`enerdatasynth.resources.pjm_load.evaluation.failure_analysis.analyze` automatically
flags peak/ramp/variance-collapse/seasonal mismatches above configured
thresholds. The research lineage additionally sliced by temperature
tercile, weekday/weekend/holiday, and season
(`data/research_history/pjm_load/scripts/week7.py`'s `task_evaluate`) — not yet
promoted into `src/` as a reusable module; see MIGRATION.md §4 if
you want to pick this up.

## A methodological finding worth keeping in mind

In the architecture comparison, the model with the lowest validation
loss (CNN, 0.01022) was not the one with the best generated-profile
RMSE (transformer, 0.0215) —
`data/research_history/pjm_load/reports/week6/arch/summary_report.md` says so
explicitly: "Judge architectures on the generated metrics, not val
loss alone." (The noise-schedule experiment, by contrast, agreed on
both metrics: linear won validation loss *and* generated RMSE — so
this isn't a universal rule, just a documented case where it mattered
and a reason to always check both rather than only validation loss.)
