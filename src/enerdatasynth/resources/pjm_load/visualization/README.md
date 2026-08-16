# `enerdatasynth.resources.pjm_load.visualization`

Validation plots comparing real and generated load profiles. Needs the optional `research` extra (`pip install enerdatasynth[research]`, mainly for matplotlib/statsmodels) since it's not on the `model().sample()` path — it's imported lazily wherever it's actually needed (`training/train.py`, `cli.py`), so a plain `import enerdatasynth` never pulls it in.

## Files

- **`plots.py`**:
  - `plot_hourly_distribution_comparison` — boxplots per hour, real vs. generated. This is the figure behind `data/reports/pjm_load/figures/hourly_distribution_comparison_*.png`.
  - `plot_first_difference_comparison`, `plot_autocorrelation_comparison` — the other two standard validation plots.
  - `plot_loss_curve` — training loss curve, saved next to each checkpoint.
  - `run_all_validation_plots(run_id)` — runs all three comparison plots for one run; backs `enerdatasynth evaluate`.

## Public API vs. internal

Not re-exported at the top level. Import `enerdatasynth.resources.pjm_load.visualization.plots` directly, or just use the CLI (`enerdatasynth evaluate`) instead of calling these by hand.

## What should not be placed here

Metric computation, which belongs in `evaluation/metrics.py`. This module only plots numbers that module already computed.
