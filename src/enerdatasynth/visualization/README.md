# `enerdatasynth.visualization`

Validation plots comparing real vs. generated load profiles. Needs the optional
`research` extra (`pip install enerdatasynth[research]` — matplotlib/statsmodels) since
it's not on the `model().sample()` path; imported lazily wherever it's needed
(`training/train.py`, `cli.py`) so a plain `import enerdatasynth` never requires it.

## Files

- **`plots.py`**:
  - `plot_hourly_distribution_comparison` — boxplots per hour, real vs. generated (the
    figure behind `data/reports/figures/hourly_distribution_comparison_*.png`)
  - `plot_first_difference_comparison`, `plot_autocorrelation_comparison` — the other
    two standard validation plots
  - `plot_loss_curve` — training loss curve, saved next to each checkpoint
  - `run_all_validation_plots(run_id)` — runs all three comparison plots for one run;
    backs `enerdatasynth evaluate`

## Public API vs. internal

Not re-exported at the top level — import `enerdatasynth.visualization.plots` directly,
or use the CLI (`enerdatasynth evaluate`) instead of calling these functions by hand.

## What should not be placed here

Metric computation (belongs in `evaluation/metrics.py`) — this module only plots
numbers that module already computed.
