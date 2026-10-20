# `enerdatasynth.resources.pjm_load.visualization`

Validation plots that compare real and generated load profiles. It needs the optional `research` extra (`pip install enerdatasynth[research]`, mainly for matplotlib and statsmodels) because it isn't on the `model().sample()` path. It is imported lazily where it's used (`training/train.py`, `cli.py`), so a plain `import enerdatasynth` never loads it.

## Files

**`plots.py`**:

- `plot_hourly_distribution_comparison`: a boxplot per hour, real vs. generated. This produces `data/reports/pjm_load/figures/hourly_distribution_comparison_*.png`.
- `plot_first_difference_comparison` and `plot_autocorrelation_comparison`: the other two standard validation plots.
- `plot_loss_curve`: the training loss curve, saved next to each checkpoint.
- `run_all_validation_plots(run_id)`: runs the three comparison plots for one run. Behind `enerdatasynth evaluate`.

The comparison plots read the processed validation arrays, so you need your own `enerdatasynth preprocess` output first.

## Public API

Not re-exported at the top level. Import `enerdatasynth.resources.pjm_load.visualization.plots` directly, or use `enerdatasynth evaluate`.

## What doesn't belong here

Metric computation. That lives in `evaluation/metrics.py`; this module only plots numbers it has already computed.
