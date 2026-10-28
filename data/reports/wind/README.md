# `data/reports/wind/`

Written by `enerdatasynth evaluate --resource wind --preset <preset>`
(`failure_analysis_<preset>.md`) and `enerdatasynth report --resource wind`
(`results_summary.md`). The files are regenerated on every run, as with PJM load's
[`../pjm_load/`](../README.md) reports, so don't edit them by hand.

These reports summarize and flag issues in the numbers stored in
`enerdatasynth.resources.wind.manifest.json`; they don't compute anything new. See
[`docs/models/wind.md`](../../../docs/models/wind.md) §7–8 for the numbers themselves.

`figures/` holds `wind_generated_examples.png`, five generated days for each requested regime from the `1ch` checkpoint. It is made by `examples/wind/plot_generated_samples.py` (the same figure as `docs/assets/wind_generated_examples.png`) and shows generated output only. PJM load's [`figures/`](../pjm_load/figures/) also has real-vs-generated validation plots, which wind can't have here because the repository holds no real wind data.
