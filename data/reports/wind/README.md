# `data/reports/wind/`

Written by `enerdatasynth evaluate --resource wind --preset <preset>`
(`failure_analysis_<preset>.md`) and `enerdatasynth report --resource wind`
(`results_summary.md`). The files are regenerated on every run, as with PJM load's
[`../pjm_load/`](../README.md) reports, so don't edit them by hand.

These reports summarize and flag issues in the numbers stored in
`enerdatasynth.resources.wind.manifest.json`; they don't compute anything new. See
[`docs/models/wind.md`](../../../docs/models/wind.md) §7–8 for the numbers themselves.
