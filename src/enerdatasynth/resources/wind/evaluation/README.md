# `enerdatasynth.resources.wind.evaluation`

Wind's evaluation code. Load's 7-metric suite and calibration are separate code in
[`src/enerdatasynth/resources/pjm_load/evaluation/`](../../pjm_load/evaluation/README.md).
Wind uses a different set of metrics and has no calibration step
(`docs/models/wind.md` §12).

## Files

- **`evaluate.py`:** `generate_multiple_for_condition` and `build_condition_vector`. The
  `enerdatasynth` wrapper (`generators/wind_conditioning.py`) calls these to turn
  `.set()` conditions into a conditioning vector and generate from it.
- **`metrics.py`:** the metrics `marginal_wasserstein`, `acf_rmse`,
  `spectral_low_band`/`spectral_high_band`, `ramp_wasserstein`, `diurnal_rmse` and
  `regime_accuracy` (`docs/models/wind.md` §7).
- **`failure_analysis.py`:** `analyze(preset)` and `write_failure_analysis_md(result)`
  flag weak spots, such as near-chance `mid` accuracy, in the numbers already in
  `../manifest.json`. They don't run a new comparison, because there's no real data
  here. Used by `enerdatasynth evaluate --resource wind`; matches
  `resources/pjm_load/evaluation/failure_analysis.py`.
- **`results_report.py`:** `generate_results_summary()` regenerates
  `data/reports/wind/results_summary.md` for all three presets. Used by
  `enerdatasynth report --resource wind`; matches
  `resources/pjm_load/evaluation/results_report.py`.

## Public API vs. internal

None of these are exported from the top-level `enerdatasynth` package. Import them from
`enerdatasynth.resources.wind.evaluation.<module>`, or use the CLI with
`--resource wind`.
