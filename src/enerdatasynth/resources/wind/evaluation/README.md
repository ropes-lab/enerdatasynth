# `enerdatasynth.resources.wind.evaluation`

**Wind's scoring tooling specifically** — load's 7-metric suite and calibration are
independent code in
[`src/enerdatasynth/resources/pjm_load/evaluation/`](../../pjm_load/evaluation/README.md).
Wind has no calibration step at all (`docs/models/wind.md` §12) — everything below is a
different metric suite for a different pipeline.

## Files

- **`evaluate.py`** — `generate_multiple_for_condition`, `build_condition_vector`: what
  the `enerdatasynth` bridge (`generators/wind_conditioning.py`) calls to turn `.set()`
  conditions into a raw conditioning vector and generate against it.
- **`metrics.py`** — the fidelity-gate metric suite (`marginal_wasserstein`, `acf_rmse`,
  `spectral_low_band`/`spectral_high_band`, `ramp_wasserstein`, `diurnal_rmse`,
  `regime_accuracy`) — `docs/models/wind.md` §7.
- **`failure_analysis.py`** — `analyze(preset)` / `write_failure_analysis_md(result)`:
  flags weak spots (e.g. near-chance `mid`-regime accuracy) already present in
  `../manifest.json`'s real, measured numbers. Does not run a new comparison — there's
  no real data in this repo to compare against. Mirrors
  `resources/pjm_load/evaluation/failure_analysis.py`; backs
  `enerdatasynth evaluate --resource wind`.
- **`results_report.py`** — `generate_results_summary()`: regenerates
  `data/reports/wind/results_summary.md` across all three presets. Mirrors
  `resources/pjm_load/evaluation/results_report.py`; backs
  `enerdatasynth report --resource wind`.

## Public API vs. internal

None of these are re-exported at the top-level `enerdatasynth` package — reached via
`enerdatasynth.resources.wind.evaluation.<module>` directly, or through the CLI
(`--resource wind`).
