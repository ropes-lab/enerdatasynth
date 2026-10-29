# `tests/pjm_load/`

Tests for the load pipeline on its own. The bridge layer that sits above it is covered by `tests/test_generators.py` and `test_public_api.py`.

- **`test_preprocessing.py`**: raw-CSV loading, DST and missing-data cleaning, building the conditioning vector.
  It also covers zones made of several load areas (AE, DPL, PL): areas are summed, partial hours are dropped, and 25-hour days survive.
- **`test_zones.py`**: the 17 per-zone models: the `zones/<ZONE>/` folder layout, `--zone` on the CLI and `$ENERDATASYNTH_ZONE`, the SDK's case-insensitive zone lookup, and that every zone has its scalers and checkpoint and generates plausible load.
- **`test_models.py`**: the Transformer and MLP denoisers.
- **`test_diffusion.py`**: forward and reverse diffusion. It includes the two sampler-regression tests that stop the old ancestral-sampler bug coming back (`docs/research_status.md` §E). Read these before touching `resources/pjm_load/diffusion/ddpm.py`.
- **`test_sampling.py`**: `generate_for_run` in `sampling/generate.py`, using a small synthetic validation set, plus the error you get when the data is missing.
- **`test_calibration.py`**: the scalar bias-correction fit.
- **`test_metrics.py`**: the 7-metric evaluation suite.
- **`test_failure_analysis.py`** and **`test_results_report.py`**: the failure-mode detector and the results-summary generator. The failure-analysis tests build a synthetic dataset because the repository has no real data; reports go to a temp folder. They mirror the same-named files in `tests/wind/`.
- **`test_api.py`**: the standalone `resources/pjm_load/api.py` (`generate_load_profile`, `list_models`, `describe_model`), independent of `enerdatasynth.model()`. It mirrors `tests/wind/test_api.py`.

These run as part of the full suite; [`../README.md`](../README.md) has the command. The wind equivalent is [`../wind/`](../wind/README.md).
