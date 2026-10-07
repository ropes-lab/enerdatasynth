# `tests/pjm_load/`

Load's own pipeline, tested separately from the `enerdatasynth` bridge layer (that part's covered by `tests/test_generators.py` and `test_public_api.py`):

- **`test_preprocessing.py`** — raw-CSV loading, DST/missing-data cleaning, conditioning-vector construction.
- **`test_models.py`** — the Transformer and MLP denoisers.
- **`test_diffusion.py`** — forward/reverse diffusion, including the two sampler-regression tests that guard against reintroducing the historical ancestral-sampler bug (`docs/research_status.md` §E). Look here first before touching `resources/pjm_load/diffusion/ddpm.py`.
- **`test_sampling.py`** — `sampling/generate.py`'s `generate_for_run`.
- **`test_calibration.py`** — the scalar bias-correction fit.
- **`test_metrics.py`** — the 7-metric evaluation suite.
- **`test_failure_analysis.py`** and **`test_results_report.py`** — the failure-mode detector and the results-summary generator, run against the real bundled validation set and sweep results (reports go to a temp dir). Mirror the same-named files in `tests/wind/`.
- **`test_api.py`** — the standalone `resources/pjm_load/api.py` (`generate_load_profile`, `list_models`, `describe_model`), independent of the `enerdatasynth.model()` bridge. Mirrors `tests/wind/test_api.py`.

See [`../README.md`](../README.md) for how to run the full suite — these run as part of it, no separate invocation needed. Mirrors [`../wind/`](../wind/README.md), which is wind's equivalent per-resource test suite.
