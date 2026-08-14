# `tests/pjm_load/`

Load's own pipeline, tested in isolation from the `enerdatasynth` bridge layer
(`tests/test_generators.py`/`test_public_api.py` cover the bridge itself):

- **`test_preprocessing.py`** — raw-CSV loading, DST/missing-data cleaning,
  conditioning-vector construction.
- **`test_models.py`** — the Transformer and MLP denoisers.
- **`test_diffusion.py`** — forward/reverse diffusion, including the two
  sampler-regression tests guarding against reintroducing the historical ancestral-
  sampler bug (`docs/research_status.md` §E) — look here first before touching
  `resources/pjm_load/diffusion/ddpm.py`.
- **`test_sampling.py`** — `sampling/generate.py`'s `generate_for_run`.
- **`test_calibration.py`** — the scalar bias-correction fit.
- **`test_metrics.py`** — the 7-metric evaluation suite.
- **`test_api.py`** — the standalone `resources/pjm_load/api.py`
  (`generate_load_profile`, `list_models`, `describe_model`), independent of the
  `enerdatasynth.model()` bridge. Mirrors `tests/wind/test_api.py`.

See [`../README.md`](../README.md) for how to run the full suite; these run as part of
it, no separate invocation needed. Mirrors [`../wind/`](../wind/README.md) — that
folder is wind's per-resource test suite, this is load's.
