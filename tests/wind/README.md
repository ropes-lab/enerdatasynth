# `tests/wind/`

- **`test_integration.py`** — tests the `enerdatasynth`-side bridge specifically (`generators/wind.py`, `wind_conditioning.py`): checkpoints, conditioning, and the fluent `.set()`/`.sample()` path through `enerdatasynth.model("wind", ...)`.
- **Everything else** (`test_api.py`, `test_registry.py`, `test_transforms.py`, `test_samplers.py`, `test_train.py`, `test_metrics.py`, `test_evaluate.py`, `test_plotting.py`, `test_golden.py`, `test_failure_analysis.py`, `test_results_report.py`, `test_sampling_generate.py`) — the vendored wind pipeline's own internals, plus its added-for-parity modules (`api.py`, `evaluation/failure_analysis.py`, `evaluation/results_report.py`, `sampling/generate.py`), tested separately from the bridge above.

See [`../README.md`](../README.md) for how to run the full suite — these run as part of it, no separate invocation needed. Mirrors [`../pjm_load/`](../pjm_load/), load's equivalent per-resource test suite.
