# `tests/wind/`

Tests for the vendored `wind_diffusion` package: `test_api.py`, `test_registry.py`
(checkpoint resolution), `test_transforms.py`, `test_samplers.py`, `test_train.py`,
`test_metrics.py`, `test_evaluate.py`, `test_plotting.py`, and `test_golden.py`
(fixed-shape pipeline regression checks). See [`../README.md`](../README.md) for how to
run the full suite; these run as part of it, no separate invocation needed.

Paired with `../test_wind_integration.py` one level up, which tests the
`enerdatasynth`-side bridge (`generators/wind.py`, `wind_conditioning.py`) specifically
— this folder tests the vendored package in isolation.
