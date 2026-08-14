# `tests/`

230 tests, all currently passing (`pytest tests/`, ~15-20s including the wind suite).
Fast by design — no test requires the full 10-year raw dataset.

## Layout

| File | Covers |
|---|---|
| `test_config.py` | `RunConfig`, path resolution (`PROJECT_ROOT` from cwd/env var, not `__file__` — see `MIGRATION.md` for why that distinction matters for non-editable installs) |
| `test_preprocessing.py` | Raw-CSV-to-tensor pipeline shapes and cleaning rules |
| `test_models.py` | Denoiser architectures (`build_denoiser`, `TransformerDenoiser`, `SimpleMLPDenoiser`) |
| `test_diffusion.py` | `DDPM` forward/reverse process, beta schedules, **the sampler-bug regression tests added during this audit** (`test_sampler_recomputes_noise_prediction_every_step`, `test_sampler_noise_prediction_differs_across_steps`) |
| `test_sampling.py` | Batch generation against a trained checkpoint |
| `test_calibration.py` | Bias computation and correction |
| `test_metrics.py` | The 7-metric suite |
| `test_generators.py` | `BaseGenerator` contract enforcement, `LoadDiffusionModel` |
| `test_public_api.py` | The fluent `enerdatasynth.model().set().sample()` contract |
| `test_wind_integration.py`, `wind/` | Wind resource: vendored `wind_diffusion` package, checkpoints, conditioning bridge |
| `conftest.py` | Shared fixtures |

## What a researcher should look at first

`test_diffusion.py`'s two sampler-regression tests, if you're touching
`diffusion/ddpm.py` for any reason — they exist specifically to catch a reintroduction
of the historical sampler bug (`docs/research_status.md` §E). They assert the model is
called exactly once per reverse step and that its prediction actually changes as
sampling progresses; a passing test suite with a lower call count or a stale-prediction
bug reintroduced would still show as `230 passed` unless these specific tests exist and
are checked.

## How to run

```bash
pytest tests/                    # everything, ~15-20s
pytest tests/test_diffusion.py -v  # just the diffusion/sampler tests
```

## What's not covered yet (flagged, not silently ignored)

Per `docs/release_audit.md`: no dedicated CLI-level integration test exercises each
`enerdatasynth <subcommand>` end-to-end against real data (the `--help` sweep was
verified manually during this audit, not automated as a test); no test currently trains
a Transformer against the current 23-dim conditioning to confirm parity with — or
honest divergence from — the Week 7 research numbers.
