# `tests/`

230 tests, all passing (`pytest tests/`, ~15-20s). Fast by design — no test needs the
full raw dataset.

## Layout

Resource-nested, matching `src/enerdatasynth/resources/`: `pjm_load/` and `wind/` hold
each resource's own tests; the top level holds the shared-contract tests
(`BaseGenerator`, the fluent SDK, `RunConfig`/`PROJECT_ROOT`).

| Path | Covers |
|---|---|
| `test_config.py` | `RunConfig`, `PROJECT_ROOT` resolution (cwd/env var, not `__file__` — see `MIGRATION.md`) |
| `test_generators.py` | `BaseGenerator` contract (resource-agnostic) + `LoadDiffusionModel` specifics |
| `test_public_api.py` | The fluent `enerdatasynth.model().set().sample()` contract |
| `conftest.py` | Shared fixtures — available to `pjm_load/` and `wind/` automatically via pytest's directory-based discovery |
| `pjm_load/test_{preprocessing,models,diffusion,sampling,calibration,metrics}.py` | Load's pipeline, architectures, sampler (incl. the sampler-bug regression tests), calibration, metrics |
| `wind/test_integration.py` | The `enerdatasynth`-side wind bridge (checkpoints, conditioning, `.set()`/`.sample()`) |
| `wind/test_{api,evaluate,golden,metrics,plotting,registry,samplers,train,transforms}.py` | The vendored wind pipeline's own internals, in isolation from the bridge |

The shared/per-resource line is drawn by *purpose*, not by what today's assertions
happen to touch — `test_generators.py`/`test_config.py` test genuinely shared contracts
even though their concrete examples lean on load (wind has no `RunConfig`-driven
training path to test against yet). A third resource's own config/generator specifics
would get their own `<resource>/test_*.py`, same as `pjm_load/` and `wind/`.

## What to look at first

`pjm_load/test_diffusion.py`'s two sampler-regression tests, before touching
`resources/pjm_load/diffusion/ddpm.py` — they catch a reintroduction of the historical
sampler bug (`docs/research_status.md` §E) by asserting the model is called fresh once
per reverse step. A regression here would still show `230 passed` without them.

## How to run

```bash
pytest tests/                                # everything
pytest tests/pjm_load/test_diffusion.py -v   # just the sampler tests
pytest tests/wind/ -v                        # just the wind suite
```

## Not covered yet

No CLI-level integration test exercises each `enerdatasynth <subcommand>` against real
data (verified manually instead); no test trains a Transformer against the current
23-dim conditioning to check parity with the Week 7 numbers. See `docs/release_audit.md`.
