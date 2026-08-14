# `tests/`

252 tests, all passing (`pytest tests/`, ~20s). Fast by design — no test needs the
full raw dataset.

## Layout

Resource-nested, matching `src/enerdatasynth/resources/`: `pjm_load/` and `wind/` hold
each resource's own tests; the top level holds the shared-contract tests
(`BaseGenerator`, the fluent SDK, `RunConfig`/`PROJECT_ROOT`, the CLI's `--resource`
dispatch).

| Path | Covers |
|---|---|
| `test_config.py` | `RunConfig`, `PROJECT_ROOT` resolution (cwd/env var, not `__file__` — see `MIGRATION.md`) |
| `test_generators.py` | `BaseGenerator` contract (resource-agnostic) + `LoadDiffusionModel` specifics |
| `test_public_api.py` | The fluent `enerdatasynth.model().set().sample()` contract |
| `test_cli.py` | `enerdatasynth.cli`'s `--resource {pjm_load,wind}` dispatch on `generate`/`evaluate`/`report` and the parser shape — not each pjm_load-only command's own body, which is covered via the underlying modules' tests instead |
| `conftest.py` | Shared fixtures — available to `pjm_load/` and `wind/` automatically via pytest's directory-based discovery |
| [`pjm_load/`](pjm_load/README.md) | Load's pipeline, architectures, sampler (incl. the sampler-bug regression tests), calibration, metrics, and its standalone `api.py` |
| [`wind/`](wind/README.md) | The `enerdatasynth`-side wind bridge, plus the vendored wind pipeline's own internals (API, evaluation reporting, sampling, registry, transforms, ...) |

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
pytest tests/test_cli.py -v                  # just the CLI dispatch tests
```

## Not covered yet

`test_cli.py` covers the `--resource` dispatch and parser shape, not every pjm_load-only
command's full body against real data end-to-end (`preprocess`/`train`/`sweep`/
`calibrate`/`pipeline` — verified manually instead, since each already delegates to a
module with its own tests). No test trains a Transformer against the current 23-dim
conditioning to check parity with the Week 7 numbers. See `docs/release_audit.md`.
