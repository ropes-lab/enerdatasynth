# `tests/`

261 tests, all passing (`pytest tests/`, about 45s). Fast by design — no test needs the full raw dataset.

## Layout

Resource-nested, matching `src/enerdatasynth/resources/`: `pjm_load/` and `wind/` hold each resource's own tests, and the top level holds the shared-contract tests (`BaseGenerator`, the fluent SDK, `RunConfig`/`PROJECT_ROOT`, the CLI's `--resource` dispatch).

| Path | Covers |
|---|---|
| `test_config.py` | `RunConfig`, `PROJECT_ROOT` resolution (cwd/env var, not `__file__` — see `MIGRATION.md`) |
| `test_generators.py` | `BaseGenerator` contract (resource-agnostic) plus `LoadDiffusionModel` specifics |
| `test_public_api.py` | The fluent `enerdatasynth.model().set().sample()` contract |
| `test_cli.py` | `enerdatasynth.cli`'s `--resource {pjm_load,wind}` dispatch on `generate`/`evaluate`/`report` and the parser shape. Not each pjm_load-only command's own body — those are covered through the underlying modules' own tests |
| `conftest.py` | Shared fixtures, available to `pjm_load/` and `wind/` automatically via pytest's directory-based discovery |
| [`pjm_load/`](pjm_load/README.md) | Load's pipeline, architectures, sampler (including the sampler-bug regression tests), calibration, metrics, and its standalone `api.py` |
| [`wind/`](wind/README.md) | The `enerdatasynth`-side wind bridge, plus the vendored wind pipeline's own internals — API, evaluation reporting, sampling, registry, transforms, and so on |

The line between shared and per-resource tests is drawn by purpose, not by what today's assertions happen to touch. `test_generators.py` and `test_config.py` test genuinely shared contracts even though their concrete examples lean on load — wind doesn't have a `RunConfig`-driven training path to test against yet. A third resource's config/generator specifics would get their own `<resource>/test_*.py`, same as `pjm_load/` and `wind/` do now.

## What to look at first

`pjm_load/test_diffusion.py`'s two sampler-regression tests, before touching `resources/pjm_load/diffusion/ddpm.py`. They catch a reintroduction of the historical sampler bug (`docs/research_status.md` §E) by asserting the model gets called fresh once per reverse step. Without them, a regression here would still show all tests passing.

## How to run

```bash
pytest tests/                                # everything
pytest tests/pjm_load/test_diffusion.py -v   # just the sampler tests
pytest tests/wind/ -v                        # just the wind suite
pytest tests/test_cli.py -v                  # just the CLI dispatch tests
```

## Not covered yet

`test_cli.py` covers the `--resource` dispatch and parser shape, not every pjm_load-only command's full body against real data end to end — `preprocess`/`train`/`sweep`/`calibrate`/`pipeline` were verified manually instead, since each already delegates to a module with its own tests. No test trains a Transformer against the current 23-dim conditioning to check parity with the Week 7 numbers. See `docs/release_audit.md`.
