# `tests/`

266 tests, all passing (`pytest tests/`, about 30 seconds). No test needs the raw dataset; the ones that need data-shaped inputs build small synthetic ones.

## Layout

The folders follow `src/enerdatasynth/resources/`. `pjm_load/` and `wind/` hold each resource's own tests. The top level holds the tests for the shared pieces: `BaseGenerator`, the fluent SDK, `RunConfig` and `PROJECT_ROOT`, and the CLI's `--resource` dispatch.

| Path | Covers |
|---|---|
| `test_config.py` | `RunConfig`, `PROJECT_ROOT` resolution (cwd or env var, not `__file__`; see `MIGRATION.md`), and the missing-data error from `load_processed` |
| `test_generators.py` | The `BaseGenerator` contract (any resource) plus `LoadDiffusionModel` specifics |
| `test_public_api.py` | The fluent `enerdatasynth.model().set().sample()` contract |
| `test_cli.py` | `--resource {pjm_load,wind}` dispatch for `generate`, `evaluate` and `report`, and the parser shape. It doesn't test the body of each load-only command; those are covered by the tests for the modules they call |
| `conftest.py` | Shared fixtures, picked up by `pjm_load/` and `wind/` automatically |
| [`pjm_load/`](pjm_load/README.md) | Load's pipeline, architectures, sampler (including the sampler-bug regression tests), calibration, metrics and standalone `api.py` |
| [`wind/`](wind/README.md) | The wind bridge plus the vendored wind pipeline's own internals: API, evaluation reporting, sampling, registry, transforms and more |

A test goes at the top level when it covers a shared contract, even if its examples happen to use load. `test_generators.py` and `test_config.py` are like that, since wind has no `RunConfig`-driven training path to test against. A third resource would get its own `<resource>/test_*.py`.

## Read first

`pjm_load/test_diffusion.py` has two sampler-regression tests. Look at them before touching `resources/pjm_load/diffusion/ddpm.py`. They check that the model is called fresh at every reverse step, which is how the old sampler bug (`docs/research_status.md` §E) would come back. Without them, that regression would still show a green suite.

## Running them

```bash
pytest tests/                                # everything
pytest tests/pjm_load/test_diffusion.py -v   # just the sampler tests
pytest tests/wind/ -v                        # just the wind suite
pytest tests/test_cli.py -v                  # just the CLI dispatch tests
```

If you have `enerdatasynth` installed from PyPI as well, run with `PYTHONPATH=src` or install the repo in editable mode, otherwise pytest may test the installed wheel instead of your checkout.

## Not covered yet

`test_cli.py` checks dispatch and parser shape, not a full end-to-end run of `preprocess`, `train`, `sweep`, `calibrate` or `pipeline` against real data. Those were checked by hand, since each calls a module that has its own tests. No test trains a Transformer on the current 23-dimensional conditioning to check it against the Week 7 numbers. See `docs/release_audit.md`.
