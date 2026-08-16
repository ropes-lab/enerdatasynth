# `src/enerdatasynth/`

The installable package. It never imports from `data/research_history/`, so it stays fully self-contained.

**Start here:** `sdk.py` is the public entry point (`enerdatasynth.model(...)`), then `registry.py` for how a resource plugs in, then `generators/load.py` and `generators/load_conditioning.py` as the reference implementation a new resource would mirror. Before trusting any number this package produces, read the "Critical" callouts in `docs/research_status.md`.

## Public API vs. internal

`__init__.py`'s `__all__` is the real contract, split into two tiers:

- **Tier 1 (everyday use):** `model`, `ResourceModel`, `list_resources`, `describe_resource`. Never exposes tensors, architecture classes, or checkpoint paths.
- **Tier 2 (research/development):** `BaseGenerator`, `LoadDiffusionModel`, `RunConfig`, `set_seed`, `DEFAULT_ARCHITECTURE`, `build_denoiser`, `TransformerDenoiser`, `SimpleMLPDenoiser`, `DDPM`, `run_preprocessing`, `train_one_config`, `generate_for_run`, `calibrate_checkpoint`, `compute_metrics`.

Everything else — `evaluation.baselines`, `visualization.plots`, wind's internals — counts as implementation detail. Import the submodule directly if you need it.

## Package map

Every resource nests under `resources/<name>/` with the same category taxonomy (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, ...). See [`docs/architecture.md`](../../docs/architecture.md) for why. `generators/`, `registry.py`, `conditioning.py`, `sdk.py`, `utils/`, and `cli.py` sit above that as the shared layer both resources go through.

| Path | Scope | What it does |
|---|---|---|
| [`generators/`](generators/README.md) | Shared | `BaseGenerator` contract + both resources |
| `registry.py` | Shared | How a resource registers itself with `sdk.py` |
| `conditioning.py` | Shared | Generic `ConditionField` → raw vector machinery |
| `sdk.py` | Shared | The public fluent API |
| [`utils/`](utils/README.md) | Shared | Config, paths, seeding, device resolution |
| `cli.py` | Mostly PJM load | `preprocess`/`train`/`sweep`/`calibrate`/`pipeline` are load's from-scratch training workflow (load-only — wind ships pre-trained checkpoints instead); `generate`/`evaluate`/`report` take a `--resource {pjm_load,wind}` flag and dispatch to either resource |
| [`resources/pjm_load/`](resources/pjm_load/README.md) | PJM load only | `data/`, `models/`, `diffusion/`, `training/`, `sampling/`, `evaluation/`, `visualization/` plus `api/config/exceptions/manifest.json` at its root — each subfolder with its own README |
| [`resources/wind/`](resources/wind/README.md) | Wind only | Vendored pipeline; own `data/models/diffusion/training/evaluation/sampling/visualization` plus `api/config/registry/exceptions/manifest.json` at its root — `registry.py` (external-download resolution) is wind's one piece with no load equivalent, since load has no external-download path to resolve |

## What should not be changed casually

- `resources/pjm_load/diffusion/ddpm.py`'s reverse sampler. Recomputing the noise prediction at every step is load-bearing: a historical bug here silently corrupted results for weeks (see `docs/research_status.md` §E). `tests/pjm_load/test_diffusion.py` guards against it happening again.
- `generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC` and `COND_DIM`. A runtime assertion ties them together; change one without the other and checkpoints break silently.
- `resources/pjm_load/evaluation/metrics.py`'s `compute_metrics`. This is the single canonical implementation, replacing four inconsistent versions that used to be scattered across the research history.

## Where the scientific provenance lives

`../../data/research_history/pjm_load/` has the frozen scripts and reports. `../../docs/research_status.md`, `results.md`, and `experiments.md` are the synthesized, traceable version of the same story.
