# `src/enerdatasynth/`

The installable package. It never imports from `data/research_history/`, so it stays self-contained.

**Where to start:** `sdk.py` is the public entry point (`enerdatasynth.model(...)`). Then `registry.py` shows how a resource plugs in, and `generators/load.py` and `generators/load_conditioning.py` are the reference implementation a new resource would copy. Before you trust any number this package produces, read the "Critical" callouts in `docs/research_status.md`.

## Public API

The `__all__` in `__init__.py` is the real contract, in two tiers:

- **Tier 1 (everyday use):** `model`, `ResourceModel`, `list_resources`, `describe_resource`. These never expose tensors, architecture classes or checkpoint paths.
- **Tier 2 (research and development):** `BaseGenerator`, `LoadDiffusionModel`, `RunConfig`, `set_seed`, `DEFAULT_ARCHITECTURE`, `build_denoiser`, `TransformerDenoiser`, `SimpleMLPDenoiser`, `DDPM`, `run_preprocessing`, `train_one_config`, `generate_for_run`, `calibrate_checkpoint`, `compute_metrics`.

Everything else (`evaluation.baselines`, `visualization.plots`, wind's internals) is an implementation detail. Import the submodule directly if you need it.

## Package map

Each resource lives under `resources/<name>/` with the same folders (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/` and so on); [`docs/architecture.md`](../../docs/architecture.md) explains why. `generators/`, `registry.py`, `conditioning.py`, `sdk.py`, `utils/` and `cli.py` sit above them as the shared layer both resources go through.

| Path | Scope | What it does |
|---|---|---|
| [`generators/`](generators/README.md) | Shared | The `BaseGenerator` contract and both resources' generators |
| `registry.py` | Shared | How a resource registers itself with `sdk.py` |
| `conditioning.py` | Shared | The generic `ConditionField` to raw-vector machinery |
| `sdk.py` | Shared | The public fluent API |
| [`utils/`](utils/README.md) | Shared | Config, paths, seeding, device selection |
| `cli.py` | Mostly load | `preprocess`, `train`, `sweep`, `calibrate` and `pipeline` are load's training workflow (wind ships pre-trained instead). `generate`, `evaluate` and `report` take `--resource {pjm_load,wind}` and dispatch to either |
| [`resources/pjm_load/`](resources/pjm_load/README.md) | Load only | `data/`, `models/`, `diffusion/`, `training/`, `sampling/`, `evaluation/`, `visualization/`, plus `api`, `config`, `exceptions` and `manifest.json` at its root. Each subfolder has its own README |
| [`resources/wind/`](resources/wind/README.md) | Wind only | The vendored pipeline, with its own `data`, `models`, `diffusion`, `training`, `evaluation`, `sampling` and `visualization` folders plus `api`, `config`, `registry`, `exceptions` and `manifest.json`. `registry.py` resolves external checkpoint downloads, which load has no equivalent of |

## Be careful with

- The reverse sampler in `resources/pjm_load/diffusion/ddpm.py`. Recomputing the noise prediction at every step matters: a past bug here corrupted results for weeks (`docs/research_status.md` §E). `tests/pjm_load/test_diffusion.py` guards against a repeat.
- `LOAD_CONDITION_SPEC` in `generators/load_conditioning.py` and `COND_DIM`. A runtime assertion ties them together, and changing one without the other breaks checkpoints.
- `compute_metrics` in `resources/pjm_load/evaluation/metrics.py`. It is the single implementation, replacing four inconsistent copies from the research history.

## Provenance

`../../data/research_history/pjm_load/` has the frozen scripts and reports. `../../docs/research_status.md`, `results.md` and `experiments.md` are the written-up, traceable version of the same story.
