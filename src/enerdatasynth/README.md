# `src/enerdatasynth/`

The installable package. Never imports from `data/research_history/` — fully
self-contained.

**Start here:** `sdk.py` (the public entry point, `enerdatasynth.model(...)`), then
`registry.py` (how a resource plugs in), then `generators/load.py` +
`generators/load_conditioning.py` (the reference implementation a new resource mirrors).
Before trusting any number this package produces, read `docs/research_status.md`'s
"Critical" callouts.

## Public API vs. internal

`__init__.py`'s `__all__` is the real contract, in two tiers:

- **Tier 1 (everyday use):** `model`, `ResourceModel`, `list_resources`,
  `describe_resource` — never exposes tensors, architecture classes, or checkpoint paths.
- **Tier 2 (research/development):** `BaseGenerator`, `LoadDiffusionModel`, `RunConfig`,
  `set_seed`, `DEFAULT_ARCHITECTURE`, `build_denoiser`, `TransformerDenoiser`,
  `SimpleMLPDenoiser`, `DDPM`, `run_preprocessing`, `train_one_config`,
  `generate_for_run`, `calibrate_checkpoint`, `compute_metrics`.

Everything else (e.g. `evaluation.baselines`, `visualization.plots`, wind's internals)
is implementation detail — import the submodule directly if you need it.

## Package map

Every resource nests under `resources/<name>/` with the same category taxonomy
(`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, ...) — see
[`docs/architecture.md`](../../docs/architecture.md) for why. `generators/`,
`registry.py`, `conditioning.py`, `sdk.py`, `utils/`, `cli.py` sit above that as the
shared layer both resources go through.

| Path | Scope | What it does |
|---|---|---|
| [`generators/`](generators/README.md) | Shared | `BaseGenerator` contract + both resources |
| `registry.py` | Shared | How a resource registers itself with `sdk.py` |
| `conditioning.py` | Shared | Generic `ConditionField` → raw vector machinery |
| `sdk.py` | Shared | The public fluent API |
| [`utils/`](utils/README.md) | Shared | Config, paths, seeding, device resolution |
| `cli.py` | PJM load only | Load's from-scratch training workflow |
| [`resources/pjm_load/`](resources/pjm_load/) | PJM load only | `data/`, `models/`, `diffusion/`, `training/`, `sampling/`, `evaluation/`, `visualization/` — each with its own README |
| [`resources/wind/`](resources/wind/README.md) | Wind only | Vendored pipeline; own `data/models/diffusion/training/evaluation/visualization` plus `api/config/registry/exceptions` with no load equivalent |

## What should not be changed casually

- `resources/pjm_load/diffusion/ddpm.py`'s reverse sampler — recomputing the noise
  prediction every step is load-bearing (a historical bug here silently corrupted
  results for weeks — see `docs/research_status.md` §E). Guarded by
  `tests/pjm_load/test_diffusion.py`.
- `generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC` / `COND_DIM` — a runtime
  assertion ties them together; changing one without the other breaks checkpoints
  silently.
- `resources/pjm_load/evaluation/metrics.py`'s `compute_metrics` — the single canonical
  implementation, replacing four inconsistent versions across research history.

## Where the scientific provenance lives

`../../data/research_history/pjm_load/` (frozen scripts/reports) and
`../../docs/research_status.md` / `results.md` / `experiments.md` (the synthesized,
traceable version).
