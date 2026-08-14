# `src/enerdatasynth/`

The installable package. **Never imports from `data/research_history/`** — everything
here is self-contained and works independently of the research provenance directory.

## What a researcher should look at first

1. `sdk.py` — the public entry point (`enerdatasynth.model(...)`). Read this before
   anything else; it explains the whole fluent API in its module docstring.
2. `registry.py` — how a resource type (electric load, wind) plugs into `sdk.py` without
   `sdk.py` needing to know it exists.
3. `resources/pjm_load/../generators/load.py` / `generators/load_conditioning.py` — the
   reference implementation a new resource type mirrors.
4. `../../docs/research_status.md` — **before trusting any number this package
   produces**, read the "Critical" callouts about which architecture/conditioning is
   actually loaded by default today.

## Public API vs. internal implementation

`__init__.py`'s `__all__` is the actual contract — already curated into two tiers:

**Tier 1 (everyday use):** `model`, `ResourceModel`, `list_resources`,
`describe_resource`. This is all most callers need. It never exposes tensors,
architecture classes, or checkpoint paths.

**Tier 2 (research/development):** `BaseGenerator`, `LoadDiffusionModel`, `RunConfig`,
`set_seed`, `DEFAULT_ARCHITECTURE`, `build_denoiser`, `TransformerDenoiser`,
`SimpleMLPDenoiser`, `DDPM`, `run_preprocessing`, `train_one_config`,
`generate_for_run`, `calibrate_checkpoint`, `compute_metrics` — training new
checkpoints, computing metrics directly, running calibration.

Everything else (e.g. `resources.pjm_load.evaluation.baselines`,
`resources.pjm_load.visualization.plots`, wind's internals) is reached by importing the
specific submodule directly — it is not re-exported at the top level, and that's
intentional: these are implementation details or CLI-only tooling, not part of the
everyday contract.

## Package map

Every resource nests under `resources/<name>/` with the same internal category
taxonomy (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, plus whatever
else that resource actually needs) — see [`docs/architecture.md`](../../docs/architecture.md)
for the full reasoning. `generators/`, `registry.py`, `conditioning.py`, `sdk.py`,
`utils/`, and `cli.py` sit above that as the shared layer both resources go through.

| Path | Scope | What it does | README |
|---|---|---|---|
| `generators/` | **Shared** | `BaseGenerator` contract + the two concrete resources (load, wind) | [generators/README.md](generators/README.md) |
| `registry.py` | **Shared** | How a resource type registers itself with `sdk.py` | — |
| `conditioning.py` | **Shared** | Generic `ConditionField` → raw vector machinery, resource-agnostic | — |
| `sdk.py` | **Shared** | The public fluent API | — |
| `utils/` | **Shared** | Config, paths, seeding, device resolution | [utils/README.md](utils/README.md) |
| `cli.py` | PJM load only | `enerdatasynth` console script — load's from-scratch training workflow | — |
| `resources/pjm_load/data/` | PJM load only | Raw CSV → cleaned daily tensors; NOAA temperature fetch/cache | [resources/pjm_load/data/README.md](resources/pjm_load/data/README.md) |
| `resources/pjm_load/models/` | PJM load only | Denoiser architectures (Transformer, MLP) | [resources/pjm_load/models/README.md](resources/pjm_load/models/README.md) |
| `resources/pjm_load/diffusion/` | PJM load only | DDPM forward/reverse process, beta schedules | [resources/pjm_load/diffusion/README.md](resources/pjm_load/diffusion/README.md) |
| `resources/pjm_load/training/` | PJM load only | Training loop + hyperparameter sweep | [resources/pjm_load/training/README.md](resources/pjm_load/training/README.md) |
| `resources/pjm_load/sampling/` | PJM load only | Batch generation against a trained checkpoint | [resources/pjm_load/sampling/README.md](resources/pjm_load/sampling/README.md) |
| `resources/pjm_load/evaluation/` | PJM load only | 7-metric suite, calibration, classical baselines, failure analysis | [resources/pjm_load/evaluation/README.md](resources/pjm_load/evaluation/README.md) |
| `resources/pjm_load/visualization/` | PJM load only | Validation plots (matplotlib/seaborn — optional `[research]` extra) | [resources/pjm_load/visualization/README.md](resources/pjm_load/visualization/README.md) |
| `resources/wind/` | Wind only | Vendored diffusion pipeline (own data/models/diffusion/training/evaluation/visualization, plus api/config/registry/exceptions with no load equivalent) | [resources/wind/README.md](resources/wind/README.md) |

## What should not be changed casually

- `resources/pjm_load/diffusion/ddpm.py`'s reverse sampler — recomputing the noise
  prediction at every step is load-bearing; a historical bug here silently corrupted
  results for weeks before being caught (see `docs/research_status.md` §E). A regression
  test (`tests/test_diffusion.py`) guards this.
- `generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC` / `COND_DIM` — changing the
  conditioning schema changes what any existing checkpoint expects as input; there's a
  runtime assertion tying them together for exactly this reason.
- `resources/pjm_load/evaluation/metrics.py`'s `compute_metrics` — this is the single
  canonical implementation replacing four previously-inconsistent versions across
  research history; don't reintroduce a second version elsewhere.

## Where the scientific provenance lives

`../../data/research_history/pjm_load/` (frozen historical scripts/reports) and
`../../docs/research_status.md` / `results.md` / `experiments.md` (the synthesized,
traceable version of the same evidence).
