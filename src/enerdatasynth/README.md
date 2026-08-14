# `src/enerdatasynth/`

The installable package. **Never imports from `data/research_history/`** — everything
here is self-contained and works independently of the research provenance directory.

## What a researcher should look at first

1. `sdk.py` — the public entry point (`enerdatasynth.model(...)`). Read this before
   anything else; it explains the whole fluent API in its module docstring.
2. `registry.py` — how a resource type (electric load, wind) plugs into `sdk.py` without
   `sdk.py` needing to know it exists.
3. `generators/load.py` / `generators/load_conditioning.py` — the reference
   implementation a new resource type mirrors.
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

Everything else (e.g. `evaluation.baselines`, `evaluation.failure_analysis`,
`visualization.plots`, wind's internals) is reached by importing the specific submodule
directly — it is not re-exported at the top level, and that's intentional: these are
implementation details or CLI-only tooling, not part of the everyday contract.

## Subpackage map

| Subpackage | What it does | README |
|---|---|---|
| `data/` | Raw CSV → cleaned daily tensors; NOAA temperature fetch/cache | [data/README.md](data/README.md) |
| `models/` | Denoiser architectures (Transformer, MLP) | [models/README.md](models/README.md) |
| `diffusion/` | DDPM forward/reverse process, beta schedules | [diffusion/README.md](diffusion/README.md) |
| `training/` | Training loop + hyperparameter sweep | [training/README.md](training/README.md) |
| `sampling/` | Batch generation against a trained checkpoint | (see `training/README.md`, closely related) |
| `evaluation/` | 7-metric suite, calibration, classical baselines, failure analysis | [evaluation/README.md](evaluation/README.md) |
| `generators/` | `BaseGenerator` contract + the two concrete resources (load, wind) | [generators/README.md](generators/README.md) |
| `visualization/` | Validation plots (matplotlib/seaborn — optional `[research]` extra) | — |
| `utils/` | Config, paths, seeding, device resolution | — |
| `conditioning.py` | Generic `ConditionField` → raw vector machinery, resource-agnostic | — |
| `registry.py` | How a resource type registers itself with `sdk.py` | — |
| `sdk.py` | The public fluent API | — |
| `cli.py` | `enerdatasynth` console script (thin argparse wrapper, no new logic) | — |

## What should not be changed casually

- `diffusion/ddpm.py`'s reverse sampler — recomputing the noise prediction at every step
  is load-bearing; a historical bug here silently corrupted results for weeks before
  being caught (see `docs/research_status.md` §E). A regression test
  (`tests/test_diffusion.py`) guards this.
- `generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC` / `COND_DIM` — changing the
  conditioning schema changes what any existing checkpoint expects as input; there's a
  runtime assertion tying them together for exactly this reason.
- `evaluation/metrics.py`'s `compute_metrics` — this is the single canonical
  implementation replacing four previously-inconsistent versions across research
  history; don't reintroduce a second version elsewhere.

## Where the scientific provenance lives

`../../data/research_history/` (frozen historical scripts/reports) and
`../../docs/research_status.md` / `results.md` / `experiments.md` (the synthesized,
traceable version of the same evidence).
