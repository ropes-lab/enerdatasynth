# `enerdatasynth.models`

**PJM load's denoiser architectures specifically** — wind's denoiser is independent
code in [`src/wind_diffusion/models.py`](../../wind_diffusion/README.md), not this
subpackage. The two are deliberately separate implementations behind one shared
interface; see [`docs/architecture.md`](../../../docs/architecture.md) for why.

Selected by name via `build_denoiser(architecture, ...)`
rather than importing a specific class, so training/sampling/the public API never
special-case a constructor signature.

## Files

- **`transformer.py`** — `TransformerDenoiser`, the Week 6 architecture-comparison
  winner and current `DEFAULT_ARCHITECTURE`. Per-hour tokenization + learned positional
  embedding + `nn.TransformerEncoder`. See its module docstring for the honest caveat
  about which conditioning dimensionality it was actually validated against — read that
  before assuming a bundled checkpoint uses this architecture (most don't; see
  `docs/research_status.md` §D).
- **`denoiser.py`** — `SimpleMLPDenoiser` (the original baseline, `architecture="mlp"`)
  and `SinusoidalTimeEmbedding` (shared by both architectures, no learnable parameters).

## Public API vs. internal

`build_denoiser`, `TransformerDenoiser`, `SimpleMLPDenoiser` are all re-exported at the
top level. Direct instantiation of either class is a Tier-2/research-development
pattern — most callers should go through `enerdatasynth.model(...)` (Tier 1) instead.

## What should not be changed casually

`TransformerDenoiser`'s submodule attribute names (`cond`, `val`, `pos`, `enc`, `out`)
are kept identical to the original research script specifically so old checkpoints
remain loadable via `load_state_dict`. Renaming them breaks every existing checkpoint
silently (a shape-mismatched or missing-key error, not a helpful one).
