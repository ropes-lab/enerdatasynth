# `enerdatasynth.resources.pjm_load.models`

PJM load's denoiser architectures specifically. Wind's denoiser is separate code, in [`src/enerdatasynth/resources/wind/models.py`](../../wind/README.md) — the two are deliberately independent implementations behind one shared interface. See [`docs/architecture.md`](../../../../../docs/architecture.md) for why.

Architectures get selected by name through `build_denoiser(architecture, ...)` rather than importing a specific class directly, so training, sampling, and the public API never need to special-case a constructor signature.

## Files

- **`transformer.py`** — `TransformerDenoiser`, the Week 6 architecture-comparison winner and current `DEFAULT_ARCHITECTURE`. Per-hour tokenization, a learned positional embedding, and a standard `nn.TransformerEncoder`. Its module docstring has an honest caveat about which conditioning dimensionality it was actually validated against — read that before assuming a bundled checkpoint uses this architecture, because most don't (see `docs/research_status.md` §D).
- **`denoiser.py`** — `SimpleMLPDenoiser`, the original baseline (`architecture="mlp"`), plus `SinusoidalTimeEmbedding`, shared by both architectures and has no learnable parameters.

## Public API vs. internal

`build_denoiser`, `TransformerDenoiser`, and `SimpleMLPDenoiser` are all re-exported at the top level. Instantiating either class directly is a Tier-2/research pattern — most callers should just go through `enerdatasynth.model(...)` (Tier 1).

## What should not be changed casually

`TransformerDenoiser`'s submodule attribute names (`cond`, `val`, `pos`, `enc`, `out`) match the original research script exactly, so old checkpoints stay loadable via `load_state_dict`. Rename them and every existing checkpoint breaks silently — a shape-mismatch or missing-key error, not a helpful one.
