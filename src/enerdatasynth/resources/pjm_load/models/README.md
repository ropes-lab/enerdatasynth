# `enerdatasynth.resources.pjm_load.models`

The denoiser architectures for PJM load. Wind's denoiser is separate code in [`resources/wind/models.py`](../../wind/README.md). The two are independent implementations behind one shared interface, for reasons given in [`docs/architecture.md`](../../../../../docs/architecture.md).

Architectures are picked by name through `build_denoiser(architecture, ...)` instead of importing a class, so training, sampling and the public API never need to know a constructor's signature.

## Files

- **`transformer.py`**: `TransformerDenoiser`, the winner of the Week 6 architecture comparison and the current `DEFAULT_ARCHITECTURE`. It uses per-hour tokens, a learned positional embedding and a standard `nn.TransformerEncoder`. The module docstring is honest about which conditioning size it was validated against; read it before assuming a bundled checkpoint uses this architecture, because most don't (`docs/research_status.md` §D).
- **`denoiser.py`**: `SimpleMLPDenoiser`, the original baseline (`architecture="mlp"`), and `SinusoidalTimeEmbedding`, which both architectures share and which has no learnable parameters.

## Public API

`build_denoiser`, `TransformerDenoiser` and `SimpleMLPDenoiser` are re-exported at the top level. Building either class directly is a Tier 2 (research) pattern. Most callers should use `enerdatasynth.model(...)` (Tier 1).

## Be careful with

The submodule names in `TransformerDenoiser` (`cond`, `val`, `pos`, `enc`, `out`) match the original research script exactly, so old checkpoints still load through `load_state_dict`. Rename them and every existing checkpoint fails with a missing-key or shape error that doesn't say why.
