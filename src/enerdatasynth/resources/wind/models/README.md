# `enerdatasynth.resources.wind.models`

Wind's denoiser. Load's transformer and MLP denoisers are separate code in
[`src/enerdatasynth/resources/pjm_load/models/`](../../pjm_load/models/README.md). The
two are independent implementations behind one shared interface;
[`docs/architecture.md`](../../../../../docs/architecture.md) explains why.

## Files

- **`models.py`:** the transformer denoiser (`hidden_dim=128, num_heads=4` by default),
  vendored from the upstream `wind_diffusion` project. Wind has only this one
  architecture, with no MLP baseline.

## Public API vs. internal

Not exported at the top level. Import `enerdatasynth.resources.wind.models.models`
directly if needed; `enerdatasynth.model("wind", ...)` doesn't require it.

## What should not be changed casually

This is vendored code from a separate project (`docs/models/wind.md` §5). The bundled
checkpoints in `data/checkpoints/wind/` were trained on this exact architecture, so
changes here can break them.
