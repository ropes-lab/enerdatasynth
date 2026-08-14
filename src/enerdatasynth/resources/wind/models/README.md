# `enerdatasynth.resources.wind.models`

**Wind's denoiser architecture specifically** — load's Transformer/MLP denoisers are
independent code in
[`src/enerdatasynth/resources/pjm_load/models/`](../../pjm_load/models/README.md). The
two are deliberately separate implementations behind one shared interface; see
[`docs/architecture.md`](../../../../../docs/architecture.md) for why.

## Files

- **`models.py`** — the Transformer-style denoiser (`hidden_dim=128, num_heads=4`
  config defaults), vendored from the upstream `wind_diffusion` project. Unlike load,
  there's only one architecture here — no MLP baseline to compare against.

## Public API vs. internal

Not re-exported at the top level — reached via `enerdatasynth.resources.wind.models.models`
directly, or (for everyday use) not at all: `enerdatasynth.model("wind", ...)` never
needs to touch this module by name.

## What should not be changed casually

This is vendored code from a separately-developed, previously-published project — see
`docs/models/wind.md` §5. Changing it risks breaking compatibility with the bundled
checkpoints (`data/checkpoints/wind/`), which were trained against this exact
architecture.
