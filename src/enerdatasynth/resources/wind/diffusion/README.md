# `enerdatasynth.resources.wind.diffusion`

**Wind's diffusion process specifically** — load's forward/reverse process and sampler
are independent code in
[`src/enerdatasynth/resources/pjm_load/diffusion/`](../../pjm_load/diffusion/README.md)
(corrected ancestral DDPM, a different choice for a different pipeline — see
[`docs/architecture.md`](../../../../../docs/architecture.md) for the comparison).

## Files

- **`diffusion.py`** — `DiffusionSchedule`: forward/reverse process, `T=200` steps,
  **cosine** beta schedule (`beta_start=1e-4, beta_end=0.02`) — independently chosen,
  not forced to match load's linear default.
- **`samplers.py`** — DDIM sampling with classifier-free guidance and dynamic
  thresholding. A materially different algorithm from load's ancestral sampler,
  appropriate to this pipeline's own tuning (`docs/models/wind.md` §5/§8).

## What should not be changed casually

`recommended_generation` settings per preset in `../manifest.json` (`eta`,
`guidance_scale`, `dynamic_threshold`) are measured-best values, not defaults picked for
convenience — e.g. `eta=1.0` is a large, real win for `1ch`/`3ch` but actively hurts
`2ch`'s `high` regime. See `docs/models/wind.md` §8 before changing a default here.
