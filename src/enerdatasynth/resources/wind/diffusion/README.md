# `enerdatasynth.resources.wind.diffusion`

Wind's diffusion process. Load's version is separate code in
[`src/enerdatasynth/resources/pjm_load/diffusion/`](../../pjm_load/diffusion/README.md)
and uses a corrected ancestral DDPM sampler instead.
[`docs/architecture.md`](../../../../../docs/architecture.md) compares the two.

## Files

- **`diffusion.py`:** `DiffusionSchedule`, the forward and reverse process: `T=200` steps
  with a cosine beta schedule (`beta_start=1e-4, beta_end=0.02`). Load uses a linear
  schedule.
- **`samplers.py`:** DDIM sampling with classifier-free guidance and dynamic
  thresholding, tuned for this pipeline (`docs/models/wind.md` §5 and §8).

## What should not be changed casually

The `recommended_generation` settings for each preset in `../manifest.json` (`eta`,
`guidance_scale`, `dynamic_threshold`) were chosen from measurements. `eta=1.0` helps
`1ch` and `3ch` a lot but makes `2ch`'s `high` regime worse. Check
`docs/models/wind.md` §8 before changing a default here.
