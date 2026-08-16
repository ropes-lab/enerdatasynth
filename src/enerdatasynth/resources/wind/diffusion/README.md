# `enerdatasynth.resources.wind.diffusion`

Wind's diffusion process specifically. Load's forward/reverse process and sampler are separate code, in [`src/enerdatasynth/resources/pjm_load/diffusion/`](../../pjm_load/diffusion/README.md) — a corrected ancestral DDPM, a different choice for a different pipeline. See [`docs/architecture.md`](../../../../../docs/architecture.md) for the comparison.

## Files

- **`diffusion.py`** — `DiffusionSchedule`: forward/reverse process, `T=200` steps, cosine beta schedule (`beta_start=1e-4, beta_end=0.02`). This was chosen independently and isn't forced to match load's linear default.
- **`samplers.py`** — DDIM sampling with classifier-free guidance and dynamic thresholding, a materially different algorithm from load's ancestral sampler, tuned for this pipeline specifically (`docs/models/wind.md` §5/§8).

## What should not be changed casually

The per-preset `recommended_generation` settings in `../manifest.json` (`eta`, `guidance_scale`, `dynamic_threshold`) are measured-best values, not defaults picked for convenience. `eta=1.0` is a large, real win for `1ch`/`3ch`, but it actively hurts `2ch`'s `high` regime. Check `docs/models/wind.md` §8 before changing a default here.
