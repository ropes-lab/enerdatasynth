# `enerdatasynth.resources.pjm_load.diffusion`

The diffusion process for PJM load. Wind's forward and reverse process and its sampler are separate code, in [`resources/wind/diffusion.py`](../../wind/README.md) and `samplers.py`. Wind uses DDIM with guidance, a different choice for a different pipeline; [`docs/architecture.md`](../../../../../docs/architecture.md) compares them.

Load uses plain DDPM, on purpose: no custom losses, no Fourier or spectral terms, no moment matching.

## Files

- **`ddpm.py`**: the `DDPM` class. `q_sample` adds noise going forward, `training_loss` is plain MSE, and `p_sample_step`, `sample` and `generate_batch` implement the corrected ancestral reverse sampler. The research depends on this file, so see below.
- **`schedules.py`**: `make_betas(schedule, ...)` builds linear, cosine or quadratic beta schedules. It was pulled out of the Week 6 noise-schedule experiment so a caller can build a non-default schedule without redoing the math. `DDPM` itself only builds a linear one.

## Be careful with

`DDPM.p_sample_step` recomputes `eps_hat = model(x_t, t_vec, z)` at every reverse step. That is the fix for a real bug: an earlier sampler reused a single noise prediction across steps and quietly corrupted every generated profile for weeks before anyone noticed (`docs/research_status.md` §E tells the story, and `docs/results.md` §5 has the before and after numbers). `tests/pjm_load/test_diffusion.py::test_sampler_recomputes_noise_prediction_every_step` and `::test_sampler_noise_prediction_differs_across_steps` exist to catch it coming back. Run them before touching this file.

## Provenance

`data/research_history/pjm_load/scripts/sampler_bugfix.py` is the original diagnosis and fix, kept as it was. It doesn't run as-is; that directory's README says why.
