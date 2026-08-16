# `enerdatasynth.resources.pjm_load.diffusion`

PJM load's diffusion process specifically. Wind's forward/reverse process and sampler are separate code, in [`src/enerdatasynth/resources/wind/diffusion.py`](../../wind/README.md) and `samplers.py` (DDIM with guidance — a different choice for a different pipeline; see [`docs/architecture.md`](../../../../../docs/architecture.md) for the comparison).

This is the DDPM forward/reverse process for load: standard DDPM only, no custom losses, no Fourier or spectral terms, no moment-matching. That's by design.

## Files

- **`ddpm.py`** — the `DDPM` class: `q_sample` for forward noising, `training_loss` (plain MSE), and `p_sample_step`/`sample`/`generate_batch` for the corrected ancestral reverse sampler. This is load-bearing research infrastructure — see below.
- **`schedules.py`** — `make_betas(schedule, ...)`, linear/cosine/quadratic beta schedule construction pulled out of the Week 6 noise-schedule experiment so a caller can build a non-default schedule without re-deriving the math. `DDPM` itself only builds a linear schedule internally.

## What should not be changed casually

`DDPM.p_sample_step` recomputes `eps_hat = model(x_t, t_vec, z)` fresh at every reverse step, and that's the exact fix for a real historical bug: an earlier sampler reused one noise prediction across steps and silently corrupted every generated profile for weeks before anyone caught it (`docs/research_status.md` §E has the story; before/after numbers are in `docs/results.md` §5). `tests/pjm_load/test_diffusion.py::test_sampler_recomputes_noise_prediction_every_step` and `::test_sampler_noise_prediction_differs_across_steps` exist specifically to catch this coming back. Run them before touching this file.

## Where the scientific provenance lives

`data/research_history/pjm_load/scripts/sampler_bugfix.py` is the actual diagnosis-and-fix script, preserved verbatim. It doesn't run as-is — see that directory's README for why.
