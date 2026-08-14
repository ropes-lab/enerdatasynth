# `enerdatasynth.resources.pjm_load.diffusion`

**PJM load's diffusion process specifically** — wind's forward/reverse process and
sampler are independent code in
[`src/enerdatasynth/resources/wind/diffusion.py`](../../wind/README.md) /
`samplers.py` (DDIM with guidance, a different choice for a different pipeline — see
[`docs/architecture.md`](../../../../../docs/architecture.md) for the comparison and why
they're separate).

The DDPM forward/reverse process here. Standard DDPM only — no custom losses, no
Fourier/spectral terms, no moment-matching terms, by design.

## Files

- **`ddpm.py`** — `DDPM` class: `q_sample` (forward noising), `training_loss` (plain
  MSE), `p_sample_step`/`sample`/`generate_batch` (the corrected ancestral reverse
  sampler). **This is load-bearing research infrastructure** — see below.
- **`schedules.py`** — `make_betas(schedule, ...)`: linear/cosine/quadratic beta
  schedule construction, extracted from the Week 6 noise-schedule experiment so any
  caller can build the non-default schedules without re-deriving the math. `DDPM`
  itself only builds a linear schedule internally.

## What should not be changed casually

`DDPM.p_sample_step` recomputing `eps_hat = model(x_t, t_vec, z)` fresh at every reverse
step is the exact fix for a historical bug: an earlier sampler reused a noise prediction
across steps and silently corrupted every generated profile for weeks before it was
caught (`docs/research_status.md` §E; before/after numbers in `docs/results.md` §5's
sibling section). `tests/test_diffusion.py::test_sampler_recomputes_noise_prediction_every_step`
and `::test_sampler_noise_prediction_differs_across_steps` exist specifically to catch a
reintroduction of this bug — run them (or trust CI to) before touching this file.

## Where the scientific provenance lives

`data/research_history/pjm_load/scripts/sampler_bugfix.py` — the actual diagnosis-and-fix script,
preserved verbatim (does not run as-is; see that directory's README for why).
