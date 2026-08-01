# Architecture

## Overview

A conditional DDPM (Denoising Diffusion Probabilistic Model) generates
a 24-hour normalized load profile `x` conditioned on a vector `z`
(month, day-of-week, holiday flag, temperature, daily temperature
spread). Forward and reverse processes live in
`pjm_diffusion.diffusion.ddpm.DDPM`; the denoiser network is selected
by `pjm_diffusion.models.build_denoiser(architecture, ...)`.

## Denoiser: transformer (default)

`pjm_diffusion.models.transformer.TransformerDenoiser` treats each of
the 24 hours as one token:

1. A learned per-hour positional embedding.
2. A shared projection of `(timestep embedding, conditioning vector)`,
   broadcast to every token.
3. A linear embedding of that hour's noisy value.
4. Sum the three, run through a standard `nn.TransformerEncoder`
   (2 layers, 4 heads, dim_feedforward = 4x model_dim, GELU).
5. A final linear layer maps each token back to a scalar noise
   prediction.

This is the architecture research found best — see
`research_history/reports/week6/arch/summary_report.md` (best
generated-profile RMSE across MLP/CNN/attention/transformer,
controlled comparison) and
`research_history/reports/week7/summary_report.md` (beats the MLP
baseline on 6/7 metrics at ~6x fewer parameters: 110,401 vs 688,408).

## Denoiser: MLP (legacy baseline)

`pjm_diffusion.models.denoiser.SimpleMLPDenoiser` concatenates the
noisy profile, a sinusoidal timestep embedding, and the conditioning
vector, and passes the result through `n_layers` fully-connected SiLU
layers. No residual connections, no attention. Available via
`architecture="mlp"` for comparison; not recommended for new work.

## Diffusion process

`pjm_diffusion.diffusion.ddpm.DDPM`: standard forward noising
(`q_sample`), standard MSE training loss (`training_loss`), and
**ancestral** reverse sampling (`sample` / `generate_batch`) — the
model recomputes its noise prediction at every one of `T` reverse
steps.

An earlier DDIM-style sampler (reusing one noise prediction across
multiple steps) was tried in research and found to silently corrupt
every result it touched — values pinned near clamp bounds, growing
out-of-range as step count increased. See
`research_history/scripts/sampler_bugfix.py` for the diagnosis and
fix. The corrected ancestral sampler is the *only* sampler in the
maintained package; the buggy DDIM path was never carried forward.

## Beta schedule

Linear (`beta_start=1e-4`, `beta_end=0.02`) is the validated default.
Cosine and quadratic alternatives are implemented in
`pjm_diffusion.diffusion.schedules.make_betas` but are **not** the
recommended choice — a controlled noise-schedule experiment found
linear had both the lowest validation loss and the best generated
RMSE (`research_history/reports/week6/noise/summary_report.md`).

## Post-processing: calibration

The trained denoiser reproduces daily *shape* well but carries a
small, systematic per-hour level offset that compounds into a large
daily-energy error when summed over 24 hours. A scalar correction,
fit on the training set only, removes ~66% of that error at zero cost
to shape metrics. See `pjm_diffusion.evaluation.calibration` and
`docs/evaluation.md`.

## Conditioning — an open question, not silently resolved

`pjm_diffusion.data.preprocess.build_conditioning_vector` produces a
23-dimensional one-hot conditioning vector (month one-hot ×12, day-of-
week one-hot ×7, holiday binary, Tmax/Tmin/spread z-scored). Every
validated result quoted above, however, was produced using a different,
6-dimensional *raw* conditioning vector from the research lineage
(month and day-of-week as plain integers). These encode the same
underlying information differently and are not interchangeable at the
checkpoint level. See `MIGRATION.md` §5 for the full reasoning — this
package deliberately keeps the one-hot encoding (a defensible,
documented choice) without claiming the transformer/calibration
numbers above are proven under it; no pretrained checkpoint compatible
with the current preprocessing exists yet.

## SDK / extension points

`pjm_diffusion.generators.BaseGenerator` is the common interface any
domain-specific generator implements (`LoadDiffusionModel` today). See
`docs/extending.md` for what adding a new resource type — e.g. wind —
would involve; nothing wind-specific is implemented in this package.
