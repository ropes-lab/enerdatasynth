# Architecture

## Unified interface, resource-specific implementations

Both resources — PJM load and wind — are conditional diffusion models exposed through
the same interface: `enerdatasynth.generators.BaseGenerator` (`domain` attribute +
`.sample()`), wrapped by the fluent public API
(`enerdatasynth.model()`/`.set()`/`.sample()`, `src/enerdatasynth/sdk.py`). A resource
registers itself with `enerdatasynth.registry.register_resource(...)` at import time
(`generators/load_conditioning.py`, `generators/wind_conditioning.py`) — `sdk.py` never
hardcodes a per-resource dispatch table, it only reads from `registry.get_registry()`.
This is the layer that makes the product genuinely one thing: adding a third resource
means registering it the same way, not forking the SDK.

**Below that shared interface, the two resources are implemented completely
independently, on purpose:**

| | PJM load | Wind |
|---|---|---|
| Implementation | Built for this project (`src/enerdatasynth/{models,diffusion,training,sampling,evaluation,data}/`) | Vendored from a separate, previously-published Data-For-Good research project (`src/enerdatasynth/resources/wind/`) |
| Denoiser | Transformer / MLP, this project's own code | Its own Transformer-style denoiser, independent code |
| Sampler | Corrected ancestral DDPM (this project's own fix — see below) | DDIM with classifier-free guidance, independent tuning |
| Evaluation | This project's 7-metric suite | Its own fidelity-gate metric suite (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, ...) |
| CLI / from-scratch training | `enerdatasynth train`/`sweep`/`calibrate`, etc. | None — ships pre-trained, bundled checkpoints instead |

**This is a deliberate design choice, not an accidental seam.** Wind is a real,
independently-developed pipeline this project integrates rather than reimplements —
forcing it to share load's DDPM/Transformer/metrics code would mean rewriting a working
pipeline for no scientific benefit. The product-level contract
(`model()`/`.set()`/`.sample()`, `BaseGenerator`) is what's unified; the two model
report cards ([`docs/models/pjm_load.md`](models/pjm_load.md),
[`docs/models/wind.md`](models/wind.md)) document each implementation on its own terms.
See [`docs/extending.md`](extending.md) for what registering a *third* resource would
involve, and `src/README.md` / `src/enerdatasynth/resources/wind/README.md` for the code-level split.

**The rest of this page documents PJM load's implementation specifically** — the
`enerdatasynth.resources.pjm_load.models`/`diffusion`/`training`/`sampling`/`evaluation`/`data`
subpackages named below all belong to load, not to the product generally. For wind's
architecture (a genuinely different pipeline), see
[`docs/models/wind.md`](models/wind.md) §5–§7 instead of looking for it here.

---

## PJM load: overview

A conditional DDPM (Denoising Diffusion Probabilistic Model) generates
a 24-hour normalized load profile `x` conditioned on a vector `z`
(month, day-of-week, holiday flag, temperature, daily temperature
spread). Forward and reverse processes live in
`enerdatasynth.resources.pjm_load.diffusion.ddpm.DDPM`; the denoiser network is selected
by `enerdatasynth.resources.pjm_load.models.build_denoiser(architecture, ...)`.

## PJM load: denoiser — transformer (default)

`enerdatasynth.resources.pjm_load.models.transformer.TransformerDenoiser` treats each of
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
`data/research_history/pjm_load/reports/week6/arch/summary_report.md` (best
generated-profile RMSE across MLP/CNN/attention/transformer,
controlled comparison) and
`data/research_history/pjm_load/reports/week7/summary_report.md` (beats the MLP
baseline on 6/7 metrics at ~6x fewer parameters: 110,401 vs 688,408).

## PJM load: denoiser — MLP (legacy baseline)

`enerdatasynth.resources.pjm_load.models.denoiser.SimpleMLPDenoiser` concatenates the
noisy profile, a sinusoidal timestep embedding, and the conditioning
vector, and passes the result through `n_layers` fully-connected SiLU
layers. No residual connections, no attention. Available via
`architecture="mlp"` for comparison; not recommended for new work.
**This is also what the package currently loads by default** for
`pjm_load` — see [`docs/models/pjm_load.md`](models/pjm_load.md) §5 for why.

## PJM load: diffusion process

`enerdatasynth.resources.pjm_load.diffusion.ddpm.DDPM`: standard forward noising
(`q_sample`), standard MSE training loss (`training_loss`), and
**ancestral** reverse sampling (`sample` / `generate_batch`) — the
model recomputes its noise prediction at every one of `T` reverse
steps.

An earlier DDIM-style sampler (reusing one noise prediction across
multiple steps) was tried in research and found to silently corrupt
every result it touched — values pinned near clamp bounds, growing
out-of-range as step count increased. See
`data/research_history/pjm_load/scripts/sampler_bugfix.py` for the diagnosis and
fix. The corrected ancestral sampler is the *only* sampler in the
maintained package; the buggy DDIM path was never carried forward.
(Wind's sampler is DDIM — a different, independently-tuned choice for
a different pipeline; see the comparison table above.)

## PJM load: beta schedule

Linear (`beta_start=1e-4`, `beta_end=0.02`) is the validated default.
Cosine and quadratic alternatives are implemented in
`enerdatasynth.resources.pjm_load.diffusion.schedules.make_betas` but are **not** the
recommended choice — a controlled noise-schedule experiment found
linear had both the lowest validation loss and the best generated
RMSE (`data/research_history/pjm_load/reports/week6/noise/summary_report.md`).

## PJM load: post-processing — calibration

The trained denoiser reproduces daily *shape* well but carries a
small, systematic per-hour level offset that compounds into a large
daily-energy error when summed over 24 hours. A scalar correction,
fit on the training set only, removes ~66% of that error at zero cost
to shape metrics. See `enerdatasynth.resources.pjm_load.evaluation.calibration` and
`docs/evaluation.md`. (Wind has no equivalent calibration step — its
per-preset `recommended_generation` settings serve a similar tuning
role in a different way; see `docs/models/wind.md` §12.)

## PJM load: conditioning — an open question, not silently resolved

`enerdatasynth.resources.pjm_load.data.preprocess.build_conditioning_vector` produces a
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
