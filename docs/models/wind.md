# Wind Model

`enerdatasynth.model("wind", zone=<preset>)` — synthetic 24-hour (288-step, 5-minute
resolution) wind profiles, conditioned on calendar, temperature, and wind-regime
context, across three channel presets (`1ch`/`2ch`/`3ch`).

This report is the scientific "report card" for this resource, structured to parallel
[`docs/models/pjm_load.md`](pjm_load.md) — the underlying science is different (a
separately-developed diffusion pipeline, vendored rather than built in-house), so the
content below is wind-specific even where the section headings match.

## 1. Overview

Generates synthetic wind profiles at 5-minute resolution over a 24-hour window (288
steps), for one of three channel presets: windspeed only (`1ch`), wind-vector U/V
components (`2ch`), or windspeed plus sin/cos-encoded direction (`3ch`). Conditioned on
month, day-of-week, a derived temperature-spread feature, and a wind-regime bucket
(low/mid/high). **Like the load resource, this is a conditional generator, not a
forecaster.**

## 2. Input Data

**No raw wind training data exists in this repository.** `data/raw/wind/`
contains only a placeholder README. The bundled checkpoints (§5) were trained elsewhere,
on data this repository does not include and cannot reproduce preprocessing for.
`h5py`/`h5pyd`/`python-dotenv` are declared dependencies (suggesting an HSDS/NREL WIND
Toolkit-style original data source) but have **zero actual usage** anywhere in
`src/enerdatasynth/resources/wind/` — the original data-acquisition step was never vendored into this
package. This is the single largest gap in the wind resource relative to the PJM load
resource, which does ship its raw data pipeline.

What *is* documented, from the vendored package's own code and manifest:

| Item | Value |
|---|---|
| Sequence length | 288 steps |
| Frequency | 5-minute |
| Feature columns | `windspeed_100m` (`1ch`); `windspeed_100m`, `winddirection_100m` (`2ch`/`3ch`, decomposed differently per preset) |

## 3. Data Representation

One generated sample is a `(288, n_channels)` array at the vendored-package level
(`n_channels` = 1/2/2 for `1ch`/`2ch`/`3ch`). The public `.sample()` interface always
returns `(N, 288, 2)` regardless of preset: `[..., 0]` = windspeed (m/s), `[..., 1]` =
direction (degrees) — a fixed, resource-level output contract, independent of the
underlying preset's internal channel count.

## 4. Conditioning

Config-level default conditioning fields (`enerdatasynth.resources.wind.config.WindDiffusionConfig.condition_on`):
`month`, `day_of_week`, `temp_min_max_spread` (a derived, jointly z-scored transform,
not one column per raw input), and `wind_regime_bucket` (checkpoint-level choice of
one-hot or ordinal encoding — one-hot is the default, kept for backward compatibility;
an ordinal alternative exists specifically because one-hot's `mid` bucket has never
achieved better than chance-level steering accuracy, see §10).

**This does not fit the `ConditionField`-based system `pjm_load` uses** — which fields
exist is checkpoint-dependent, one field is a derived joint transform, and the regime
encoding is itself a checkpoint-level choice. `ResourceModel` supports a
`condition_builder` escape hatch for exactly this case; wind is the only resource using
it today. There is no separate "historical vs. package" conditioning split for wind the
way load has one (§4 of `docs/models/pjm_load.md`) — wind has always used one
conditioning convention, defined by `wind_diffusion` itself.

## 5. Generative Model

A conditional DDPM, implemented in the vendored `wind_diffusion` package
(`src/enerdatasynth/resources/wind/`, from a separate Data-For-Good research project) — not built or
trained by this project. Architecture: Transformer-style denoiser,
`hidden_dim=128, num_heads=4` (config defaults). Diffusion: `T=200` steps, **cosine**
beta schedule (`beta_start=1e-4, beta_end=0.02`) — a different, independently-chosen
schedule from load's linear default; not forced to match load's choice. Sampling: DDIM
with classifier-free guidance and dynamic thresholding, **not** the ancestral sampler
load uses — a materially different sampling algorithm, appropriate to this pipeline's
own tuning (§8).

## 6. Training

Not performed by this project — training happened upstream, in the source
`wind_diffusion` project, against data this repository doesn't include (§2). The
manifest records the training recipe behind each bundled checkpoint:

- **`1ch`/`3ch`**: Post-LN transformer (`norm_first=False`), `hidden_dim=128`,
  `warmup_epochs=8`, 60 epochs, `cond_dropout_prob=0.3, cond_dropout_scope="wind_regime"`.
- **`2ch`**: the same recipe, plus `joint_consistency_channels=[0,1], joint_consistency_weight=1.0`
  — a non-default recipe, specific to fixing `2ch`'s originally-broken behavior (§8).

`resources/wind/manifest.json` marks every preset's `status` as `"pending"` (its own
external-download mechanism has nothing published) — **this does not mean wind is
unavailable in this package.** This package's own convention (`data/checkpoints/wind/<preset>/`)
bundles real, working checkpoints for all three presets directly, checked before falling
back to the manifest's (empty) download path. See
[`src/enerdatasynth/generators/README.md`](../../src/enerdatasynth/generators/README.md).

## 7. Evaluation

A different metric suite from load's (`enerdatasynth.resources.wind.evaluation.metrics`), all lower-is-better
except `regime_accuracy` (higher is better):

| Metric | What it measures | Why it matters |
|---|---|---|
| `marginal_wasserstein` | Wasserstein-1 distance between real/synthetic windspeed value distributions (m/s) | Overall value-distribution fidelity |
| `acf_rmse` | RMSE between mean real/synthetic autocorrelation curves | Temporal persistence structure, not just marginal values |
| `spectral_low_band` / `spectral_high_band` | Log-ratio of average power spectra, split into low/high frequency bands | Whether generated turbulence has the right frequency content, not just the right autocorrelation |
| `ramp_wasserstein` | Wasserstein-1 distance between real/synthetic within-day first-difference distributions | Ramp/volatility fidelity, wind's analogue of load's `Ramp` |
| `diurnal_rmse` | RMSE between mean daily profile curves | Whether the average diurnal cycle shape is right |
| `regime_accuracy` | Fraction of generated days whose actual mean windspeed lands in the *requested* regime bucket | Whether conditioning on `wind_regime` actually controls output, not just whether output looks plausible |

## 8. Quantitative Results

Real numbers from `src/enerdatasynth/resources/wind/manifest.json`, measured 2026-07-21 against each
preset's own validation split (100 synthetic realizations per regime):

**`1ch`** (best-tuned preset — `sampler=ddim, steps=50, eta=1.0, guidance_scale=40.0`):

| Regime | n (real) | `marginal_wasserstein` | `acf_rmse` | `regime_accuracy` |
|---|---|---|---|---|
| Low | 156 | 2.25 | 0.060 | 0.69 |
| Mid | 300 | 2.00 | 0.108 | **0.41 (near chance)** |
| High | 147 | 2.11 | 0.101 | 0.55 |

**`3ch`** (`sampler=ddim, steps=50, eta=1.0, guidance_scale=40.0`):

| Regime | `marginal_wasserstein` | `acf_rmse` | `regime_accuracy` |
|---|---|---|---|
| Low | 2.67 | 0.074 | 0.41 |
| Mid | 2.21 | 0.109 | **0.43 (near chance)** |
| High | 1.16 | 0.118 | 0.82 |

**`2ch`** requires different generation settings (`eta=0.0, guidance_scale=1.0` — guidance
measurably breaks this preset even with dynamic thresholding on) and a non-default
training recipe (joint-consistency loss). It was originally measured badly broken
(~2.8× inflated windspeed, effectively no regime control) and has since improved
substantially with the joint-consistency fix, but **is not treated as equivalent to
`1ch`/`3ch`** — its own manifest entry explicitly says so. Exact current per-regime gate
values for `2ch` were not re-extracted for this documentation pass; see
`src/enerdatasynth/resources/wind/manifest.json` directly for the authoritative numbers.

**The one problem nothing fixed, across every preset:** `mid`-regime steering accuracy
never exceeded ~41–43% — barely above the one-third no-conditioning chance floor for a
3-way bucket, let alone a meaningfully-above-chance number. `low` and `high` both
improved substantially with generation-setting tuning (`eta=1.0` over DDIM's default
`eta=0.0`); `mid` did not move.

## 9. Actual vs. Generated Examples

**Cannot be generated from this repository** — no real wind observations exist
anywhere in it (§2), so there is nothing here to compare a generated profile against
live, the way §9 of [`docs/models/pjm_load.md`](pjm_load.md) does. Unlike that section,
what follows is **not** generated by this package or for this documentation pass.

The figures below are copied, unmodified, from the upstream `wind_diffusion` project's
own evaluation output (`presentation_plots/real_vs_synthetic/{low,mid,high}/` in the
source Data-For-Good repository), generated via that project's
`wind_diffusion.plotting.compare_to_real` against **its own** real held-out validation
split — the split the `1ch` checkpoint lineage bundled here was evaluated on upstream
(not trained on). The exact generation settings used for these plots (sampler, `eta`,
`guidance_scale`, `dynamic_threshold`) are **not recorded in the images and cannot be
verified from this repository**; do not assume they match the manifest's
`recommended_generation` (§8). They are included for reference because they are the
only real-vs-generated wind evidence that exists anywhere in this checkpoint's
lineage, not because this repository can reproduce or independently verify them.

| low | mid | high |
|---|---|---|
| ![low regime: real vs. synthetic envelope](../assets/wind_upstream_real_vs_synthetic_low_envelope.png) | ![mid regime: real vs. synthetic envelope](../assets/wind_upstream_real_vs_synthetic_mid_envelope.png) | ![high regime: real vs. synthetic envelope](../assets/wind_upstream_real_vs_synthetic_high_envelope.png) |

Each panel shows, for 100 generated realizations in that regime, the synthetic mean
(orange line) and 2.5-97.5 percentile band (shaded), against the **mean** of the real
validation days in that regime (blue line — `compare_to_real` plots the real-day mean,
not a single real day). The x-axis date is a placeholder anchor, not a real calendar day.
How to read them:

- **The percentile band is uninformative.** In all three panels it runs flat from
  0 m/s to ~26.6 m/s for the entire day. Those are exactly the sampler's output bounds:
  26.6 m/s = `target_mean + sample_clip_std × target_std` = 9.31 + 3.5 × 4.95
  (`data/checkpoints/wind/1ch/norm_stats.json`, `config.py`'s `sample_clip_std=3.5`), and
  0 m/s is the non-negativity clip applied on the way back to physical units
  (`data/transforms.py`). A band pinned at both bounds at every timestep means at
  least a few of the 100 realizations sit at each bound throughout the day — the
  clip-boundary lock-in documented in `diffusion.py`'s `clamp_x0`. It says nothing
  about how well generated spread matches real spread.
- **Only the mean curves carry signal.** `mid`'s synthetic mean tracks the real mean
  within ~1 m/s for most of the day. `low`'s synthetic mean sits ~2-3 m/s *above* the
  real mean all day, in the same direction as the manifest's `regime_confusion` for
  `low` (31% of `low`-conditioned days land in the `mid` or `high` bucket). `high`'s synthetic mean dips ~3 m/s below the real mean
  in the afternoon.
- A close mean curve does not mean the regime label was honored: `mid`'s mean looks
  best here, yet its `regime_accuracy` is only ~41% (§8). Averaging over realizations
  can hide a mix of too-low and too-high days.

Once raw wind data is added to (or connected to) this repository, the same live
methodology used for load — matched conditioning, generated from *this package's*
actual checkpoint at its recommended settings, for this documentation pass — should
replace these borrowed figures.

## 10. Conditional / Regime Evaluation

Covered in §8's per-regime tables — `wind_regime` (low/mid/high) is the primary
conditioning slice this pipeline evaluates against, analogous to load's temperature/
calendar/season slices. The `mid` regime is the resource's clearest and most persistent
weak point (§8), directly comparable in kind (though not in cause) to load's DOM-zone
and hot-day weaknesses.

## 11. Multi-Zone / Multi-Site Results

Not applicable in the geographic sense load's 17-zone study covers — wind's three
"presets" (`1ch`/`2ch`/`3ch`) are channel-representation variants of the same
underlying site/dataset, not different geographic sites. No multi-site wind study
exists in this repository.

## 12. Calibration

**No calibration step exists for wind**, unlike load's scalar bias correction. Each
preset's `recommended_generation` settings (sampler/steps/eta/guidance) function as a
form of tuned generation-time correction, but this is a different mechanism from load's
post-hoc additive calibration — not a gap, just a different pipeline with no equivalent
step.

## 13. Limitations

1. **No raw wind data or acquisition pipeline exists in this repository** — the largest
   gap relative to load. Checkpoints work; nothing here can retrain or independently
   revalidate them against ground truth.
2. **`mid`-regime conditioning is unreliable** across every preset and every generation
   setting tried (§8) — a real, unresolved scientific limitation, not a documentation
   gap.
3. **`2ch` is not equivalent to `1ch`/`3ch`** — improved but explicitly not validated to
   the same standard.
4. `resources/wind/manifest.json`'s own `status: "pending"` fields could mislead a
   reader into thinking no checkpoint is available; this package's local-checkpoint
   convention supersedes that status for all three presets (§6).
5. No real-vs-generated visual evidence **generated by or for this repository**
   exists (§9). The envelope figures shown there are borrowed unmodified from the
   upstream project's own evaluation output, with unrecorded generation settings, and
   their percentile bands are saturated at the sampler's clip bounds (0 / ~26.6 m/s),
   so only their mean curves are informative. They are not reproduced or independently
   verified here.
6. `units="normalized"` is not supported for any preset — only `"m/s"` — a permanent
   interface gap in the vendored evaluation code, not prioritized for a fix.
7. **Generation is CPU-slow and scales poorly with batch size**, verified directly
   during this documentation pass: `n=2` completes in ~6s, `n=10` in ~9s, but `n=100`
   did not finish within 2.5 minutes on CPU (`1ch`, default DDIM settings). Request
   small batches, or expect a long wait for large ones — this is not a hang.

## 14. Reproduction

**Not reproducible from this repository alone.** There is no raw data to preprocess and
no training entry point exposed through `enerdatasynth`'s own CLI for wind (unlike
`pjm_load`'s `enerdatasynth train`). The bundled checkpoints can be used for generation
immediately:

```python
import enerdatasynth
wind = enerdatasynth.model("wind", zone="1ch")   # or "2ch" / "3ch"
wind.set(wind_regime="mid", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=10, seed=0)              # (10, 288, 2): m/s, degrees
```

Retraining or re-evaluating against real data would require obtaining the original
`wind_diffusion` project's dataset and its own training code (`src/enerdatasynth/resources/wind/train.py`).
The package's own error messages reference a `wind-diffusion-train --preset <preset>
--data-path <csv> --output-dir <dir>` command (`resources/wind/registry.py`), but **this
is not registered as an installable console script by this package's `pyproject.toml`**
— it describes `enerdatasynth.resources.wind.training.train`'s intended invocation from the upstream project,
not a command this repository currently exposes. See [`docs/api_reference.md`](../api_reference.md)
for the full generation-side parameter reference.
