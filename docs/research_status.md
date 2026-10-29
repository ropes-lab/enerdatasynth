# Research Status — Source of Truth

This document is the canonical, current-as-of-audit statement of what is scientifically
established in this repository, what the installable package actually does today, and
where those two things diverge. It was produced by directly reading the code and the
result files under `data/research_history/pjm_load/` and `data/reports/pjm_load/` — every claim below is
either traced to a specific file or explicitly marked as not documented.

**If anything in the main `README.md`, `MODEL_CARD.md`, or any `docs/` page contradicts
this document, this document is the one to trust** — it is the only page written by
cross-checking prose claims against the actual code and the actual result files in the
same pass.

---

## A. Research Objective

The project generates synthetic 24-hour PJM electricity load profiles conditioned on
contextual variables (calendar and weather).

**This is not a forecasting model.** The objective is not:

> "What will tomorrow's load be?"

The objective is:

> "Can a conditional generative model reproduce the statistical, temporal, and
> operational characteristics of observed daily load profiles?"

Nothing in this repository predicts a specific future day's actual load. A generated
profile is a plausible sample conditioned on month/day-of-week/holiday/temperature — not
a point forecast, and not validated for any forecasting, dispatch, or control use.

## B. Data

| Item | Value | Source |
|---|---|---|
| Raw load data | `data/raw/pjm_load/hrl_load_metered_{2016..2025}.csv` — 10 annual files, originally sourced from PJM Data Miner 2. **Removed from this repository as of a later pass** (see `docs/data_card.md` "Licensing / Redistribution") — not bundled anymore; bring your own copy to reproduce preprocessing. | directory listing at time of original audit |
| Raw temperature data | `data/raw/pjm_load/station_hourly_temps.csv`, `pjm_hourly_temps_wide.csv` — **also removed**, same pass. | directory listing at time of original audit |
| Temperature source | NOAA Integrated Surface Database (ISD), one or more airport weather stations per PJM zone, resolved by USAF-WBAN station ID at runtime | `src/enerdatasynth/resources/pjm_load/data/temperature.py` docstring + `STATION_ICAO`/`ZONE_STATIONS` tables |
| Temperature mapping | Equal-weighted average across each zone's constituent stations (`ZONE_STATIONS` dict); handles station renumbering across years by re-resolving IDs from NOAA's inventory | `src/enerdatasynth/resources/pjm_load/data/temperature.py` |
| Temperature cache | Per-station-year Parquet cache at `data/cache/pjm_load/{station}_{year}.parquet`, so re-runs don't re-download. **Emptied in the same later pass** — regenerates automatically given network access; nothing here is bundled source data. | `data/cache/README.md` |
| Hourly resolution | 24 values/day, fixed-width | `src/enerdatasynth/resources/pjm_load/data/preprocess.py` |
| Zones with full Phase-2 research validation | 17 zones: AP, DAY, PL, DEOK, CE, PN, DPL, BC, PS, RECO, JC, ME, DUQ, PE, AE, EKPC, DOM | `data/research_history/pjm_load/reports/phase2/summary_report.md` |
| Zones exposed by the installed package today | **All 17**, one trained Transformer each (PS, AE, AP, BC, CE, DAY, DEOK, DOM, DPL, DUQ, EKPC, JC, ME, PE, PL, PN, RECO) | `src/enerdatasynth/generators/load_conditioning.py`: `register_resource(..., zones=list(PJM_ZONES), ...)`; per-zone scores in `data/reports/pjm_load/zone_results.csv` |
| Training/validation split of the dataset the shipped PS checkpoint was trained on | 2,785 train / 600 validation days (zone PS), from 3,653 raw days before cleaning (other zones: see `zone_results.csv`) | `data/reports/pjm_load/results_summary.md` |
| DST handling | Days with 23 or 25 hourly rows (detected pre-pivot via `detect_dst_days`) are flagged and removed — a fixed 24-column output can't represent them | `src/enerdatasynth/resources/pjm_load/data/preprocess.py` (`detect_dst_days`, `is_dst` flag) |
| Missing-load handling | Days with any NaN among the 24 hourly load columns (excluding days already flagged DST) are removed | `src/enerdatasynth/resources/pjm_load/data/preprocess.py` (`missing_load` flag) |
| Missing-weather handling | Days with NaN `Tmax` or `Tmin` after the zone temperature merge are removed | `src/enerdatasynth/resources/pjm_load/data/preprocess.py` (`missing_weather` flag) |
| COVID period | A separate `is_covid` flag exists and is also excluded from the cleaned set | `src/enerdatasynth/resources/pjm_load/data/preprocess.py` (`remove_mask` includes `is_covid`) — exact date range: **not verified in this audit pass** |
| Train/validation methodology | **Chronological split: 2016–2023 train / 2024–2025 validation** (not random) | `data/research_history/pjm_load/scripts/week3.py`, Step 6 |

## C. Conditioning — Critical Distinction

**Two different conditioning representations exist in this repository, and they are not
interchangeable:**

1. **Research-history conditioning ("Z5"/"Z9")** — the raw, lower-dimensional
   representation used to produce the reported Week 6/Week 7 Transformer results
   (`cond_dim=6` for "Z5"). The Week 6/7 scripts under `data/research_history/pjm_load/scripts/`
   show how it was built. The `Z5_train.npy` / `Z9_train.npy` tensors themselves were removed
   from the repository along with the rest of the data.

2. **Current package conditioning** — `src/enerdatasynth/utils/config.py`'s
   `COND_DIM = N_MONTHS + N_DOW + 1 + 1 + 1 + 1 = 23`:

   | Field | Encoding | Dims |
   |---|---|---|
   | month | one-hot (1–12) | 12 |
   | day of week | one-hot (0–6) | 7 |
   | holiday | binary | 1 |
   | Tmax | standardized (z-score) | 1 |
   | Tmin | standardized (z-score) | 1 |
   | spread (Tmax − Tmin) | standardized (z-score) | 1 |
   | **Total** | | **23** |

   Implemented generically via `ConditionField` entries in
   `src/enerdatasynth/generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC`, consumed
   by `src/enerdatasynth/conditioning.py`'s `build_condition_vector`.

**Do not assume the 23-dimensional package conditioning has reproduced any Week 6/7
Transformer result.** It has not been tested against those results in this repository.
See section D and `docs/results.md`'s "Critical Scientific Integrity Note".

**Documentation contradiction found during this audit, and resolved by checking the
actual data:** `week3.py`'s own header labels its baseline conditioning vector "Z5 matrix
(N_days × 5)" (implying 5 dimensions), while `src/enerdatasynth/resources/pjm_load/models/transformer.py`'s
docstring states the validated Transformer checkpoint used `cond_dim=6`. Loading
`data/research_history/pjm_load/outputs/week3_outputs/Z5_train.npy` directly confirmed shape
`(2785, 6)` (when that file was still in the repository; it has since been removed) — **6 columns, not 5.** "Z5" is a research-era name referring to 5 conceptual
input fields (month, day-of-week, holiday, temperature, spread) packed compactly into 6
numeric columns — not a one-hot scheme like the current package's 23-dimensional
conditioning. `transformer.py`'s `cond_dim=6` claim is correct; `week3.py`'s header
comment is the stale one. (`Z9_train.npy` is confirmed `(2784, 9)`, the experimental
richer variant mentioned in the same script.)

## D. Model — Current Recommended Architecture

`src/enerdatasynth/resources/pjm_load/models/transformer.py`'s `TransformerDenoiser`, promoted to
`DEFAULT_ARCHITECTURE = "transformer"` in `src/enerdatasynth/utils/config.py`.

Verified directly from the code (not from old reports):

| Property | Value | Source |
|---|---|---|
| Per-hour tokenization | Each of the 24 hourly values is one token | `TransformerDenoiser.forward` |
| Positional encoding | Learned parameter `self.pos`, shape `(1, x_dim, model_dim)`, initialized `N(0, 0.02²)` — not sinusoidal | `transformer.py:80` |
| Timestep embedding | `SinusoidalTimeEmbedding` (from `models/denoiser.py`), no learnable parameters | `transformer.py:40,74` |
| Conditioning injection | `z` and the timestep embedding are concatenated, passed through a 2-layer MLP (`SiLU` activation), then broadcast-added to every token alongside the positional embedding | `transformer.py:75-78,90-91` |
| Encoder | Standard `nn.TransformerEncoder`, `dropout=0.0`, `activation="gelu"`, `batch_first=True` | `transformer.py:81-85` |
| Default hyperparameters | `model_dim=64`, `num_heads=4`, `num_layers=2`, `t_embed_dim=64` | `transformer.py:67-69`, `utils/config.py` `DEFAULT_HIDDEN_DIM/DEFAULT_N_HEADS/DEFAULT_N_LAYERS` |
| Diffusion steps default | `DEFAULT_T = 1000` ("matches the validated transformer/calibration research" per code comment) | `utils/config.py:139` |
| Parameter count (Week 6 architecture-comparison run) | **110,401** — this specific number is for the Week-6 controlled comparison, which used `cond_dim=6` (Z5), not the current package's `cond_dim=23` | `data/research_history/pjm_load/reports/week6/arch/metrics.csv` |
| Optimizer / learning rate / training procedure | Not re-verified against `src/enerdatasynth/resources/pjm_load/training/train.py` in this pass beyond what's in the Results doc — **treat specific values (lr, optimizer type, batch size) as needing direct code confirmation before quoting in a paper.** | — |

**The critical caveat, already honestly disclosed in the code itself:**

> "the checkpoint this architecture was originally validated against
> (`transformer_baseline_z5.pt`) was trained with `cond_dim=6` (the research lineage's
> raw 'Z5' conditioning vector), not the 23-dimensional one-hot conditioning
> `enerdatasynth.resources.pjm_load.data.preprocess` produces, so those Week 6/7 numbers
> describe a different model from the one the package ships."
> — `src/enerdatasynth/resources/pjm_load/models/transformer.py`, module docstring

**Status: resolved on 2026-10-07.** Until then, no Transformer checkpoint compatible with the
23-dim conditioning existed. The history below is kept because several earlier results
were produced during that period.

**What this meant before the shipped Transformer, verified by actually running it:** every checkpoint
bundled under `data/checkpoints/pjm_load/` (`T100_H256_L4`, `T100_H128_L6`, `T200_H128_L4`,
`T100_H64_L4`, `T100_H128_L4`, `T100_H128_L2`, `T50_H128_L4`) has no `"architecture"` key
in its `result.json` — the historical marker for the original MLP hyperparameter sweep,
before the `architecture` field existed. `generators/load.py`'s `from_run_id` deliberately
defaults a missing `architecture` key to `"mlp"`, not to the current
`DEFAULT_ARCHITECTURE`, specifically because every checkpoint without that key really is
an MLP run. `generators/load_conditioning.py`'s `_discover_best_checkpoint()` prefers a
`"transformer"`-tagged run when one exists. None did, so it fell back to the best legacy
MLP run by validation loss (`T100_H256_L4`, 224,024 params, best val loss 0.08164).

**The fix:** a Transformer was trained against the current 23-dimensional conditioning with
the validated config (`enerdatasynth train --config data/configs/pjm_load/transformer.yaml`,
seed 42), calibrated, and added as `data/checkpoints/pjm_load/transformer_T1000_H64_L2/`.
`_discover_best_checkpoint()` now selects it, so **`enerdatasynth.model("pjm_load", zone="PS")`
loads the calibrated Transformer**; `tests/pjm_load/test_api.py` checks that the manifest
and the selection logic agree. On the 600 validation days it beats the legacy MLP on all
seven metrics in every one of 5 generation seeds, before and after calibration
(`data/reports/pjm_load/checkpoint_comparison.csv`; reproduce with
`examples/pjm_load/compare_checkpoints.py`).

**What this does not do.** The Week 7 numbers (RMSE 0.0195, Peak 0.0089, ...) are still
research-conditioning results; the shipped model's own numbers differ and are reported
separately. It is also one training run with research-validated hyperparameters rather than
a from-scratch sweep, and the checkpoint-selection rule prefers any Transformer over any MLP
because validation losses aren't comparable across diffusion-step counts.

## E. Diffusion Process

Implemented in `src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py`. Standard DDPM — no custom losses,
no Fourier/spectral terms, no moment-matching terms (verified: `training_loss` is a bare
`nn.functional.mse_loss` call).

- **Forward process:** `q(x_t | x_0) = N(√ᾱ_t · x_0, (1 − ᾱ_t) · I)`
- **Training loss:** `L = MSE(ε̂, ε)` — standard diffusion MSE only
- **Beta schedule:** linear, `beta_start=1e-4`, `beta_end=0.02`, `T` steps
  (`torch.linspace`) — confirmed the winning schedule in the Week 6 noise-schedule
  experiment (see `docs/results.md` §5)
- **Reverse sampler:** standard ancestral DDPM sampling (Ho et al., 2020), recomputing
  the noise prediction `ε̂ = model(x_t, t, z)` fresh at every reverse step
  (`ddpm.py`'s `p_sample_step`), then applying the proper posterior mean/variance update

### The sampler bug — documented honestly, not hidden

An earlier (never-shipped-in-this-package) sampler implementation reused a noise
prediction incorrectly across sampling steps rather than recomputing it at every step.
This was diagnosed and fixed in `data/research_history/pjm_load/scripts/sampler_bugfix.py`. The
concrete before/after numbers, both preserved in this repository:

| | Hourly | FirstDiff | ACF | RMSE | Peak | Energy |
|---|---|---|---|---|---|---|
| Buggy sampler ("Diffusion (converged)", pre-fix) | 0.5799 | 2.2864 | 0.0612 | 0.5069 | 0.5246 | 1.4684 |
| Corrected sampler ("Diffusion (converged)", post-fix) | 0.0436 | 0.0179 | 0.0404 | 0.0261 | 0.0119 | 0.0269 |

Source: `data/research_history/pjm_load/reports/week5_comparison_table.csv` /
`week5_metrics.json` (pre-fix) vs. `week5_comparison_corrected.csv` (post-fix).

**The corrected ancestral sampler in `src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py` is
load-bearing research infrastructure.** A regression test
(`tests/pjm_load/test_diffusion.py::test_sampler_recomputes_noise_prediction_every_step` and
`::test_sampler_noise_prediction_differs_across_steps`, added during this audit) asserts
the model is called fresh exactly once per reverse step, specifically to catch a
reintroduction of this bug. **Every benchmark number in `docs/results.md` is a post-fix
number**; the pre-fix numbers above are shown only as the historical record of the bug's
impact, not as valid results.

## F. Calibration

Implemented in `src/enerdatasynth/resources/pjm_load/evaluation/calibration.py`.

- **Bias observed:** the trained denoiser reproduces daily *shape* well (first-difference,
  ramp, ACF) but carries a small systematic per-hour level offset. Summed over 24 hours,
  that offset compounds into a large daily-energy error even though every shape metric is
  mathematically unaffected by a constant shift.
- **Why it matters:** `Energy` (one of the 7 evaluation metrics) sums all 24 hours, so a
  per-hour offset as small as ~0.006 (normalized units) compounds ~24× into the dominant
  error term for an otherwise well-shaped profile.
- **How it's fit:** `compute_bias()` measures `real.mean() − generated.mean()` (scalar
  mode) or per-hour (`per_hour` mode) on a sample of **1,200 training days — never
  validation** (`calibrate_from_samples`, `n_fit=1200`).
- **When applied:** at generation time, added back onto sampled output
  (`apply_calibration`), auto-discovered from a `calibration.json` file stored next to
  the checkpoint (`LoadDiffusionModel.__init__`).
- **What it changes:** Energy error 0.1644 → 0.0563 (scalar mode, −66%); Peak error also
  improves. **What it does not change:** FirstDiff, ACF, Ramp — mathematically invariant
  under a constant shift.
- **Why scalar, not per-hour:** the per-hour variant was tested and found *worse* on ACF
  (0.0317 → 0.0501) and RMSE (0.0210 → 0.0249) than no calibration at all. The scalar
  variant matched or improved every metric. Source:
  `data/research_history/pjm_load/reports/week7/calibration_report.md`.
- **Train-fit, not an independent test-set improvement.** The bias is measured on
  training days and applied uniformly; the reported improvement is not a validation-set
  holdout result for the calibration step itself (the *shape* metrics it doesn't touch
  were validated on the validation set, but the calibration offset itself was never
  independently re-tested on held-out days beyond the 1,200-day training sample it was
  fit on).
- **Current package status: the shipped default carries a `calibration.json`.**
  `transformer_T1000_H64_L2/calibration.json` holds a scalar offset of +0.02342, fit on 1,200
  training days (seed 42), and `LoadDiffusionModel` applies it automatically, so default
  `pjm_load` sampling is calibrated. The seven legacy MLP checkpoints have none;
  `LoadDiffusionModel` treats a missing file as "not yet calibrated" and skips correction
  silently (`load_calibration` returns `None` rather than raising).

## G. Evaluation

Seven metrics, implemented once in `src/enerdatasynth/resources/pjm_load/evaluation/metrics.py`
(`compute_metrics`), replacing four previously-incompatible ad hoc versions across the
research-history scripts.

| Metric | Definition | Interpretation |
|---|---|---|
| `Hourly` | `mean(\|mean(real,axis=0) − mean(gen,axis=0)\|) + mean(\|IQR(real,axis=0) − IQR(gen,axis=0)\|)` | Custom metric (not a standard named statistic) combining per-hour mean and interquartile-range mismatch. Lower is better. |
| `FirstDiff` | `\|std(diff(real, axis=1)) − std(diff(gen, axis=1))\|` | Mismatch in hour-to-hour volatility. Lower is better. |
| `ACF` | `mean(\|autocorrelation(real) − autocorrelation(gen)\|)`, lags 0–12 | Mismatch in temporal autocorrelation structure. Lower is better. |
| `RMSE` | `sqrt(mean((mean(real,axis=0) − mean(gen,axis=0))²))` | Root-mean-square error of the *mean* profile shape (not per-sample RMSE). Lower is better. |
| `Peak` | `\|mean(max(real,axis=1)) − mean(max(gen,axis=1))\|` | Mismatch in average daily peak. Lower is better. |
| `Energy` | `\|mean(sum(real,axis=1)) − mean(sum(gen,axis=1))\|` | Mismatch in average total daily energy. Lower is better. Directly affected by the level-offset bias calibration corrects. |
| `Ramp` | `\|mean(\|diff(real,axis=1)\|) − mean(\|diff(gen,axis=1)\|)\|` | Mismatch in average hour-to-hour ramp magnitude. Lower is better. |

**`Hourly` is a project-specific composite metric, not a standard statistical term** —
it is not called out as such anywhere else in the historical reports, so this document
makes that explicit.

All values are computed in normalized `[0,1]` load units unless otherwise noted; see
`docs/results.md` for actual reported numbers per metric.

---

## Summary of What Changed During This Audit

- Fixed hardcoded local machine paths (`/Users/...`) baked into 7 bundled checkpoints'
  `result.json` metadata (informational fields, not read by any loader, but a hygiene/
  privacy issue for open-source release). See `docs/release_audit.md`.
- Moved `statsmodels`/`matplotlib` (needed only by the classical-baseline and
  validation-plotting code paths, never on the `model().set().sample()` path) out of
  core runtime dependencies into an optional `research` extra.
- Corrected `requires-python` from `>=3.9` to `>=3.10` to match what CI actually tests
  (`3.10`/`3.12`) and what `classifiers` already declared — no Python 3.9 test coverage
  exists in this repository, so the metadata should not claim it.
- Added a regression test for the historical sampler bug (see §E above).

No model architecture, conditioning representation, diffusion process, or calibration
logic was changed. This audit is a documentation and packaging-hygiene pass, not a new
research iteration.
