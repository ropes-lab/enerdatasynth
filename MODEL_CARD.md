# Model Card: enerdatasynth (PJM Load, Zone PS)

This card describes the model behind `enerdatasynth.model("pjm_load", zone="PS")`, in the style of a Hugging Face model card. It documents what the model was trained on, what it's validated to do, and — as importantly — what it isn't validated for. Every claim here is traceable to a specific file in this repository; see [MIGRATION.md](MIGRATION.md) for the full audit trail.

**This card covers PJM load only.** For wind (`enerdatasynth.model("wind", ...)`), see [Wind](#wind-enerdatasynthmodelwind-) below — a separate model, from a separate project, with its own separate evidence trail; nothing in the sections below (Training Data, Conditioning Variables, Evaluation, Calibration) describes it.

## Model Details

- **Task**: Conditional generation of synthetic 24-hour electric load profiles.
- **Architecture**: Conditional DDPM (Denoising Diffusion Probabilistic Model) with a transformer denoiser (2 encoder layers, 4 attention heads, model dimension 64; 110,401 parameters) — see [`docs/architecture.md`](docs/architecture.md). A simpler MLP denoiser is also available (`architecture="mlp"`) as a legacy baseline.
- **Sampler**: Ancestral DDPM reverse sampling, 1000 steps. An earlier DDIM-style sampler was found to silently corrupt results and is not used — see `data/research_history/scripts/sampler_bugfix.py`.
- **Post-processing**: A scalar per-hour bias correction ("calibration"), fit on training data, applied automatically at generation time when available.
- **Output**: One 24-value hourly load profile per input condition, in MW by default (`ResourceModel.sample(..., units="MW")`) or normalized `[0,1]` (`units="normalized"`).
- **License**: MIT.

## Training Data

- **Source**: PJM Interconnection hourly metered load data (`hrl_load_metered_*.csv`, 2016–2025) and NOAA ISD hourly temperature observations, weighted to zone-level via `src/enerdatasynth/data/temperature.py`.
- **Zone**: `PS` (PJM zone code for PSE&G, New Jersey) only. See "Supported Zones" below.
- **Date range**: 2016–2025 raw; **3,653** raw calendar days before cleaning.
- **Exclusions applied** (in order):

  | Reason | Days removed |
  |---|---|
  | DST transition days (23h/25h, incompatible with a fixed 24-step profile) | 20 |
  | Missing load readings | 0 |
  | Missing weather data | 128 |
  | COVID-19 lockdown period (March–June 2020) | 122 |
  | **Final clean days** | **3,385** |

- **Train/validation split**: Chronological, not random — **2,785 days (2016–2023)** train, **600 days (2024–2025)** validation. This is a development split, not a forecasting evaluation; the task is unconditional-on-time generation given calendar/weather conditions, not forecasting future load from past load.
- **Normalization**: Load values scaled per-hour with `MinMaxScaler`, fit on training data only. Temperature and load-spread conditioning scaled with `StandardScaler`, also fit on training data only.

Regenerate this exact dataset with `enerdatasynth preprocess`; regenerating will refresh `data/processed/metadata.json`, which is the authoritative source for the exact numbers above (pulled directly from that file, generated 2026-06-22).

## Conditioning Variables

The model generates conditioned on a 23-dimensional vector:

| Variable | Encoding | Dimensions |
|---|---|---|
| Month | One-hot | 12 |
| Day of week | One-hot | 7 |
| US federal holiday | Binary | 1 |
| Daily max temperature (Tmax, °F) | Z-scored | 1 |
| Daily min temperature (Tmin, °F) | Z-scored | 1 |
| Daily load spread (max MW − min MW) | Z-scored | 1 |

`enerdatasynth.model(...).set(...)` accepts human-readable versions of the first five (`month`, `Tmax`, `Tmin`, `weekday`/`day_of_week`, `holiday`); `spread` is optional and defaults to the training-set mean (see [`docs/api_reference.md`](docs/api_reference.md) for exactly why — it's a property of the load curve being generated, not something a caller predicting from weather alone would know in advance).

**Important caveat, not silently resolved**: the transformer architecture and calibration procedure described in "Evaluation" below were *validated* using a different, 6-dimensional raw conditioning encoding (month/day-of-week as plain integers, from `data/research_history/`), not the 23-dimensional one-hot encoding this package's preprocessing actually produces. See [`docs/architecture.md`](docs/architecture.md) for the full reasoning. This means: the architecture choice (transformer over MLP/CNN/attention) is a reasonable, well-evidenced choice under either encoding, but the *exact validated numbers* are not guaranteed to reproduce identically under the current one-hot pipeline.

## Supported Zones

Only **PS** (PSE&G, New Jersey). Requesting any other zone via `enerdatasynth.model("pjm_load", zone=...)` raises a clear error rather than silently returning wrong results. `data/research_history/scripts/zone_expansion.py` explored 17 PJM zones (best: AP, RMSE 0.0086; worst: DOM, RMSE 0.1998, cause undiagnosed) but that work was never promoted into this package's production pipeline — see [`docs/research_notes.md`](docs/research_notes.md) and [`MIGRATION.md`](MIGRATION.md) §4.

## Evaluation

7-metric suite (`Hourly`, `FirstDiff`, `ACF`, `RMSE`, `Peak`, `Energy`, `Ramp`; see [`docs/evaluation.md`](docs/evaluation.md) for exact definitions), computed identically throughout this project's history for direct comparability.

- **Architecture selection**: controlled 4-way comparison (MLP/CNN/attention/transformer), identical data/conditioning/schedule/seed. Transformer had the best generated-profile RMSE (0.0215) despite CNN having the lowest training validation loss (0.01022) — `data/research_history/reports/week6/arch/summary_report.md`.
- **Head-to-head vs. classical baselines** (under the research-validated 6-dim conditioning): beats AR/ARMA/ARIMA/SARIMA on 7 of 7 metrics; beats the old MLP baseline on 6 of 7 at ~6x fewer parameters — `data/research_history/reports/week7/summary_report.md`.
- **Under this package's own current pipeline** (23-dim one-hot conditioning, the model actually shipped today): beats every classical baseline outright on 5 of 7 metrics; ARMA edges it out on ACF, ARIMA on Ramp. See the README's [Benchmark Results](README.md#-benchmark-results) for the full table, from a real run against the full 600-day validation set.
- **Consistency across conditions**: RMSE 0.0195–0.0306 across temperature/calendar/season slices under the research-validated conditioning; hardest slice is Holiday (n=17, the smallest slice — sparse-data effect, not necessarily a real generalization failure).

## Calibration

The denoiser reproduces daily *shape* accurately but carries a small systematic per-hour level offset that compounds into a large daily-energy error when summed over 24 hours. A scalar correction — one number added to every hour, fit on the training set only — reduces energy error by ~66% at zero cost to shape metrics (a constant shift cannot change first-difference, ramp, or autocorrelation). A more elaborate per-hour correction was tested and found *worse* on ACF and RMSE than no calibration at all; this package deliberately ships the simpler, better-performing scalar correction (`data/research_history/reports/week7/calibration_report.md`; the original script's own output recorded `"mode_recommended": "scalar"`). Applied automatically by `ResourceModel.sample()` and `LoadDiffusionModel.sample()` whenever `calibration.json` exists next to a checkpoint.

## Known Limitations

- **Single zone.** Only PS; see "Supported Zones."
- **Conditioning-encoding gap.** See the caveat under "Conditioning Variables" — the exact validated numbers came from a different conditioning encoding than what's shipped today.
- **`spread` cannot be predicted from weather alone.** Defaults to the training-set mean when not explicitly set; this is a documented simplification, not a validated estimate.
- **Weekday sampling is coarse.** `.set(weekday=True)` samples a uniformly random weekday per generated profile rather than modeling which specific weekday (Monday vs. Friday) a real scenario would be; use `day_of_week` for exact control.
- **Holiday conditioning is data-sparse.** Only 17 holiday days in the validation set under the research-validated split; treat holiday-conditioned generations with more caution than others.
- **DOM zone (explored, not shipped) failed validation** (RMSE 4–10x worse than every other zone in the 17-zone exploration) with an undiagnosed cause — a reminder that this pipeline's quality is not uniformly guaranteed across arbitrary zones/utilities even where data exists.
- **No formal uncertainty quantification.** Samples are draws from the model's learned distribution; there's no calibrated confidence interval or explicit epistemic/aleatoric uncertainty decomposition.

## Intended Use

- Generating synthetic load profiles for research, education, and non-production analysis: forecasting-model stress tests, scenario planning, methods research on synthetic energy data.
- Privacy-preserving data sharing where realistic-but-not-real load shapes are sufficient (this model was not evaluated for differential-privacy or re-identification guarantees — see "Out-of-Scope Uses").
- As a reference implementation / baseline for research building on conditional diffusion models for time series.

## Out-of-Scope Uses (PJM load)

- **Grid operations, dispatch, or real-time decision-making.** This is a research artifact, not a validated operational forecasting or planning tool.
- **Zones other than PS** — not supported; see "Supported Zones."
- **Formal privacy guarantees.** Do not rely on generated profiles as a certified anonymization or differential-privacy mechanism without independent evaluation — none was performed here.
- **Extrapolation far outside the training distribution** (e.g. extreme temperatures well beyond the training range, zones with very different load characteristics) — behavior is unvalidated there.
- **Any use presented as real, historical, or utility-reported data.** Output is synthetic and must be clearly labeled as such.

---

## Wind (`enerdatasynth.model("wind", ...)`)

A separate model from a separate project (`wind_diffusion`, vendored from a Data-For-Good research repository — see `src/wind_diffusion/`), reachable through the same `BaseGenerator`/`ResourceModel` interface load uses. This section summarizes what's bundled and measured; **the full evidence trail (training data, architecture rationale, every ablation) lives in the source project's own `MODEL_CARD.md`/`API.md`, not here** — this card doesn't re-derive or re-validate any of it, only reports it.

### Model Details

- **Task**: conditional generation of synthetic 24-hour wind profiles at 5-minute resolution (288 steps), conditioned on month, day-of-week, temperature range, and a discretized wind-speed regime (low/mid/high).
- **Architecture**: hybrid dilated-Conv1D + Transformer denoiser, DDPM with DDIM sampling. Three channel presets, each a genuinely different model/checkpoint: `1ch` (windspeed only), `2ch` (windspeed as U/V vector components), `3ch` (windspeed + sin/cos direction encoding).
- **Training data**: NREL WIND Toolkit (WTK), NOW-23 Mid-Atlantic dataset, a single grid point (22532, offshore New Jersey), 2010–2020. **Single location only** — generalization to other sites is unvalidated.
- **Output**: `(N, 288, 2)` — `[..., 0]` = windspeed (m/s), `[..., 1]` = direction (degrees) — via `ResourceModel.sample()`. `units="m/s"` is the only supported unit for every preset (see [`docs/api_reference.md`](docs/api_reference.md#wind-units-only-ms-is-supported)).
- **License**: the WIND Toolkit is CC-BY — any redistribution of this model or its generated data must credit NREL WIND Toolkit as the training-data source (this requirement is inherited from the source project, not newly introduced here).

### Bundled checkpoints

All three presets ship with a real trained checkpoint in this repo (`data/checkpoints/wind/{1ch,2ch,3ch}/`) — copied from the source project's local training runs (never published there; see [`docs/extending.md`](docs/extending.md) for why this is "ad hoc bundling," not a proper release). Recipes, per the source project's own measured-best configuration:

- **`1ch`/`3ch`**: `cond_dropout_prob=0.3`, `cond_dropout_scope="wind_regime"`, `hidden_dim=128`. Generation: `sampler=ddim, steps=50, eta=1.0, guidance_scale=40.0, guidance_scope="wind_regime", dynamic_threshold=true`.
- **`2ch`**: the same, plus `joint_consistency_channels=[0,1], joint_consistency_weight=1.0` (a training-time fix for a measured ~2.8x inflated windspeed magnitude in the raw checkpoint). Generation: `sampler=ddim, steps=50, eta=0.0, guidance_scale=1.0` — **guidance must stay off and `eta` must stay `0.0`** for this preset specifically (see Known Limitations). `WindDiffusionModel` resolves each preset's correct settings automatically from `wind_diffusion`'s bundled `manifest.json`; only relevant if you're calling it directly (Tier 2) rather than through `.set()`/`.sample()`.

### Evaluation (as measured by the source project, reproduced here for reference)

Fidelity gates: real held-out validation days vs. 100 synthetic realizations per regime, comparing marginal distribution fit (Wasserstein distance), autocorrelation RMSE, and regime-steering accuracy (does the requested `wind_regime` actually come through).

**`1ch`** (`n_real_days`: low=156, mid=300, high=147):

| regime | marginal_wasserstein | acf_rmse | regime_accuracy |
|---|---|---|---|
| low | 2.25 | 0.060 | 0.69 |
| mid | 2.00 | 0.108 | 0.41 |
| high | 2.11 | 0.101 | 0.55 |

**`3ch`** (same recipe, same validation split):

| regime | marginal_wasserstein | acf_rmse | regime_accuracy |
|---|---|---|---|
| low | 2.67 | 0.074 | 0.41 |
| mid | 2.21 | 0.109 | 0.43 |
| high | 1.16 | 0.118 | 0.82 |

**`2ch`**: no equivalent clean per-regime table exists in the source manifest (`gates_by_regime` is `null` there) — only a narrative history: originally measured broken (~2.8x inflated windspeed, 100% of days landing in "high" regardless of request), substantially improved by the joint-consistency loss + dynamic thresholding above (`marginal_wasserstein` roughly 8.0/4.2/3.2 → 4.5/1.5/4.4 for low/mid/high), "now in the same range as `1ch`/`3ch`" per the source project's own assessment — but explicitly **not** treated as equivalent to them. See "Known Limitations" below.

### Known Limitations (wind)

- **"Mid" regime steering is weak for `1ch`/`3ch` (0.41/0.43 accuracy)** — this is a data property the source project root-caused, not an encoding defect: month/day-of-week/temperature alone predict low/high regimes better than chance but predict mid *worse* than chance, because mid is definitionally the non-extreme residual category. The source project's fix (regime rejection sampling) is not exposed through this package's `.set()`/`.sample()` interface today — only through the source project's own lower-level API, if you import `wind_diffusion` directly.
- **`2ch` is measurably the weakest preset.** Requires its non-default training recipe, cannot use guidance (breaks `ramp_wasserstein`, and "mid" requests resolve to "low" 83% of the time at `guidance_scale=40`), and cannot use `eta=1.0` (measured *mixed*: helps low/mid, clearly hurts high). Treat `2ch` output with more caution than `1ch`/`3ch`.
- **Single grid point.** All training/evaluation used one WTK location (offshore NJ) — generalization to other sites/climates is untested.
- **No calibration equivalent.** Unlike load's scalar bias correction, no analogous post-processing is applied to wind output.
- **Checkpoints aren't formally published or versioned** — see "Bundled checkpoints" above and [`docs/extending.md`](docs/extending.md).

### Intended Use (wind)

Research use — data augmentation for downstream wind-energy models, stress-testing forecasting pipelines, exploratory analysis — same spirit as load's intended use above. Because of the "mid" and `2ch` limitations, treat generated wind data as a research tool, not a validated substitute for real measurements in any decision with real-world consequences.

### Out-of-Scope Uses (wind)

- **Any real-world energy/grid decision** (turbine load certification, dispatch planning) without independent validation against real data.
- **`2ch` with guidance enabled** — measurably breaks temporal realism.
- **Locations other than the single trained grid point** (offshore NJ) — untested.
- **Relying on `wind_regime="mid"` steering being reliable** through this package's `.set()`/`.sample()` interface — it isn't (see Known Limitations); the source project's own mitigation (rejection sampling) isn't wired into this package's Tier-1 interface today.
