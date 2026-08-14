# Experiment Map

A chronological map of the research program, from the original EDA pass to the
Transformer-based, calibrated, 17-zone result. Each entry answers: what question was
being asked, how it was answered, what the result was, and what to do with that result
today. Source files are under `data/research_history/` unless noted.

---

## Week 1 — Exploratory Data Analysis

- **Question:** What does the raw PJM RTO load + temperature data actually look like?
- **Method:** Two independent EDA passes over the raw data (later merged into one frozen
  script — see below).
- **Experiment:** `scripts/eda_exploration.py` (merged from `week1_testing.py` +
  `eda_plots.py`).
- **Result:** Figures under `reports/eda_figures/` (seasonal patterns, hourly profiles,
  a temperature/zone heatmap, a COVID-era trend break, a "V-curve" temperature/load
  relationship signal).
- **Conclusion:** Load has strong seasonal and temperature-driven structure, motivating
  conditioning on both.
- **Still scientifically relevant:** Yes, as motivation — not as a quantitative result.
- **Source files:** `scripts/eda_exploration.py`, `reports/eda_figures/*.png`.
- **Part of the public package:** No (see `examples/exploratory_data_analysis.py` for a
  small demo against the processed dataset instead).
- **Paper-worthy:** As background/motivation figures, potentially yes.

## Week 3 — Baseline Pipeline (MLP + DDPM)

- **Question:** Can a conditional DDPM generate a plausible 24-hour PJM load profile at
  all?
- **Method:** Full pipeline from raw CSVs to a trained conditional MLP-denoiser DDPM,
  zone PS, chronological 2016–2023/2024–2025 split.
- **Experiment:** `scripts/week3.py`.
- **Result:** A working baseline (`baseline_model_z5.pt`) and the Z5 (6-dim)/Z9 (9-dim)
  conditioning representations used throughout the rest of the research lineage.
- **Conclusion:** Conditional diffusion is viable for this data; established the MLP
  baseline architecture and the raw-conditioning convention.
- **Still scientifically relevant:** Yes — the MLP baseline is the comparison point every
  later architecture result is measured against.
- **Source files:** `scripts/week3.py`, `outputs/week3_outputs/`.
- **Part of the public package:** Superseded by `src/enerdatasynth/data/preprocess.py`,
  `models/denoiser.py`, `diffusion/ddpm.py` — but the package's own conditioning
  (23-dim one-hot) differs from Z5/Z9 (see `docs/research_status.md` §C).
- **Paper-worthy:** As the baseline row in a results table, yes.

## Week 5 — Classical Benchmark, Convergence, and the Sampler Bug

- **Question (Part A):** How does the diffusion baseline compare to classical time-series
  models (AR/ARMA/ARIMA/SARIMA)?
- **Method:** Run all four classical models plus the Week 3 diffusion baseline through
  the same 7-metric evaluation.
- **Experiment:** `scripts/week5_benchmark.py` — **used the original, later-found-buggy
  DDIM sampler.**
- **Result:** Diffusion beat the classical models even under the buggy sampler, but the
  numbers are invalid for the diffusion row specifically (see below).
- **Question (Part B):** Does training the baseline to full convergence help?
- **Method:** Train to convergence with validation tracking (`scripts/week5_part2.py`),
  producing `converged_model_z5.pt`.
- **Result:** Convergence made some generation metrics *worse* (FirstDiff, RMSE), which
  was unexpected and triggered the next investigation.
- **Question (Part C):** Is the convergence regression a real model property or a
  sampler artifact?
- **Method:** `scripts/sampler_bugfix.py` Part 1 — checks value-range overshoot and
  sensitivity to DDIM step count.
- **Result:** **Sampler bug confirmed.** The DDIM implementation reused a noise
  prediction incorrectly across steps. Fixed in `sampler_bugfix.py` Part 2 with a
  corrected ancestral DDPM sampler that recomputes the prediction every step.
- **Concrete before/after:** FirstDiff 2.2864 → 0.0179; RMSE 0.5069 → 0.0261
  (`reports/week5_comparison_table.csv` vs. `week5_comparison_corrected.csv`).
- **Conclusion:** The pre-fix Week 5 diffusion numbers are invalid and must never be
  quoted as final results. The corrected ancestral sampler is the one used everywhere
  from this point forward, including in the shipped package.
- **Still scientifically relevant:** Critically — this is the single most important
  correction in the project's history.
- **Source files:** `scripts/week5_benchmark.py`, `week5_part2.py`, `sampler_bugfix.py`,
  `reports/week5_comparison_table.csv`, `week5_comparison_corrected.csv`,
  `week5_metrics.json`.
- **Part of the public package:** Yes — `src/enerdatasynth/diffusion/ddpm.py` ships only
  the corrected sampler; a regression test was added during this audit
  (`tests/test_diffusion.py`) to prevent silently reintroducing the bug.
- **Paper-worthy:** Yes, and should be reported honestly, including the bug — this is
  exactly the kind of finding a credible methods section should disclose.

## Week 6 — Architecture Comparison and Noise Schedule

- **Question (Part A):** Does the denoiser architecture matter, holding everything else
  constant?
- **Method:** MLP / CNN / attention / Transformer, identical data, conditioning,
  schedule, epochs, seed, sampler.
- **Experiment:** `scripts/arch_compare.py`.
- **Result:** Transformer had the best generation-quality metrics (RMSE, Peak); CNN had
  the lowest validation loss but took 118 minutes to train (vs. 2.8–4.4 for everything
  else) — an anomaly never explained.
- **Conclusion:** Transformer selected as the go-forward architecture.
- **Question (Part B):** Does the diffusion noise (β) schedule matter?
- **Method:** Linear / cosine / quadratic, everything else identical.
- **Experiment:** `scripts/week6.py --task noise`.
- **Result:** Linear schedule won on both validation loss and generation RMSE, and was
  dramatically better on Energy error.
- **Conclusion:** Linear schedule adopted as the default.
- **Bonus:** `scripts/week6.py --task conditional` ran the first temperature/calendar/
  season slice evaluation, establishing the conditional-evaluation methodology used again
  (more rigorously) in Week 7.
- **Still scientifically relevant:** Yes — direct justification for two of the package's
  current defaults (`DEFAULT_ARCHITECTURE="transformer"`, linear β schedule).
- **Source files:** `scripts/arch_compare.py`, `scripts/week6.py`,
  `reports/week6/arch/`, `reports/week6/noise/`, `reports/week6/conditional/`.
- **Part of the public package:** Yes — `models/transformer.py`,
  `diffusion/ddpm.py`'s default betas.
- **Paper-worthy:** Yes — the architecture-comparison table and the CNN anomaly are both
  worth including.

## Week 7 — Transformer to Convergence, Calibration, Error Analysis

- **Question:** Does the Transformer's Week 6 advantage hold up when trained fully and
  re-evaluated head-to-head against everything (MLP, 4 classical models)?
- **Method:** Train Transformer to convergence; full re-evaluation.
- **Experiment:** `scripts/week7.py`.
- **Result:** Transformer beats every classical baseline 7/7 and the MLP 6/7 (loses only
  on Energy) at ~6× fewer parameters. Consistent across temperature/calendar/season
  slices, hardest on Holiday (small sample).
- **Conclusion:** Transformer promoted to primary/recommended architecture.
- **Question (follow-up):** Why does Energy regress even though every shape metric
  improves?
- **Method:** Measure per-hour bias, test whether a constant offset explains the energy
  error.
- **Experiment:** `scripts/week7_erroranalysis.py`.
- **Result:** A constant −0.00596/hour offset explains 100% of the energy error. Worst
  10% of days are 68.3% hot-condition days (vs. 25.3% baseline rate).
- **Conclusion:** A simple post-hoc calibration should fix Energy without touching shape
  metrics; hot-day generation quality is a separate, real weakness.
- **Method (calibration):** Fit scalar/per-hour bias on 1,200 training days, apply at
  generation time, re-score all 7 metrics.
- **Experiment:** `scripts/week7_calibrate.py`.
- **Result:** Scalar calibration cuts Energy error 66% (0.1644→0.0563) at zero cost to
  shape metrics; per-hour calibration is strictly worse.
- **Conclusion:** Scalar calibration adopted, stored as `calibration.json` next to a
  checkpoint.
- **Still scientifically relevant:** Yes — this is the current primary result.
- **Source files:** `scripts/week7*.py`, `reports/week7/*`.
- **Part of the public package:** Architecture and sampler yes; **the actual trained,
  calibrated Transformer checkpoint from this work is not bundled** — see
  `docs/research_status.md` §D/§F.
- **Paper-worthy:** Yes — this is the headline result.

## Phase 2 — Zone Expansion

- **Question:** Does the Week 7 pipeline (Transformer, corrected sampler, calibration)
  generalize to other PJM zones without per-zone architecture changes?
- **Method:** Replicate the exact PS pipeline across 16 additional zones, only the load
  data/scalers/temperature column changing.
- **Experiment:** `scripts/zone_expansion.py`.
- **Result:** 16 of 17 zones generalize well (RMSE 0.0086–0.0436). DOM fails badly
  (RMSE 0.1998, ~4.6× the next-worst zone) with no diagnosed cause.
- **Conclusion:** The pipeline generalizes broadly but not universally; DOM needs
  dedicated investigation before being claimed as "solved."
- **Still scientifically relevant:** Yes — this is the generalization evidence for any
  claim beyond a single zone.
- **Source files:** `scripts/zone_expansion.py`, `reports/phase2/`.
- **Part of the public package:** No — `pjm_load` exposes zone `PS` only today.
- **Paper-worthy:** Yes, including the DOM failure — reporting only the 16 successes
  without the outlier would overstate generalization.

## Software / Refactoring Phase (this repository's current form)

- **Question:** Can the research pipeline become a reusable, installable SDK without
  losing scientific fidelity or hiding what's actually validated vs. not?
- **Method:** Extract the Week 7 architecture/sampler/calibration/metrics logic into
  `src/enerdatasynth/`, build a fluent public API (`model().set().sample()`), add a
  generic multi-resource registry (enabling wind as a second resource type), write tests,
  and — in this audit pass — document honestly where the package and the research
  results currently diverge.
- **Result:** A working, tested (230/230), installable package — but one that currently
  ships the MLP baseline (not the validated Transformer) for `pjm_load`, and exposes only
  1 of 17 researched zones. See `docs/research_status.md` for the full, current state.
- **Still in progress:** training a Transformer checkpoint against the package's actual
  23-dim conditioning is the concrete next step to close that gap — see
  `docs/release_audit.md`.
