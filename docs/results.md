# Results

Canonical results summary. Every number below is traceable to a specific file under
`data/research_history/pjm_load/reports/` or `data/reports/pjm_load/`; the source is given after each
table. **See `docs/research_status.md` §C/§D first** — the numbers here come from the
research-history "Z5" (6-dimensional) conditioning representation, not the current
package's 23-dimensional one-hot conditioning, unless stated otherwise.

All metrics are defined in `docs/research_status.md` §G / `src/enerdatasynth/resources/pjm_load/evaluation/metrics.py`.
Lower is better on every metric below.

---

## 1. Main Transformer Result

The Transformer denoiser (Week 6 architecture-comparison winner), trained to convergence
and promoted to the primary/recommended architecture in Week 7.

| Model | Hourly | FirstDiff | ACF | RMSE | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| **Transformer (new baseline)** | 0.0243 | 0.0031 | 0.0325 | 0.0195 | 0.0089 | 0.1405 | 0.0025 |
| MLP (old baseline) | 0.0399 | 0.0185 | 0.0429 | 0.0259 | 0.0115 | 0.0008 | 0.016 |

Transformer beats the old MLP baseline on **6 of 7** metrics, at **~6× fewer parameters**
(110,401 vs. 688,408 — Week 6 architecture-comparison configuration). The one metric it
loses on is `Energy` — see §6 (Calibration) for why, and §7 (Worst-Day Analysis) for the
root cause.

Source: `data/research_history/pjm_load/reports/week7/summary_report.md`.

## 2. MLP Comparison (Architecture Selection)

Controlled experiment (Week 6): identical data, Z5 conditioning, `T=1000`, linear
schedule, epochs, learning rate, batch size, seed, and sampler — **only the denoiser
architecture changes.**

| Architecture | Params | Best val loss | RMSE | Hourly | FirstDiff | ACF | Peak | Energy | Ramp | Train time (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 688,408 | 0.01407 | 0.0241 | 0.0333 | 0.017 | 0.0448 | 0.0116 | 0.0833 | 0.0149 | 2.8 |
| **CNN** | 158,401 | **0.01022** | 0.0231 | 0.0287 | 0.0006 | 0.0467 | 0.0009 | 0.1072 | 0.0006 | **118.0** |
| Attention | 60,417 | 0.01589 | 0.022 | 0.0423 | 0.0179 | 0.0408 | 0.0092 | 0.0582 | 0.0153 | 2.9 |
| **Transformer** | 110,401 | 0.01087 | **0.0215** | 0.0364 | 0.0013 | 0.0388 | **0.0003** | 0.075 | 0.0011 | 4.4 |

CNN reached the lowest validation loss, but **validation loss (noise-prediction MSE) is
not directly comparable to generation quality** — the report's own note. Judged on
generated-profile metrics, the Transformer has the best RMSE and best Peak error.

**Unexplained anomaly, documented rather than hidden:** the CNN configuration took
**118.0 minutes** to train versus 2.8–4.4 minutes for every other architecture in the
same controlled experiment — a ~30–40× slowdown with no explanation recorded in the
source report. This was not re-investigated during this audit; it is flagged here as an
open, unresolved item rather than silently omitted.

Source: `data/research_history/pjm_load/reports/week6/arch/summary_report.md` /
`week6/arch/metrics.csv`.

## 3. Classical Baseline Comparison

Same Week 7 head-to-head run as §1, full table:

| Model | Hourly | FirstDiff | ACF | RMSE | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| AR | 1.36 | 0.423 | 0.1464 | 0.1887 | 0.7557 | 4.4371 | 0.1252 |
| ARMA | 1.3611 | 0.4191 | 0.1084 | 0.1918 | 0.7557 | 4.4636 | 0.1223 |
| ARIMA | 1.3535 | 0.1633 | 0.327 | 0.1808 | 0.3124 | 4.2818 | 0.0057 |
| SARIMA | 1.3614 | 0.6345 | 0.2478 | 0.1905 | 0.6764 | 4.4705 | 0.2927 |
| MLP (old baseline) | 0.0399 | 0.0185 | 0.0429 | 0.0259 | 0.0115 | 0.0008 | 0.016 |
| **Transformer (new baseline)** | 0.0243 | 0.0031 | 0.0325 | 0.0195 | 0.0089 | 0.1405 | 0.0025 |

**Transformer beats every classical baseline (AR/ARMA/ARIMA/SARIMA) on 7 of 7 metrics.**
The classical models are orders of magnitude worse across the board — none of them is a
conditional generative model in the same sense; they are included as a sanity-check
lower bound, not a close competitor.

Source: `data/research_history/pjm_load/reports/week7/summary_report.md`.

## 4. Conditional Evaluation

**Final (post-convergence Transformer), validation-set slices:**

| Group | Slice | n | RMSE | Peak | Ramp |
|---|---|---|---|---|---|
| temp | Cold (bottom 25%) | 152 | 0.026 | 0.0256 | 0.0004 |
| temp | Mild (mid 50%) | 296 | 0.0209 | 0.0053 | 0.0023 |
| temp | Hot (top 25%) | 152 | 0.0205 | 0.001 | 0.0049 |
| calendar | Weekday | 414 | 0.023 | 0.0184 | 0.0021 |
| calendar | Weekend | 169 | 0.0195 | 0.01 | 0.0032 |
| calendar | Holiday | 17 | 0.0306 | 0.0359 | 0.0044 |
| season | Winter | 150 | 0.0283 | 0.0297 | 0.0002 |
| season | Spring | 182 | 0.0209 | 0.0017 | 0.0019 |
| season | Summer | 178 | 0.0266 | 0.0124 | 0.005 |
| season | Fall | 90 | 0.0193 | 0.0115 | 0.0023 |

Hardest slice: **calendar/Holiday** (RMSE 0.0306) — also the smallest slice (n=17), so
some of this is small-sample noise rather than a systematic holiday failure.

Source: `data/research_history/pjm_load/reports/week7/summary_report.md`.

*(An earlier, pre-convergence conditional evaluation exists at
`data/research_history/pjm_load/reports/week6/conditional/summary_report.md` with a fuller RMSE/
FirstDiff/ACF/Peak/Energy/Ramp breakdown; it is superseded by the table above and kept
only for provenance.)*

## 5. Noise Schedule Experiment

Three diffusion models, identical except the β schedule.

| Schedule | Best val loss | RMSE | FirstDiff | ACF | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| **Linear** | **0.01407** | **0.0241** | 0.0170 | 0.0448 | 0.0116 | 0.0833 | 0.0149 |
| Cosine | 0.02545 | 0.0302 | 0.0116 | 0.0472 | 0.0247 | 0.4227 | 0.0097 |
| Quadratic | 0.02376 | 0.0244 | 0.0105 | 0.0470 | 0.0083 | 0.1216 | 0.0094 |

Linear wins on both validation loss and generated-profile RMSE, and is dramatically
better on `Energy`. This is why `src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py` defaults to a
linear schedule (`beta_start=1e-4, beta_end=0.02`).

Source: `data/research_history/pjm_load/reports/week6/noise/summary_report.md`.

## 6. Calibration

Bias measured on 1,200 training days (validation never used for fitting).

| Variant | Hourly | FirstDiff | ACF | RMSE | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.0296 | 0.0031 | 0.0317 | 0.0210 | 0.0084 | 0.1644 | 0.0025 |
| **Scalar (+0.00920/hr)** | 0.0296 | 0.0031 | 0.0317 | 0.0200 | 0.0008 | **0.0563** | 0.0025 |
| Per-hour | 0.0333 | 0.0034 | 0.0501 | 0.0249 | 0.0046 | 0.0563 | 0.0028 |

Scalar calibration cuts Energy error 66% (0.1644 → 0.0563) at zero cost to shape metrics,
and is strictly better than the per-hour variant on every other metric. See
`docs/research_status.md` §F for the full mechanism and the train-fit caveat.

Source: `data/research_history/pjm_load/reports/week7/calibration_report.md`.

## 7. Worst-Day / Error Analysis

Root cause of the Transformer's one weak metric (`Energy`, §1): a **systematic per-hour
level offset**, not a shape problem.

- Per-hour level offset (generated − real): **−0.00596**
- Signed daily energy bias: **−0.14305** (offset × 24h)
- The constant offset alone explains **100.0%** of the reported energy error — removing
  it brings the residual to 0.00000.
- Median per-day RMSE: **0.0680**; p90 **0.1335**; max **0.4593**

**Worst-10%-of-days composition vs. overall day mix:**

| Condition | Share of worst 10% | Share of all days |
|---|---|---|
| Hot | **68.3%** | 25.3% |
| Cold | 10.0% | 25.3% |
| Holiday | 3.3% | 2.8% |
| Weekend | 26.7% | 28.2% |

**Hot days are ~2.7× over-represented among the worst-generated days.** This is a real,
unresolved limitation, not a rounding artifact — the three worst individual days (RMSE
0.46, 0.38, 0.34) are all June/July days with above-average `Tmax`.

Source: `data/research_history/pjm_load/reports/week7/energy_diagnosis.md`.

## 8. Zone Expansion (Phase 2)

17 PJM zones, identical pipeline (same preprocessing, Z5 conditioning, Transformer
architecture, corrected sampler, calibration step) — only the load data, scaler
parameters, and temperature column differ per zone.

| Zone | Train days | RMSE | Peak | Ramp | Energy |
|---|---|---|---|---|---|
| **AP** (best) | 2793 | **0.0086** | 0.0084 | 0.0034 | 0.1121 |
| DAY | 2793 | 0.0087 | 0.0011 | 0.0066 | 0.1427 |
| PL | 2793 | 0.0114 | 0.0023 | 0.0034 | 0.2260 |
| DEOK | 2793 | 0.0149 | 0.0061 | 0.0035 | 0.3217 |
| CE | 2793 | 0.0159 | 0.0066 | 0.0058 | 0.0793 |
| PN | 2793 | 0.0161 | 0.0090 | 0.0023 | 0.3506 |
| DPL | 2793 | 0.0171 | 0.0214 | 0.0016 | 0.3262 |
| BC | 2793 | 0.0183 | 0.0092 | 0.0037 | 0.2742 |
| PS | 2793 | 0.0193 | 0.0014 | 0.0049 | 0.0633 |
| RECO | 2793 | 0.0195 | 0.0095 | 0.0015 | 0.1190 |
| JC | 2793 | 0.0238 | 0.0069 | 0.0044 | 0.1690 |
| ME | 2793 | 0.0238 | 0.0144 | 0.0026 | 0.5148 |
| DUQ | 2793 | 0.0242 | 0.0044 | 0.0030 | 0.1460 |
| PE | 2793 | 0.0250 | 0.0161 | 0.0028 | 0.5608 |
| AE | 2786 | 0.0392 | 0.0397 | 0.0068 | 0.9280 |
| EKPC | 2793 | 0.0436 | 0.0383 | 0.0038 | 1.0288 |
| **DOM** (worst) | 2793 | **0.1998** | 0.1958 | 0.0108 | 4.7833 |

**DOM is a clear, unresolved outlier**: its RMSE (0.1998) is ~4.6× worse than the next
worst zone (EKPC, 0.0436) and ~23× worse than the best (AP, 0.0086). No root cause for
DOM's failure is recorded in this report. **16 of 17 zones generalize well; DOM does
not, and this is not fixed.**

**Only `PS` is exposed by the installed package's `pjm_load` resource today** — see
`docs/research_status.md` §B. The 16-of-17-zone generalization result is a Phase 2
research finding, not yet wired into the package's public zone list.

Source: `data/research_history/pjm_load/reports/phase2/summary_report.md`.

## 9. Limitations

Consolidated from the sections above — see `docs/research_status.md` and
`docs/models/pjm_load.md` for full discussion:

1. **Conditioning mismatch (critical):** the Transformer results in §1–§7 above used the
   research-history 6-dimensional "Z5" conditioning, not the current package's
   23-dimensional one-hot conditioning. They should not be quoted as "what the installed
   package produces". The shipped Transformer was trained on the 23-dim conditioning and
   has its own numbers: `docs/models/pjm_load.md` §8 and
   `data/reports/pjm_load/checkpoint_comparison.csv`.
2. **The shipped default is the Transformer** (`transformer_T1000_H64_L2`), from a single
   training run with the research-validated hyperparameters, not a from-scratch sweep.
3. **The shipped default is calibrated** (scalar +0.02342, fit on training days). The seven
   legacy MLP checkpoints are not.
4. **DOM zone failure is unresolved** (§8).
5. **Calibration is train-fit**, not independently validated on a held-out calibration
   set beyond the 1,200-day sample it was fit on (§6, `docs/research_status.md` §F).
6. **In-sample validation only:** `data/reports/pjm_load/results_summary.md` states validation
   comparisons use the same conditioning vectors as the real validation set — this
   evaluates conditional fidelity, not out-of-distribution generalization.
7. **Hot-day generation quality is weaker** than the aggregate metrics suggest (§7).
8. **CNN's 118-minute training-time anomaly** (§2) was never explained.
9. **No unified multi-zone model exists** — each zone in §8 was trained independently;
   `data/reports/pjm_load/future_experiments.md` lists a unified PJM model as a recommended,
   not-yet-implemented future experiment.
10. **No wind model is scientifically validated.** See `docs/research_status.md` and
    `docs/models/wind.md` for the wind implementation's actual status (code-complete,
    checkpoints exist, but no raw training data or independent validation against
    ground-truth wind observations exists in this repository).
