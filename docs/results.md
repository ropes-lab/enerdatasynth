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

## 8. Zone Expansion

All 17 PJM zones with load and temperature data now ship with their own model (`zone=` in `enerdatasynth.model("pjm_load", ...)`). Same pipeline, settings and calibration step for every zone (see [`docs/models/pjm_load.md`](models/pjm_load.md) §11 for the full discussion). Scores on each zone's 2024–2025 validation days, 3 generation seeds, normalized units (lower is better); source `data/reports/pjm_load/zone_results.csv`, built by `examples/pjm_load/evaluate_all_zones.py`:

| Zone | Train / val days | RMSE (raw → calibrated) | Energy (raw → calibrated) | Peak | ACF | Load level, val ÷ train | Val days above train max |
|---|---|---|---|---|---|---|---|
| DUQ | 2785 / 602 | 0.0165 → 0.0121 | 0.275 → 0.047 | 0.0082 | 0.0361 | 1.001 | 0.3% |
| CE | 2785 / 600 | 0.0183 → 0.0123 | 0.330 → 0.058 | 0.0066 | 0.0303 | 0.987 | 0.0% |
| DEOK | 2785 / 602 | 0.0160 → 0.0129 | 0.353 → 0.274 | 0.0049 | 0.0115 | 1.001 | 0.3% |
| AP | 2785 / 602 | 0.0212 → 0.0135 | 0.472 → 0.262 | 0.0164 | 0.0131 | 1.012 | 0.8% |
| PN | 2785 / 602 | 0.0135 → 0.0145 | 0.258 → 0.290 | 0.0138 | 0.0127 | 0.987 | 0.0% |
| JC | 2785 / 600 | 0.0241 → 0.0150 | 0.470 → 0.122 | 0.0138 | 0.0186 | 0.993 | 0.7% |
| PS | 2785 / 600 | 0.0264 → 0.0153 | 0.519 → 0.043 | 0.0125 | 0.0296 | 0.999 | 1.3% |
| BC | 2785 / 602 | 0.0216 → 0.0165 | 0.362 → 0.139 | 0.0288 | 0.0812 | 0.992 | 0.7% |
| DAY | 2785 / 602 | 0.0138 → 0.0184 | 0.248 → 0.384 | 0.0272 | 0.0180 | 1.015 | 1.7% |
| PE | 2785 / 602 | 0.0094 → 0.0191 | 0.142 → 0.425 | 0.0194 | 0.0229 | 0.988 | 0.3% |
| PL | 2785 / 602 | 0.0197 → 0.0200 | 0.270 → 0.283 | 0.0166 | 0.0158 | 1.011 | 0.3% |
| DPL | 2785 / 602 | 0.0337 → 0.0203 | 0.708 → 0.287 | 0.0239 | 0.0200 | 1.017 | 1.5% |
| AE | 2628 / 600 | 0.0232 → 0.0228 | 0.105 → 0.042 | 0.0091 | 0.0263 | 0.988 | 0.0% |
| RECO | 2785 / 600 | 0.0229 → 0.0233 | 0.020 → 0.101 | 0.0149 | 0.0251 | 0.997 | 0.3% |
| ME | 2785 / 602 | 0.0110 → 0.0236 | 0.042 → 0.503 | 0.0228 | 0.0101 | 0.985 | 0.2% |
| **EKPC** | 2785 / 602 | 0.0396 → 0.0494 | 0.936 → 1.174 | 0.0514 | 0.0382 | 1.090 | 0.8% |
| **DOM** | 2785 / 602 | 0.2046 → 0.2022 | 4.887 → 4.830 | 0.1980 | 0.0911 | 1.219 | 7.8% |

**15 of 17 zones are modeled about as well as PS** (calibrated RMSE 0.0121–0.0236, median 0.0165). **DOM (0.2022) and EKPC (0.0494) are not, and the data explains it:** DOM's load grew 39% from 2016 to 2025, so its validation days average 22% above its training days and 7.8% exceed the training maximum; EKPC grew 9%. A model conditioned only on calendar and weather generates the training-period level. The research-phase run (older 6-dim conditioning) measured the same DOM failure without a diagnosis.

**Calibration is not uniformly helpful across zones.** It lowers RMSE in 9 zones and raises it in 5 (DAY, EKPC, ME, PE, PN; PL and RECO unchanged), because the offset is fit on training days and does not transfer where the load level drifts. It is applied by default everywhere; `apply_calibration=False` gives the raw model.

**Against a tuned MLP denoiser** (all 17 zones, `data/reports/pjm_load/diffusion_zone_comparison.csv`): the Transformer has the lower first-difference, autocorrelation and ramp error in all 17 zones, the lower MMD² in 16, RMSE in 15 and Wasserstein distance in 14, with calibration on both; daily energy is a wash. The MLP here is trained like the Transformer and tuned per zone, unlike the research-phase MLP of §2, which was under-trained (RMSE 0.0465 on PS against about 0.022). Details in [`docs/models/pjm_load.md`](models/pjm_load.md) §11.

**Research phase (kept for history).** The first zone study (research-lineage 6-dim conditioning, `data/research_history/pjm_load/scripts/zone_expansion.py`) reported 16 of 17 zones with RMSE 0.0086 (AP) to 0.0436 (EKPC) and DOM at 0.1998. That script averaged the load areas of AE, DPL and PL instead of summing them, so those zones' MW were not zone totals.

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
4. **DOM (and, less so, EKPC) is poorly modeled**, because its load level moved between the training and validation years (§8). This is diagnosed but not fixed: the models see only calendar and weather.
5. **Calibration is train-fit**, not independently validated on a held-out calibration
   set beyond the 1,200-day sample it was fit on (§6, `docs/research_status.md` §F).
6. **In-sample validation only:** `data/reports/pjm_load/results_summary.md` states validation
   comparisons use the same conditioning vectors as the real validation set — this
   evaluates conditional fidelity, not out-of-distribution generalization.
7. **Hot-day generation quality is weaker** than the aggregate metrics suggest (§7).
8. **CNN's 118-minute training-time anomaly** (§2) was never explained.
9. **No unified multi-zone model exists** — each zone in §8 is a separate model, trained independently;
   `data/reports/pjm_load/future_experiments.md` lists a unified PJM model as a recommended,
   not-yet-implemented future experiment.
10. **No wind model is scientifically validated.** See `docs/research_status.md` and
    `docs/models/wind.md` for the wind implementation's actual status (code-complete,
    checkpoints exist, but no raw training data or independent validation against
    ground-truth wind observations exists in this repository).
