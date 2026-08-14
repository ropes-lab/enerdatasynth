# Research Notes

Findings below are drawn directly from files under `data/research_history/pjm_load/`
(scripts and their generated reports). Anything not traceable to a
specific file is omitted rather than reconstructed from memory — this
page is deliberately narrower than a full project history might be.

## The sampler bug

The original sampler (`data/research_history/pjm_load/scripts/week3.py`'s
`ddim_sample`) reused a single noise prediction across a truncated
step schedule instead of recomputing it every step. Diagnosed in
`data/research_history/pjm_load/scripts/sampler_bugfix.py` Part 1
(`diagnose_converged.py`): generated values pinned near the ±3 clamp
bound, with an increasing out-of-range fraction as step count grew.
Fixed in the same file's Part 2 (`sampler_fixed.py`): a correct
ancestral sampler that recomputes the noise prediction at every
timestep. This is the only sampler in the maintained package
(`enerdatasynth.resources.pjm_load.diffusion.ddpm.DDPM.sample`) — see `docs/architecture.md`.

## Architecture comparison

`data/research_history/pjm_load/reports/week6/arch/summary_report.md`: MLP, CNN,
attention, and transformer denoisers compared under identical data,
conditioning, schedule, and sampler.

| Arch | Params | Best val loss | Generated RMSE |
|---|---|---|---|
| mlp | 688,408 | 0.01407 | 0.0241 |
| cnn | 158,401 | **0.01022** | 0.0231 |
| attention | 60,417 | 0.01589 | 0.0220 |
| transformer | 110,401 | 0.01087 | **0.0215** |

CNN had the lowest validation loss; transformer had the best generated
RMSE. The report explicitly says to judge on generated metrics, not
validation loss alone — transformer was selected on that basis.

## Noise schedule comparison

`data/research_history/pjm_load/reports/week6/noise/summary_report.md`: linear,
cosine, and quadratic beta schedules compared with everything else
held fixed. Linear won on both validation loss (0.01407) and generated
RMSE (0.0241) — unlike the architecture comparison, this one didn't
have a val-loss-vs-quality disagreement to resolve.

## Transformer promoted to baseline

`data/research_history/pjm_load/reports/week7/summary_report.md`: the transformer
trained to convergence, then re-evaluated head-to-head against the old
MLP baseline and four classical models (AR/ARMA/ARIMA/SARIMA).

- Beats every classical baseline on 7 of 7 metrics.
- Beats the old MLP baseline on 6 of 7 metrics, at ~6x fewer
  parameters (110,401 vs 688,408).
- Consistent across temperature/calendar/season slices (RMSE
  0.0195–0.0306); hardest slice is Holiday conditioning (n=17, the
  smallest slice).

## Calibration

`data/research_history/pjm_load/reports/week7/calibration_report.md`: the
transformer carries a small systematic per-hour level offset
(scalar +0.00920) that compounds into daily-energy error when summed
over 24 hours. A single scalar correction, fit on the training set
only, reduced energy error from 0.1644 to 0.0563 (a 66% reduction)
without changing shape metrics (a constant shift can't affect
first-difference, ramp, or autocorrelation). A more elaborate per-hour
correction was also tried and found *worse* on ACF and RMSE than no
calibration at all — the report's own output explicitly recorded
`"mode_recommended": "scalar"`.

## Zone generalization (Phase 2)

`data/research_history/pjm_load/reports/phase2/summary_report.md`: the full
validated pipeline (transformer + linear schedule + corrected sampler
+ scalar calibration) replicated across 17 PJM zones, each with its
own preprocessing/scalers/checkpoint. Best zone: AP (RMSE 0.0086).
Worst: DOM (RMSE 0.1998) — an outlier roughly 4-10x worse than every
other zone; the cause was not diagnosed in the available reports.
This zone-per-zone pipeline uses the same 6-dimensional raw
conditioning vector as the rest of the research lineage (see
`docs/architecture.md`'s note on conditioning), not the package's
current one-hot encoding — it was never ported into `src/`, only
noted as a candidate future extension (`MIGRATION.md` §4).

## What isn't in this page

Detailed per-experiment error analysis (`data/research_history/pjm_load/scripts/
week7_erroranalysis.py`), the full per-zone results table, and the
zone-expansion build/train/report tooling are in `data/research_history/pjm_load/`
directly — this page summarizes, it doesn't replace reading the
underlying reports.
