# Model Report Cards

Two models exist in this repository's research lineage: the **Transformer** denoiser
(recommended/primary, Week 6–7 winner, and what the installed package loads by default)
and the **MLP** denoiser (the original baseline, kept bundled for comparison). Both are
summarized below in the same fixed structure, then linked out to a full one-page card
each, so they're directly comparable at a glance.

| | [Transformer](report_cards/transformer.md) | [MLP](report_cards/mlp.md) |
|---|---|---|
| **Role** | Recommended / primary; **current package default** | Original baseline; kept for comparison |
| **Params** (Week 6 comparison config) | 110,401 | 688,408 |
| **RMSE** (Week 7 head-to-head) | 0.0195 | 0.0259 |
| **Energy error** (Week 7 head-to-head) | 0.1405 | **0.0008** (MLP wins this one) |
| **Beats classical baselines** | 7 / 7 metrics | 5–6 / 7 metrics (see card) |
| **Calibration available** | Yes, method validated (Week 7); shipped checkpoint is calibrated | Not separately validated; none of the 7 checkpoints calibrated |
| **Bundled checkpoint compatible with current 23-dim package conditioning** | **Yes** (`transformer_T1000_H64_L2`, 111,489 params) | Yes (7 checkpoints, none calibrated) |
| **What `enerdatasynth.model("pjm_load", ...)` loads today** | **This one, by default** | — |
| **On the 23-dim conditioning** (5 seeds, 600 val days, normalized) | RMSE 0.0156, Energy 0.0462, wins 7 / 7 vs. the MLP | RMSE 0.0465, Energy 0.7699 |

The first four numeric rows are the Week 6/7 research results (6-dim conditioning); the
last row is the shipped checkpoint on the package's own conditioning, a separate
measurement (`data/reports/pjm_load/checkpoint_comparison.csv`).

See `docs/research_status.md` and `docs/results.md` for the full evidence behind every
number here — this page and the two linked cards are a summary view, not a new source of
truth.

## Wind

Wind doesn't have two competing architectures the way load's Transformer/MLP
comparison does (one vendored pipeline, shared across all three presets) — so there's
no equivalent side-by-side table here. It does have three channel *presets*
(`1ch`/`2ch`/`3ch`), compared in [`report_cards/wind_presets.md`](report_cards/wind_presets.md).
