# Model Report Cards

Two models exist in this repository's research lineage: the **Transformer** denoiser
(recommended/primary, Week 6–7 winner) and the **MLP** denoiser (original baseline, and
— currently — what the installed package actually loads by default). Both are
summarized below in the same fixed structure, then linked out to a full one-page card
each, so they're directly comparable at a glance.

| | [Transformer](report_cards/transformer.md) | [MLP](report_cards/mlp.md) |
|---|---|---|
| **Role** | Recommended / primary | Original baseline; current package default |
| **Params** (Week 6 comparison config) | 110,401 | 688,408 |
| **RMSE** (Week 7 head-to-head) | 0.0195 | 0.0259 |
| **Energy error** (Week 7 head-to-head) | 0.1405 | **0.0008** (MLP wins this one) |
| **Beats classical baselines** | 7 / 7 metrics | 5–6 / 7 metrics (see card) |
| **Calibration available** | Yes, method validated (Week 7) | Not separately validated |
| **Bundled checkpoint compatible with current 23-dim package conditioning** | **No** | Yes (7 checkpoints, none calibrated) |
| **What `enerdatasynth.model("pjm_load", ...)` loads today** | — | **This one, by default** |

See `docs/research_status.md` and `docs/results.md` for the full evidence behind every
number here — this page and the two linked cards are a summary view, not a new source of
truth.

## Wind

Wind doesn't have two competing architectures the way load's Transformer/MLP
comparison does (one vendored pipeline, shared across all three presets) — so there's
no equivalent side-by-side table here. It does have three channel *presets*
(`1ch`/`2ch`/`3ch`), compared in [`report_cards/wind_presets.md`](report_cards/wind_presets.md).
