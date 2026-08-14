# `docs/models/`

The two model report cards — the primary scientific documentation for this project,
per the [root README](../../README.md#model-reports).

- [`pjm_load.md`](pjm_load.md)
- [`wind.md`](wind.md)

Both follow the same 14-section structure (Overview → Input Data → Data Representation
→ Conditioning → Generative Model → Training → Evaluation → Quantitative Results →
Actual vs. Generated Examples → Conditional/Regime Evaluation → Multi-Zone/Site Results
→ Calibration → Limitations → Reproduction) so the two resources are easy to compare
side by side, even though the underlying science, data, and results are resource-specific
and not forced to match. Start with whichever resource you're actually using; read both
if you want to understand how the product is unified across genuinely different models.

See [`../model_report_card.md`](../model_report_card.md) for a narrower, one-page
comparison of the two *architectures* used for PJM load specifically (Transformer vs.
MLP) — a different, more focused document than the resource-level reports here.
