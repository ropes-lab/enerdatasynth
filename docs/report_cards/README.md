# `docs/report_cards/`

One-page comparison cards, narrower than a full resource-level report:

- [`transformer.md`](transformer.md) and [`mlp.md`](mlp.md) — PJM load's two denoiser *architectures* (Transformer is recommended; MLP is the legacy baseline and currently the package's de facto default — see either card's Limitations section for why). Indexed together in [`../model_report_card.md`](../model_report_card.md).
- [`wind_presets.md`](wind_presets.md) — wind's three channel *presets* (`1ch`/`2ch`/`3ch`) of the same shared architecture. Wind doesn't have two competing architectures the way load does, so this compares presets instead — a different axis, not a smaller version of the same comparison.

Not to be confused with [`docs/models/`](../models/README.md), which covers the full resource-level reports (PJM load vs. wind, the two products). The cards here are narrower comparisons within a single resource.
