# `docs/report_cards/`

One-page architecture comparison cards for PJM load's two denoiser architectures —
[`transformer.md`](transformer.md) (recommended) and [`mlp.md`](mlp.md) (legacy
baseline, currently the package's de facto default — see either card's Limitations
section for why). Indexed together in [`../model_report_card.md`](../model_report_card.md).

**Not to be confused with [`docs/models/`](../models/README.md)** — those are the
full resource-level reports (PJM load vs. wind, the two *products*); these two cards
are a narrower comparison within PJM load specifically (one architecture vs. another).
There is no wind equivalent — wind doesn't have two competing architectures the way
load's Transformer/MLP comparison does.
