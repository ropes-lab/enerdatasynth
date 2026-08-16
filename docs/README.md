# Documentation Index

## By task

| I want to... | Go to |
|---|---|
| Install and run my first sample | [`quickstart.md`](quickstart.md) |
| Understand how the package is put together | [`architecture.md`](architecture.md) |
| Use / understand the PJM load model | [`models/pjm_load.md`](models/pjm_load.md) |
| Use / understand the wind model | [`models/wind.md`](models/wind.md) |
| Look up a method's exact signature | [`api_reference.md`](api_reference.md) |
| Add a new resource type | [`extending.md`](extending.md) |
| Check what's scientifically verified vs. not | [`research_status.md`](research_status.md) |
| Contribute to the package | [`developer_guide.md`](developer_guide.md) |

**Model reports** ([`models/pjm_load.md`](models/pjm_load.md),
[`models/wind.md`](models/wind.md)) are the primary scientific documentation. Start
there for anything beyond basic usage.

## Everything else

| Page | What it answers |
|---|---|
| [`research_status.md`](research_status.md) | Source of truth: what's established, and where the package diverges from it. |
| [`results.md`](results.md) | Canonical, traceable results tables. |
| [`data_card.md`](data_card.md) | Where the data comes from, cleaning, licensing status. |
| [`model_card.md`](model_card.md) | Deeper model-card cross-check for `pjm_load` (the `models/` report is the primary read). |
| [`experiments.md`](experiments.md) | Chronological map of the research program. |
| [`model_report_card.md`](model_report_card.md) + [`report_cards/`](report_cards/) | PJM load's Transformer vs. MLP architecture comparison, plus wind's 1ch/2ch/3ch preset comparison. |
| [`installation.md`](installation.md) | All install paths (PyPI, editable, wheel). |
| [`api_reference.md`](api_reference.md) | Full `model()`/`.set()`/`.sample()` contract. |
| [`architecture.md`](architecture.md) | Why Transformer over MLP/CNN/attention, ancestral sampling over DDIM, scalar calibration — traced to source. |
| [`training.md`](training.md) | Train, sweep, calibrate, evaluate a new checkpoint. |
| [`evaluation.md`](evaluation.md) | The 7-metric suite in practice. |
| [`temperature_data.md`](temperature_data.md) | NOAA ISD fetch/cache/zone-mapping mechanics. |
| [`extending.md`](extending.md) | Adding a new resource type (the pattern wind followed). |
| [`packaging.md`](packaging.md) | What `pip install enerdatasynth` does and doesn't ship. |
| [`developer_guide.md`](developer_guide.md) | Contributing to the package. |
| [`examples.md`](examples.md) | Walkthrough of `examples/`. |
| [`faq.md`](faq.md) | Common questions. |
| [`release_audit.md`](release_audit.md) | Latest audit's findings and blockers, by severity. |

## Provenance

`../data/research_history/README.md` — the frozen scripts/reports every number in
`results.md`/`research_status.md` traces back to.
