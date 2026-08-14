# Documentation Index

**Start here if you're new to this project:** `docs/research_status.md`, then
`docs/quickstart.md`.

## Scientific documentation (what's true, and how we know)

| Page | What it answers |
|---|---|
| [`research_status.md`](research_status.md) | The source of truth: what's scientifically established, and exactly where the current package diverges from the validated research. **Read this first.** |
| [`results.md`](results.md) | The canonical, traceable results tables (architecture comparison, calibration, zone expansion, limitations). |
| [`data_card.md`](data_card.md) | Where the data comes from, how it's cleaned, and what's not verified about licensing. |
| [`model_card.md`](model_card.md) | Full model card for the `pjm_load` resource — intended/non-intended use, architecture, known limitations. |
| [`experiments.md`](experiments.md) | Chronological map of the research program, Week 1 through Phase 2. |
| [`model_report_card.md`](model_report_card.md) + [`report_cards/`](report_cards/) | One-page Transformer vs. MLP comparison cards. |

## Software documentation (how to use the package)

| Page | What it answers |
|---|---|
| [`quickstart.md`](quickstart.md) | Fastest path to a working install and first sample. |
| [`installation.md`](installation.md) | All install paths (PyPI, editable, wheel). |
| [`api_reference.md`](api_reference.md) | Full `enerdatasynth.model(...)` / `.set()` / `.sample()` contract. |
| [`architecture.md`](architecture.md) | Why Transformer over MLP/CNN/attention, why ancestral sampling over DDIM, why calibration is scalar — with every claim traced to a result file. |
| [`training.md`](training.md) | How to train, sweep, calibrate, and evaluate a new checkpoint. |
| [`evaluation.md`](evaluation.md) | The 7-metric evaluation suite in practice. |
| [`temperature_data.md`](temperature_data.md) | NOAA ISD fetch/cache/zone-mapping mechanics. |
| [`extending.md`](extending.md) | Adding a new resource type (the pattern wind followed). |
| [`packaging.md`](packaging.md) | What `pip install enerdatasynth` does and doesn't ship. |
| [`developer_guide.md`](developer_guide.md) | Contributing to the package itself. |
| [`examples.md`](examples.md) | Walkthrough of the `examples/` scripts. |
| [`faq.md`](faq.md) | Common questions. |

## Process documentation (this audit)

| Page | What it answers |
|---|---|
| [`release_audit.md`](release_audit.md) | What this documentation/packaging pass changed, what's still blocking, classified by severity. |

## Provenance

`../data/research_history/README.md` — the frozen historical scripts and reports every
number in `results.md`/`research_status.md` traces back to.
