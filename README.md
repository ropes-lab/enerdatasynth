<div align="center">

# ⚡ enerdatasynth

### A research SDK for synthetic PJM energy resource profiles

[![PyPI](https://img.shields.io/pypi/v/enerdatasynth.svg)](https://pypi.org/project/enerdatasynth/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests: 230 passing](https://img.shields.io/badge/tests-230%20passing-brightgreen.svg)](tests/)
[![Packaging: verified](https://img.shields.io/badge/packaging-pip%20install%20verified-blue.svg)](MIGRATION.md)
[![Status: research software, pre-1.0](https://img.shields.io/badge/status-research%20software%20%E2%80%94%20pre--1.0-orange.svg)](docs/research_status.md)

**Generate realistic synthetic electric-load profiles from a fitted diffusion model — no ML expertise required.**

```bash
pip install enerdatasynth
```

The lines below are **Python, not shell** — run them inside a Python interpreter, e.g. paste this whole block into your terminal as-is (it wraps the code in `python3 -c "..."` for you):

```bash
python3 -c "
import enerdatasynth

load = enerdatasynth.model('pjm_load', zone='PS')
load.set(month=7, Tmax=86, Tmin=70, weekday=True)
samples = load.sample(n=100)   # (100, 24) — real MW, ready to use
print(samples.shape, samples[0])
"
```

> `pip install enerdatasynth` installs the library only — it doesn't ship trained checkpoints or data (by design; see [`docs/packaging.md`](docs/packaging.md)). The snippet above needs both, already present if you clone [this repo](.) and run from its root; see [Quick Start](#-quick-start) below for training your own from scratch elsewhere.

[Quick Start](#-quick-start) · [Documentation](#-documentation) · [Model Card](MODEL_CARD.md) · [Benchmark Results](#-benchmark-results) · [Citation](#-citation)

</div>

<br>

> **Note on figures:** this README is also this package's PyPI long description, which can't render images hosted in this repository until the repo's public URL is finalized. See [`data/research_history/reports/week7/profiles_headtohead.png`](data/research_history/reports/week7/profiles_headtohead.png) and [Benchmark Results](#-benchmark-results) below for the same comparison in table form.

---

## Why this exists

Utilities, grid operators, and energy researchers frequently need realistic synthetic load data — for privacy-preserving data sharing, stress-testing forecasting models, or scenario planning — without exposing real customer or grid data. `enerdatasynth` wraps a conditional diffusion pipeline behind an SDK that hides every ML implementation detail: you describe a day (season, temperature, calendar type), and get back a realistic profile.

## Research Question

> Can a conditional generative model reproduce the statistical, temporal, and operational characteristics of observed daily electricity-load profiles?

This is the question every experiment in [`docs/experiments.md`](docs/experiments.md) and every number in [`docs/results.md`](docs/results.md) is trying to answer. It is explicitly **not** "what will tomorrow's load be" — see below.

## What This Is / What This Is Not

**This is:**
- Conditional synthetic daily-load-profile generation (24 hourly values per day)
- A research and reproducibility framework, with every result traced to a source file ([`docs/research_status.md`](docs/research_status.md))
- Per-zone generation (`PS` packaged today; 17 zones researched — see [Zone Generalization](#zone-generalization))
- A second resource type (wind) behind the same interface, at an earlier stage of validation — see [Supported Generators](#-supported-generators)

**This is not:**
- A forecasting model — it does not predict a specific future day's actual load
- An operational dispatch or real-time grid-control system
- A causal model of what drives load
- A guarantee of physical feasibility for any generated profile

## ✨ Features

- **A fluent, implementation-hiding API** — `model()` → `.set()` → `.sample()`. No tensors, no checkpoint paths, no architecture classes ever surface.
- **A validated transformer architecture** — selected via a controlled 4-way architecture comparison (MLP/CNN/attention/transformer), not the first thing that was tried. See [Research Summary](#-research-summary). **Not yet the checkpoint loaded by default** — see [`docs/research_status.md`](docs/research_status.md) for exactly why `enerdatasynth.model("pjm_load", ...)` currently falls back to the legacy MLP baseline instead, and `enerdatasynth train` to produce a transformer checkpoint yourself.
- **A corrected diffusion sampler** — an earlier DDIM-style sampler was found to silently corrupt every result it touched; the fix is the only sampler shipped today.
- **Automatic bias calibration** — a systematic energy-error offset, diagnosed and corrected, applied automatically at generation time.
- **A second resource type, wind** — `enerdatasynth.model("wind", zone="1ch")`, wrapping a vendored diffusion pipeline (`wind_diffusion`, from a separate Data-For-Good project) through the exact same `BaseGenerator`/`ResourceModel` contract as load. All three channel presets (`1ch`/`2ch`/`3ch`) ship with real trained checkpoints bundled in this repo. See [Supported Generators](#-supported-generators) and [Extending](docs/extending.md).
- **Config-driven, YAML-first** hyperparameters (`data/configs/`), a real test suite (`tests/`, ~2–20s), and a full audit trail of every design decision ([MIGRATION.md](MIGRATION.md)) — nothing here is asserted without a file backing it.

## 🔌 Supported Generators

| Resource | Status | Zones / presets | Entry point |
|---|---|---|---|
| Electric load | ✅ Available | `PS` (PJM/PSE&G, NJ) | `enerdatasynth.model("pjm_load", zone="PS")` |
| Wind | ✅ Available | `1ch` / `2ch` / `3ch` — all bundled checkpoints | `enerdatasynth.model("wind", zone="1ch")` |

Wind wraps a separately-developed diffusion pipeline (`wind_diffusion`, vendored from a Data-For-Good research project) behind the same `.set()`/`.sample()` contract load uses — see [`docs/api_reference.md`](docs/api_reference.md) for its full parameter reference and [`docs/extending.md`](docs/extending.md) for how it's wired in (useful if you're adding a third resource type). Each preset uses its own measured-best generation recipe automatically (`2ch` specifically requires different settings than `1ch`/`3ch` — see `docs/api_reference.md`); `units="m/s"` is the only unit wind supports today, for every preset.

## 📦 Installation

```bash
pip install enerdatasynth     # from PyPI — library only, see the note above
# or, for the full repo (data + trained checkpoints included):
git clone <this repo> && cd enerdatasynth && pip install .
```

All install paths — PyPI, `pip install .` from a clone, `pip install dist/*.whl`, and `pip install -e .` for development — are verified end-to-end (see [MIGRATION.md](MIGRATION.md) for what "verified" means here, including a real packaging bug that was only caught by actually installing the built wheel).

```bash
pip install ".[viz,dev]"      # + seaborn, + pytest/ruff/mypy/build/twine
pip install ".[research]"     # + statsmodels/matplotlib — needed for `enerdatasynth evaluate`/`report`/`pipeline` and the classical-baseline benchmark, not for `.sample()` itself
```

## 🚀 Quick Start

Run inside Python (a `.py` file, `python3` interactive shell, or a notebook) — not pasted directly into your terminal's shell prompt:

```python
import enerdatasynth

load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
samples = load.sample(n=100, seed=0)   # (100, 24) ndarray, MW
```

Wind (any of `1ch`/`2ch`/`3ch` — all bundled, no extra setup needed):

```python
wind = enerdatasynth.model("wind", zone="1ch")
wind.set(wind_regime="mid", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=100, seed=0)   # (100, 288, 2) ndarray, m/s + degrees
```

That's the whole interface for generation. Training a new model, if you want to reproduce or extend the results below:

```bash
enerdatasynth preprocess   # raw CSVs -> tensors
enerdatasynth train        # transformer, validated defaults
enerdatasynth calibrate --run-id transformer_T1000_H64_L2
```

See [`docs/quickstart.md`](docs/quickstart.md) for the full walkthrough and [`examples/`](examples/) for runnable scripts.

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Raw PJM load CSVs\n+ NOAA temperature] --> B[Preprocessing\nclean · pivot · normalize]
    B --> C["Conditioning vector z\n(month, dow, holiday, Tmax, Tmin, spread)"]
    C --> D["Transformer denoiser\n(DDPM, corrected ancestral sampler)"]
    D --> E[Scalar bias\ncalibration]
    E --> F["ResourceModel.sample()\nreal MW output"]

    style D fill:#4c6ef5,color:#fff
    style E fill:#f59f00,color:#fff
```

Full detail — why transformer over MLP/CNN/attention, why the ancestral sampler over DDIM, why calibration is scalar not per-hour — is in [`docs/architecture.md`](docs/architecture.md), with every claim traced to a specific result file.

## 📊 Evaluation Pipeline

Every comparison in this project uses the same 7-metric suite (`Hourly`, `FirstDiff`, `ACF`, `RMSE`, `Peak`, `Energy`, `Ramp` — all lower-is-better, defined in [`docs/evaluation.md`](docs/evaluation.md)), computed by `enerdatasynth.evaluation.metrics.compute_metrics`. The same suite backs the automatic failure-mode detector (`evaluation.failure_analysis`) and the benchmark script below. A real per-hour distribution comparison (real vs. generated, currently auto-discovered default model) is at [`data/reports/figures/hourly_distribution_comparison_T100_H256_L4.png`](data/reports/figures/hourly_distribution_comparison_T100_H256_L4.png).

## 🏆 Benchmark Results

Run yourself: `python examples/run_benchmark.py --run-id <run_id>`. Numbers below are a real run against the full 600-day validation set, using the model `enerdatasynth.model("pjm_load")` actually auto-discovers today (the best currently-trained checkpoint under this package's own preprocessing pipeline):

| Model | Hourly↓ | FirstDiff↓ | ACF↓ | RMSE↓ | Peak↓ | Energy↓ | Ramp↓ |
|---|---|---|---|---|---|---|---|
| AR | 1.3533 | 0.4216 | 0.1458 | 0.1802 | 0.7557 | 4.2780 | 0.1242 |
| ARMA | 1.3581 | 0.4218 | **0.1081** | 0.1864 | 0.7557 | 4.3916 | 0.1239 |
| ARIMA | 1.3540 | 0.1698 | 0.3125 | 0.1813 | 0.3451 | 4.2945 | **0.0074** |
| SARIMA | 1.3625 | 0.6389 | 0.2489 | 0.1894 | 0.6834 | 4.4981 | 0.2969 |
| **enerdatasynth (default)** | **0.0590** | **0.0792** | 0.1347 | **0.0484** | **0.0362** | **0.8099** | 0.0647 |

Wins outright on 5 of 7 metrics against every classical baseline. ARMA edges it out on ACF, ARIMA on Ramp — reported honestly rather than rounded up to "wins everything." A separate, larger comparison against the validated transformer architecture (trained on a different conditioning scheme — see the caveat in [`docs/architecture.md`](docs/architecture.md)) shows an even larger margin; see [`docs/research_notes.md`](docs/research_notes.md) and the canonical version in [`docs/results.md`](docs/results.md).

## Zone Generalization

**17 PJM zones researched, 16 successful, 1 unresolved outlier.** The Week 7 pipeline (Transformer, corrected sampler, calibration) was replicated across 17 zones with only the load data, scaler parameters, and temperature column changing per zone.

- **Best zone:** AP (RMSE 0.0086)
- **16 of 17 zones** land in a tight RMSE range of 0.0086–0.0436
- **DOM is a clear, unresolved outlier** (RMSE 0.1998 — ~4.6× the next-worst zone). No root cause has been diagnosed. **This is not fixed**, and is reported here rather than omitted.
- **Only `PS` is exposed by the installed package today** — the 17-zone result is a research finding, not yet wired into `enerdatasynth.model("pjm_load", zone=...)`'s supported zone list.

Full per-zone table: [`docs/results.md`](docs/results.md#8-zone-expansion-phase-2).

## Data Card

Where the data comes from, how it's cleaned, and what's not verified about licensing (including an open flag on raw-data redistribution rights — read before any public push): [`docs/data_card.md`](docs/data_card.md).

## Model Card

Root [`MODEL_CARD.md`](MODEL_CARD.md) covers both `pjm_load` and wind. A deeper, audit-verified card for `pjm_load` specifically — including the exact current gap between validated research results and what the installed package loads by default — is at [`docs/model_card.md`](docs/model_card.md). One-page comparison cards for both architectures: [`docs/model_report_card.md`](docs/model_report_card.md).

## 📚 Documentation

| Page | Contents |
|---|---|
| [`docs/research_status.md`](docs/research_status.md) | **Start here.** The source of truth: what's scientifically established, and exactly where the package diverges from it |
| [`docs/results.md`](docs/results.md) | Canonical, traceable results tables |
| [`docs/data_card.md`](docs/data_card.md) | Dataset card — source, cleaning, licensing status |
| [`docs/model_card.md`](docs/model_card.md) / [`MODEL_CARD.md`](MODEL_CARD.md) | Model cards — training data, limitations, intended/out-of-scope use |
| [`docs/experiments.md`](docs/experiments.md) | Chronological map of the research program, Week 1 → Phase 2 |
| [`docs/model_report_card.md`](docs/model_report_card.md) | One-page Transformer vs. MLP comparison |
| [`docs/release_audit.md`](docs/release_audit.md) | This audit's findings, classified by severity, and remaining blockers |
| [`docs/installation.md`](docs/installation.md) | Install paths, extras, reproducing exact results, GPU |
| [`docs/quickstart.md`](docs/quickstart.md) | Full walkthrough: preprocess → train → calibrate → sample |
| [`docs/architecture.md`](docs/architecture.md) | Model, sampler, schedule, calibration — with evidence |
| [`docs/api_reference.md`](docs/api_reference.md) | The full `model()`/`.set()`/`.sample()` contract |
| [`docs/evaluation.md`](docs/evaluation.md) | The 7-metric suite, calibration, benchmarking |
| [`docs/temperature_data.md`](docs/temperature_data.md) | NOAA temperature ingestion, zone-station mapping |
| [`docs/training.md`](docs/training.md) | Training new checkpoints, sweeps, configs |
| [`docs/packaging.md`](docs/packaging.md) | How this package is built, versioned, and verified |
| [`docs/developer_guide.md`](docs/developer_guide.md) | Project layout, testing, contribution standards |
| [`docs/extending.md`](docs/extending.md) | Adding a new resource type (e.g. wind) |
| [`docs/faq.md`](docs/faq.md) | Common questions |
| [`data/research_history/README.md`](data/research_history/README.md) | Frozen provenance — every script behind every result number |
| [`MIGRATION.md`](MIGRATION.md) | Full audit trail behind every default in this package |

## 🧪 Examples

```bash
python examples/generate_samples.py           # LoadDiffusionModel round trip
python examples/run_full_pipeline.py           # tiny end-to-end demo
python examples/exploratory_data_analysis.py   # inspecting the processed dataset
```

## 🔬 Research Summary

- **A critical sampler bug, found and fixed.** An earlier sampler silently corrupted every diffusion result (values pinned near clamp bounds). Diagnosed and fixed; the corrected ancestral sampler is the only one shipped.
- **Architecture selected by controlled comparison, not assumption.** MLP vs CNN vs attention vs transformer, identical data/conditioning/schedule/seed. Transformer won on generated-profile quality at ~6x fewer parameters than the MLP it replaced (110,401 vs 688,408).
- **A methodological finding:** in that comparison, the *lowest validation loss* (CNN) was not the *best generated quality* (transformer) — a caution against judging generative models by training loss alone.
- **Calibration, evidence-checked rather than assumed better.** A more elaborate per-hour bias correction was tested and found *worse* on two metrics than the simpler scalar correction that's actually shipped.

Full detail, entirely traceable to specific files, is in [`docs/experiments.md`](docs/experiments.md) (chronological, Week 1 → Phase 2), [`docs/results.md`](docs/results.md) (results tables), [`docs/research_notes.md`](docs/research_notes.md), and [`MIGRATION.md`](MIGRATION.md).

## 📄 Citation

No accompanying paper has been published yet. If you use this software, please cite the repository:

```bibtex
@software{enerdatasynth,
  title  = {enerdatasynth: A Research SDK for Synthetic PJM Energy Resource Profiles},
  author = {Sharma, Sonakshi},
  year   = {2026},
  note   = {Software}
}
```

## ⚖️ License

[MIT](LICENSE) © 2026 Sonakshi Sharma

## 🗺️ Roadmap

- [ ] Re-validate the transformer/calibration pipeline under the package's current one-hot conditioning (see the caveat in [`docs/architecture.md`](docs/architecture.md))
- [ ] Multi-zone support (17 zones explored in `data/research_history/scripts/zone_expansion.py`, not yet promoted)
- [ ] A friendlier `set()` for `spread` (currently defaults to the training-set mean; see [`docs/api_reference.md`](docs/api_reference.md))
- [x] Wind generation for `1ch`/`2ch`/`3ch` (`enerdatasynth.model("wind", zone=...)`, all three bundled with real checkpoints)
- [ ] `units="normalized"` for wind (only `"m/s"` is supported today — `generate_multiple_for_condition` bakes denormalization into its output rather than exposing a raw option; a real gap, not prioritized)
- [ ] A proper weights-distribution mechanism for wind (`wind_diffusion.manifest.json`'s download-with-sha256-verification path, vs. today's ad hoc `data/checkpoints/wind/` bundling directly into git)
- [ ] Re-sweep hyperparameters for the transformer architecture specifically (the current defaults are inherited from `data/research_history/`, never re-swept against this package's own pipeline)

Tracked in detail in [`data/reports/future_experiments.md`](data/reports/future_experiments.md) and [`MIGRATION.md`](MIGRATION.md).

---

<div align="center">
<sub>Implementation details, folder structure, and the full conditioning-vector spec have moved to <a href="docs/developer_guide.md">docs/developer_guide.md</a> — this file is a landing page, not a developer reference.</sub>
</div>
