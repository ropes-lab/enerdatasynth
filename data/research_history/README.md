# Research History — Frozen Provenance

This directory is **intentionally frozen historical record**, not the public software
API. It exists so the path from the original MLP baseline to the current recommended
Transformer architecture is auditable and reproducible in spirit, not so any script here
can be run as-is against today's repository layout.

**Do not import anything from `research_history/` in the installed package.**
`src/enerdatasynth/__init__.py` states this explicitly, and it's true: the package works
completely independently of this directory.

## Why the weekly scripts exist, and why they're kept

Each script below was a single research session's self-contained pipeline — written to
run directly against that week's data on the original research machine, not designed as
a reusable library. They are preserved rather than deleted because:

- They are the actual evidence behind every number in `docs/results.md` and
  `docs/research_status.md`. Deleting them would turn traceable claims into assertions.
- The sampler bug (see below) is scientifically important precisely *because* it was a
  real mistake that was caught and fixed — the buggy version and the fix are both worth
  keeping, not just the final answer.
- A future contributor extending this work (a unified multi-zone model, wind, a paper)
  needs to see what was actually tried, not a cleaned-up retelling.

## Why they don't run as-is

Several scripts hardcode a machine-specific path
(`/Users/sonakshisharma/Downloads/data_for_everyone_pjm/...` or similar) and a
`week3_outputs/`-relative fallback that assumed the original flat project layout. That
data now lives under `research_history/outputs/week3_outputs/` in this repository. These
paths were **not rewritten** during this audit — per this project's provenance policy,
these files are a historical record of what was actually run, not maintained code. If you
need to re-run one, expect to fix its `ROOT`/`OUTDIR` constant first.

## Provenance Table

| Historical Artifact | Purpose | Status | Superseded By | Scientific Importance |
|---|---|---|---|---|
| `scripts/eda_exploration.py` | Merged `week1_testing.py` + `eda_plots.py`: two independent EDA passes over raw PJM RTO load data, kept verbatim including a real timezone-handling discrepancy between them | historical | `examples/exploratory_data_analysis.py` (demo against the processed dataset) | Establishes early data understanding; the two scripts' documented TZ discrepancy is worth knowing about if reusing either approach |
| `scripts/week3.py` | Original Week 3 pipeline: raw CSV → cleaned X/Z tensors, chronological 2016–2023/2024–2025 split, MLP+DDPM baseline architecture and training | historical | `src/enerdatasynth/data/preprocess.py`, `src/enerdatasynth/models/denoiser.py`, `src/enerdatasynth/diffusion/ddpm.py` | Establishes the original MLP baseline and the Z5 (6-dim)/Z9 (9-dim) conditioning representations still referenced throughout later work |
| `scripts/week5_benchmark.py` | Classical-baseline benchmark (AR/ARMA/ARIMA/SARIMA) against the Week 3 diffusion model, using the **original, later-found-buggy DDIM sampler** | historical — **produced invalidated numbers**, see `sampler_bugfix.py` | `src/enerdatasynth/evaluation/baselines.py` | Documents the pre-fix benchmark; its diffusion-model numbers should never be quoted as final results |
| `scripts/week5_part2.py` | Trains the Z5 baseline to full convergence (`converged_model_z5.pt`), separate from the under-trained `baseline_model_z5.pt` | historical | `src/enerdatasynth/training/train.py` | First "does more training help" experiment; motivated the sampler-bug investigation when convergence made results *worse* on some metrics |
| `scripts/sampler_bugfix.py` | **Load-bearing.** Diagnoses whether the converged model's worse FirstDiff/RMSE is a sampler artifact or a real model property (Part 1), then ships the fix (Part 2): the corrected ancestral sampler | **load-bearing — historical record of a critical fix** | `src/enerdatasynth/diffusion/ddpm.py` (the corrected sampler shipped here is exactly what the package uses today) | **Critical.** Everything in `docs/results.md` is a post-fix number; this script (and the two `week5_comparison_*.csv` files) is the only record of the invalidated pre-fix numbers and why they're invalid |
| `scripts/week6.py` | Two experiments in one: conditional-subset evaluation (temperature/calendar/season slices) and the noise-schedule comparison (linear/cosine/quadratic β) | historical experiment | `docs/results.md` §4/§5 | Established that linear β schedule wins; first systematic look at where the model struggles |
| `scripts/arch_compare.py` | Controlled architecture comparison: MLP / CNN / attention / Transformer, everything else held identical | historical experiment | `src/enerdatasynth/models/transformer.py` (the winning architecture) | **Architecture-selection evidence.** This is the source of the "why Transformer" numbers in `docs/results.md` §2, including the unexplained CNN 118-minute training-time anomaly |
| `scripts/week7.py` | Promotes the Transformer to primary baseline: trains to convergence, re-runs the full head-to-head + conditional evaluation | **final research evidence — primary result** | `src/enerdatasynth/models/transformer.py`, `src/enerdatasynth/utils/config.py` (`DEFAULT_ARCHITECTURE = "transformer"`) | **Primary result.** Source of `docs/results.md` §1/§3/§4 |
| `scripts/week7_calibrate.py` | Diagnoses and fixes the Transformer's energy-metric regression via scalar bias correction | **final research evidence** | `src/enerdatasynth/evaluation/calibration.py` | Source of `docs/results.md` §6 and `docs/research_status.md` §F |
| `scripts/week7_erroranalysis.py` | Confirms the energy bias is a constant level offset (not a deeper issue); ranks validation days by RMSE to find best/worst cases | **final research evidence** | `src/enerdatasynth/evaluation/failure_analysis.py` | Source of `docs/results.md` §7 — the hot-day weakness finding |
| `scripts/zone_expansion.py` | Phase 2: replicates the PS pipeline across 17 PJM zones, same architecture/sampler/calibration, only data/scalers differ | **final research evidence — generalization result** | Not yet wired into the package (`pjm_load` only exposes zone `PS`, see `docs/research_status.md` §B) | Source of `docs/results.md` §8 — 16/17 zones generalize, DOM is an unresolved outlier |

## Where things live

- **Corrected sampler results:** `reports/week5_comparison_corrected.csv` (post-fix) vs.
  `reports/week5_comparison_table.csv` / `reports/week5_metrics.json` (pre-fix, invalid —
  kept only so the bug's impact is verifiable).
- **Transformer results:** `reports/week7/summary_report.md`, `reports/week6/arch/`.
- **Zone expansion results:** `reports/phase2/summary_report.md`.
- **Figures:** `reports/figures/`, `reports/eda_figures/`, `reports/week6/*/`,
  `reports/week7/*.png`.
- **Trained artifacts from this lineage:** `outputs/week3_outputs/` (baseline/converged
  MLP checkpoints, architecture-comparison checkpoints, noise-schedule checkpoints) and
  `outputs/zone_outputs/<ZONE>/` (per-zone Phase 2 artifacts).

## What's superseded vs. what's still the reference

The **package** (`src/enerdatasynth/`) is the maintained, reusable implementation of the
Week 7 Transformer + corrected sampler + calibration pipeline — read that for how to
*use* this work. This directory is the maintained *evidence* for why that implementation
looks the way it does — read this for *why*. See `docs/research_status.md` §D for the one
important place these two currently diverge: **no checkpoint compatible with the
package's current 23-dimensional conditioning has been trained**, so the package does not
yet ship the validated Transformer result described here.

**If you're looking for a readable, final summary of these results — not the scripts
that produced them — you want [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md),
not this directory.** This is provenance for people extending or auditing the research;
the model report is the primary documentation for people using or evaluating the
package. Nothing in `src/enerdatasynth/` imports from here, and nothing here should be
run as a way to "use" the package — see [Where things live](#where-things-live) above
for the one-time historical scripts, and the root `README.md` / model reports for the
actual product.
