# Research History — Frozen Provenance

A frozen historical record, not the public API. It's preserved because these scripts are the actual evidence behind every number in `docs/results.md`/`docs/research_status.md`, and because the sampler bug below matters precisely because it was a real mistake that got caught — both the buggy version and the fix are worth keeping around.

Don't import this from the package (`src/enerdatasynth/__init__.py` confirms it doesn't), and don't try to run these scripts as-is — several hardcode a machine-specific path from the original research machine. If you need to re-run one, fix its `ROOT`/`OUTDIR` constant first.

If you want the results rather than the scripts, read [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) instead. This directory is for auditing why the package looks the way it does, not for using it directly.

## Provenance Table

| Artifact | Purpose | Status | Superseded By |
|---|---|---|---|
| `scripts/eda_exploration.py` | Early EDA over raw PJM load data | historical | `examples/exploratory_data_analysis.py` |
| `scripts/week3.py` | Original MLP+DDPM baseline; raw CSV → tensors, chronological split | historical | `resources/pjm_load/data/preprocess.py`, `models/denoiser.py`, `diffusion/ddpm.py` |
| `scripts/week5_benchmark.py` | Classical-baseline benchmark using the later-found-buggy DDIM sampler | invalidated, see `sampler_bugfix.py` | `resources/pjm_load/evaluation/baselines.py` |
| `scripts/week5_part2.py` | Trains the baseline to full convergence | historical | `resources/pjm_load/training/train.py` |
| `scripts/sampler_bugfix.py` | Load-bearing — diagnoses and fixes the sampler bug | critical fix, still in effect | `resources/pjm_load/diffusion/ddpm.py` |
| `scripts/week6.py` | Conditional-slice evaluation and noise-schedule comparison | historical experiment | `docs/results.md` §4/§5 |
| `scripts/arch_compare.py` | Controlled MLP/CNN/attention/Transformer comparison | historical experiment | `resources/pjm_load/models/transformer.py` |
| `scripts/week7.py` | Promotes Transformer to primary baseline, full re-evaluation | primary result | `docs/results.md` §1/§3/§4 |
| `scripts/week7_calibrate.py` | Fixes the energy-metric regression via bias correction | final evidence | `resources/pjm_load/evaluation/calibration.py` |
| `scripts/week7_erroranalysis.py` | Confirms energy bias is a constant offset; best/worst-day analysis | final evidence | `docs/results.md` §7 |
| `scripts/zone_expansion.py` | Replicates the pipeline across 17 PJM zones | generalization result | Not wired into the package (`pjm_load` ships zone `PS` only) |

## Where things live

- **Sampler fix evidence:** `reports/week5_comparison_corrected.csv` (post-fix) vs. `reports/week5_comparison_table.csv` (pre-fix, invalid, kept for verifiability).
- **Transformer / zone results:** `reports/week7/`, `reports/week6/arch/`, `reports/phase2/`.
- **Trained artifacts:** `outputs/week3_outputs/`, `outputs/zone_outputs/<ZONE>/` hold the model weights, scalers and calibration files. The datasets these scripts used are not in the repository; see [`data/README.md`](../README.md#data-sources).

## The one place this diverges from the package

The research runs used a 6-dimensional conditioning vector, not the package's 23-dimensional one. A Transformer trained on the 23-dimensional conditioning (`transformer_T1000_H64_L2`) has since been added and is the shipped default; see `docs/research_status.md` §D. The package (`src/enerdatasynth/`) is the maintained implementation; this directory is the evidence for why it's built the way it is, not a second copy of the product.

## Wind

[`wind/`](wind/README.md) is deliberately empty. Wind's research history belongs to the separate upstream project this package vendors, not to this one — this folder exists purely so the "one folder per resource" pattern holds everywhere in `data/`, including here.
