# `data/research_history/pjm_load/reports/`

The actual historical research reports — markdown summaries, CSVs, and figures — that `docs/results.md` and `docs/research_status.md` were written by reading directly.

| Folder/file | What it is |
|---|---|
| `week5_comparison_table.csv` / `week5_metrics.json` | Pre-sampler-fix benchmark numbers, invalid, kept only as evidence of the bug's impact |
| `week5_comparison_corrected.csv` | Post-fix benchmark numbers, the valid ones |
| `week6/arch/`, `week6/noise/`, `week6/conditional/` | Architecture comparison, noise-schedule experiment, conditional evaluation |
| `week7/` | Final Transformer results, calibration report, error analysis |
| `phase2/` | 17-zone generalization results |
| `figures/`, `eda_figures/` | Supporting plots |

If you want the results, read [`docs/results.md`](../../../../docs/results.md). It was built by cross-checking every number in this folder against the actual code, and tells you which numbers are current/valid versus historical-only. This folder is the primary source evidence, not the primary read.
