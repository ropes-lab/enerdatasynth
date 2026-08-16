# `data/research_history/pjm_load/outputs/`

Trained artifacts and datasets produced by the scripts in [`../scripts/`](../scripts/README.md) — the actual `.pt` checkpoints, `.npy` tensors, and `.parquet` files behind every number in `docs/results.md`.

- **`week3_outputs/`** — the original MLP/DDPM baseline, the Z5 (6-dim)/Z9 (9-dim) conditioning tensors, and the Week 6 architecture/noise-schedule comparison checkpoints (`arch_*.pt`, `noise_*.pt`).
- **`zone_outputs/<ZONE>/`** — Phase 2's per-zone datasets and results, one folder per researched PJM zone (17 total, including `DOM`, the unresolved outlier — see `docs/results.md` §8).

These are real historical data files, not the package's own bundled data (`data/processed/pjm_load/`, `data/checkpoints/pjm_load/`) — nothing in `src/enerdatasynth/` reads from here. Kept for provenance and reproducibility of the research record; see [`../../README.md`](../../README.md).
