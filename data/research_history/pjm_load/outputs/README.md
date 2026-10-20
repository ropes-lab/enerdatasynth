# `data/research_history/pjm_load/outputs/`

Trained artifacts produced by the scripts in [`../scripts/`](../scripts/README.md): the `.pt` weights, per-zone `scaler.json` and `calibration.json` files, and loss plots behind the numbers in `docs/results.md`.

- **`week3_outputs/`** — the original MLP/DDPM baseline and the Week 6 architecture and noise-schedule comparison checkpoints (`arch_*.pt`, `noise_*.pt`).
- **`zone_outputs/<ZONE>/`** — Phase 2's per-zone weights, scalers and results, one folder per researched PJM zone (17 in total, including `DOM`, the unresolved outlier; see `docs/results.md` §8).

**The datasets these runs trained on are not included.** The tensors (`X_train.npy`, `Z5_train.npy` and similar) and tables were built from PJM and NOAA data and have been removed from this repository; sources are in [`data/README.md`](../../../README.md#data-sources). The weights are kept as provenance, and nothing in `src/enerdatasynth/` reads from here. Kept for the research record; see [`../../README.md`](../../README.md).
