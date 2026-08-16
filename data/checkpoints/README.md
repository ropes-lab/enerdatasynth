# `data/checkpoints/`

The actual trained models this package ships — what lets `enerdatasynth.model(...)` work without you training anything first.

```
checkpoints/
├── pjm_load/
│   └── T50_H128_L4/ … T200_H128_L4/   # 7 legacy MLP sweep runs
│       ├── best.pt, latest.pt          #   (T=diffusion steps, H=hidden_dim, L=n_layers)
│       ├── result.json
│       └── loss_curve.png
└── wind/
    ├── 1ch/  model_checkpoint.pt, norm_stats.json
    ├── 2ch/  model_checkpoint.pt, norm_stats.json
    └── 3ch/  model_checkpoint.pt, norm_stats.json
```

One folder per resource, sitting side by side like everywhere else in `data/` — a third resource would just add one more sibling, not a parallel tree.

## PJM load checkpoints

Every `result.json` here predates the `"architecture"` field, so `from_run_id` treats them all as `"mlp"` — correctly, since all seven really are legacy MLP runs, not the Transformer. No Transformer checkpoint compatible with the current 23-dim conditioning exists in this repository yet. `_discover_best_checkpoint()` picks the best by validation loss (`T100_H256_L4` as of writing) — see [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) §5.

None of these have a `calibration.json`, so default sampling is uncalibrated. Run `enerdatasynth calibrate --run-id <run_id>` if you want one.

## Wind checkpoints

All three presets are bundled directly — this package's own convention (see [`generators/README.md`](../../src/enerdatasynth/generators/README.md)) — and get checked before the vendored pipeline's own external-download mechanism. Its manifest still says `"pending"`, but that's about its own download path, not these bundled files. See [`docs/models/wind.md`](../../docs/models/wind.md) for what's actually validated here.

## What should not be placed here

Anything not produced by `enerdatasynth train`/`calibrate` or the wind convention above. Code reads this folder by naming convention (`result.json` next to `best.pt`, or `wind/<preset>/model_checkpoint.pt`), not by scanning for arbitrary files.
