# `data/checkpoints/`

The actual trained models this package ships — what makes `enerdatasynth.model(...)`
work without training anything yourself first.

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

One folder per resource, sitting side by side exactly like everywhere else in `data/` —
a third resource adds one more sibling, not a parallel tree.

## PJM load checkpoints

Every `result.json` here predates the `"architecture"` field, so `from_run_id` treats
them as `"mlp"` — correct, **all seven are legacy MLP runs, not the Transformer.** No
Transformer checkpoint compatible with the current 23-dim conditioning exists in this
repository. `_discover_best_checkpoint()` picks the best by validation loss
(`T100_H256_L4` today) — see [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) §5.

None have a `calibration.json` — default sampling is uncalibrated. Run
`enerdatasynth calibrate --run-id <run_id>` to add one.

## Wind checkpoints

All three presets bundled directly (this package's own convention — see
[`generators/README.md`](../../src/enerdatasynth/generators/README.md)), checked
*before* the vendored pipeline's own external-download mechanism (its manifest still
says `"pending"` — that's about its own download path, not these files). See
[`docs/models/wind.md`](../../docs/models/wind.md) for what's actually validated.

## What should not be placed here

Anything not produced by `enerdatasynth train`/`calibrate` or the wind convention above
— read by naming convention (`result.json` next to `best.pt`, or
`wind/<preset>/model_checkpoint.pt`), not by scanning for arbitrary files.
