# `data/checkpoints/`

The actual trained models this package ships — this is the folder that makes
`enerdatasynth.model(...)` work without training anything yourself first.

## Layout

```
checkpoints/
├── T50_H128_L4/ … T200_H128_L4/   # 7 legacy MLP hyperparameter-sweep runs
│   ├── best.pt, latest.pt          #   (see run_id naming: T=diffusion steps,
│   ├── result.json                 #    H=hidden_dim, L=n_layers)
│   └── loss_curve.png
└── wind/
    ├── 1ch/  model_checkpoint.pt, norm_stats.json
    ├── 2ch/  model_checkpoint.pt, norm_stats.json
    └── 3ch/  model_checkpoint.pt, norm_stats.json
```

## PJM load checkpoints

Every `result.json` here predates the `"architecture"` field — `generators/load.py`'s
`from_run_id` treats a missing field as `"mlp"`, which is correct: **all seven of these
are legacy MLP sweep runs, not the recommended Transformer.** No Transformer checkpoint
compatible with the current 23-dimensional conditioning exists anywhere in this
repository. `generators/load_conditioning.py`'s `_discover_best_checkpoint()` picks the
best of these by validation loss (`T100_H256_L4` today) — see
[`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) §5 for exactly what this
means for what you actually get from `enerdatasynth.model("pjm_load", ...)`.

None of these seven have a `calibration.json` next to them — default sampling is
uncalibrated. Run `enerdatasynth calibrate --run-id <run_id>` to add one.

## Wind checkpoints

Three presets (`1ch`/`2ch`/`3ch`), each with a real trained checkpoint bundled directly
(this package's own convention — see
[`src/enerdatasynth/generators/README.md`](../../src/enerdatasynth/generators/README.md)).
These are checked *before* `wind_diffusion`'s own external-download mechanism, which
would otherwise report every preset as unavailable (its bundled manifest still says
`"pending"` — that's about its own download path, not these files). See
[`docs/models/wind.md`](../../docs/models/wind.md) for what's actually validated about
each preset.

## What should not be placed here

Anything not produced by `enerdatasynth train`/`calibrate` or the wind checkpoint
convention above — this folder is read by naming convention (`result.json` next to
`best.pt`, or `wind/<preset>/model_checkpoint.pt`), not by scanning for arbitrary files.
