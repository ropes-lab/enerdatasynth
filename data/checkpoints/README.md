# `data/checkpoints/`

The actual trained models this package ships — what lets `enerdatasynth.model(...)` work without you training anything first.

```
checkpoints/
├── pjm_load/
│   ├── transformer_T1000_H64_L2/      # the shipped default: calibrated Transformer
│   │   ├── best.pt, latest.pt, result.json, loss_curve.png
│   │   └── calibration.json
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

The default is **`transformer_T1000_H64_L2`**, a Transformer trained on the current 23-dim conditioning and calibrated. `_discover_best_checkpoint()` prefers any `"transformer"` run over any MLP run, because validation losses aren't comparable across diffusion-step counts. The seven older runs have no `"architecture"` field in their `result.json`, so `from_run_id` treats them as `"mlp"` — correctly, since all seven really are legacy MLP runs. They stay as the comparison baseline. See [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) §5.

Only the Transformer has a `calibration.json`, applied automatically at generation. Run `enerdatasynth calibrate --run-id <run_id>` to add one to another checkpoint.

## Wind checkpoints

All three presets are bundled directly — this package's own convention (see [`generators/README.md`](../../src/enerdatasynth/generators/README.md)) — and get checked before the vendored pipeline's own external-download mechanism. Its manifest still says `"pending"`, but that's about its own download path, not these bundled files. See [`docs/models/wind.md`](../../docs/models/wind.md) for what's actually validated here.

## What should not be placed here

Anything not produced by `enerdatasynth train`/`calibrate` or the wind convention above. Code reads this folder by naming convention (`result.json` next to `best.pt`, or `wind/<preset>/model_checkpoint.pt`), not by scanning for arbitrary files.
