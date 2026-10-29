# `data/checkpoints/`

The trained models this package ships. They are what lets `enerdatasynth.model(...)` work without you training anything first.

```
checkpoints/
├── pjm_load/
│   ├── transformer_T1000_H64_L2/      # the shipped default: calibrated Transformer
│   │   ├── best.pt, latest.pt, result.json, loss_curve.png
│   │   └── calibration.json
│   ├── T50_H128_L4/ ... T200_H128_L4/ # 7 legacy MLP sweep runs (PS)
│   │   ├── best.pt, latest.pt          #   (T=diffusion steps, H=hidden_dim, L=n_layers)
│   │   ├── result.json
│   │   └── loss_curve.png
│   └── zones/<ZONE>/transformer_T1000_H64_L2/   # the other 16 PJM zones (AE ... RECO)
│       └── best.pt, result.json, calibration.json
└── wind/
    ├── 1ch/  model_checkpoint.pt, norm_stats.json
    ├── 2ch/  model_checkpoint.pt, norm_stats.json
    └── 3ch/  model_checkpoint.pt, norm_stats.json
```

Each resource gets one folder, side by side like everywhere else in `data/`. A third resource would add one more sibling, not a parallel tree. These are weights and normalization statistics, not data.

## PJM load

Every one of the 17 PJM zones has a **`transformer_T1000_H64_L2`**, a calibrated Transformer trained on the current 23-dimensional conditioning and on that zone's own data. Zone `PS` keeps the flat folder above (together with the seven legacy MLP runs, which exist for PS only); the other 16 zones are in `zones/<ZONE>/`, each holding just the three files the package needs to generate (`best.pt`, `result.json`, `calibration.json`). The rest of this section describes PS; the other zones are built the same way (`examples/pjm_load/train_all_zones.py`). `_discover_best_checkpoint()` prefers any `"transformer"` run over any MLP run, because validation losses can't be compared across different diffusion-step counts. The seven older runs have no `"architecture"` field in their `result.json`, so `from_run_id` treats them as `"mlp"`, which is correct: all seven are legacy MLP runs. They stay as the comparison baseline. See [`docs/models/pjm_load.md`](../../docs/models/pjm_load.md) §5.

Only the Transformer has a `calibration.json`, and it is applied automatically at generation. To add one to another checkpoint, run `enerdatasynth calibrate --run-id <run_id>`, which needs the processed arrays from your own `enerdatasynth preprocess`.

## Wind

All three presets are bundled here, following this package's own convention (see [`generators/README.md`](../../src/enerdatasynth/generators/README.md)), and they are checked before the vendored pipeline's own external-download mechanism. That mechanism's manifest still says `"pending"`, but that refers to its download path, not to these bundled files. See [`docs/models/wind.md`](../../docs/models/wind.md) for what has and hasn't been validated.

## What doesn't belong here

Anything not produced by `enerdatasynth train` or `calibrate`, or by the wind convention above. Code finds files here by naming convention (`result.json` next to `best.pt`, or `wind/<preset>/model_checkpoint.pt`), not by scanning for whatever is present.
