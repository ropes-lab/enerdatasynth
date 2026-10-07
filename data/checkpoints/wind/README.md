# `data/checkpoints/wind/`

Three pre-trained presets, each a folder with `model_checkpoint.pt` and `norm_stats.json`:

| Preset | Channels | Parameters | Saved epoch | Val loss |
|---|---|---|---|---|
| `1ch` | windspeed only | 1,179,073 | 57 of 60 | 0.0557 |
| `2ch` | wind-vector U/V | 1,179,586 | 57 of 60 | 0.0648 |
| `3ch` | speed + sin/cos direction | 1,180,099 | 57 of 60 | 0.0495 |

All three share one architecture: a 3-layer Transformer denoiser, `hidden_dim=128`, `num_heads=4`, cosine beta schedule, 288-step sequences. A checkpoint file holds `model_state_dict`, `config`, `epoch`, and `val_loss`; `norm_stats.json` holds the target and conditioning normalization plus the wind-regime bucket edges.

These were trained in the upstream `wind_diffusion` project, not here, and this repository holds no wind training data or logs, so none of these numbers can be reproduced locally. The manifest still lists each preset's status as `pending` with no download URL or checksum, because the bundled files are this package's own convention. See the [category README](../README.md) and [`docs/models/wind.md`](../../../docs/models/wind.md).
