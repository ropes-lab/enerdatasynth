# `data/checkpoints/pjm_load/`

Seven legacy MLP sweep runs, all trained on the 23-dim one-hot conditioning in `data/processed/pjm_load/`. Folder names encode `T<diffusion steps>_H<hidden dim>_L<layers>`. Each holds `best.pt`, `latest.pt`, `result.json`, and `loss_curve.png`.

| Run | Parameters | Best val loss |
|---|---|---|
| `T100_H256_L4` | 224,024 | **0.0816** (selected by default) |
| `T200_H128_L4` | 62,872 | 0.0917 |
| `T100_H128_L4` | 62,872 | 0.1062 |
| `T100_H128_L2` | 29,848 | 0.1249 |
| `T50_H128_L4` | 62,872 | 0.1375 |
| `T100_H64_L4` | 19,160 | 0.1962 |
| `T100_H128_L6` | 95,896 | 0.3839 |

None has a `calibration.json`, and none is a Transformer; no Transformer checkpoint compatible with the current conditioning exists yet. The 110,401 vs. 688,408 parameter comparison quoted in the docs is from the Week 6 research configuration (6-dim conditioning), not these bundled models. See the [category README](../README.md) and [`docs/models/pjm_load.md`](../../../docs/models/pjm_load.md) §5.
