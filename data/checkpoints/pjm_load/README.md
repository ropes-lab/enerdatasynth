# `data/checkpoints/pjm_load/`

Eight runs, all trained on the 23-dim one-hot conditioning in `data/processed/pjm_load/`. Each holds `best.pt`, `latest.pt`, `result.json`, and `loss_curve.png`.

## The shipped default

**`transformer_T1000_H64_L2`** is what `enerdatasynth.model("pjm_load")` loads. It's a Transformer denoiser with 111,489 parameters, trained with `data/configs/pjm_load/transformer.yaml` (seed 42) in 144 seconds on CPU. It early-stopped at epoch 163, with best validation loss 0.00946 at epoch 123. `calibration.json` holds a scalar offset of +0.02342, fit on 1,200 training days and applied automatically at generation.

On the 600 validation days it beats the legacy MLP below on all seven metrics in every one of 5 generation seeds, before and after calibration (`data/reports/pjm_load/checkpoint_comparison.csv`, reproduce with `examples/compare_checkpoints.py`). It was trained once, with research-validated hyperparameters rather than a from-scratch sweep.

## Legacy MLP sweep runs

Folder names encode `T<diffusion steps>_H<hidden dim>_L<layers>`. They have no `calibration.json`.

| Run | Parameters | Best val loss |
|---|---|---|
| `T100_H256_L4` | 224,024 | **0.0816** (best of the sweep) |
| `T200_H128_L4` | 62,872 | 0.0917 |
| `T100_H128_L4` | 62,872 | 0.1062 |
| `T100_H128_L2` | 29,848 | 0.1249 |
| `T50_H128_L4` | 62,872 | 0.1375 |
| `T100_H64_L4` | 19,160 | 0.1962 |
| `T100_H128_L6` | 95,896 | 0.3839 |

Validation losses aren't comparable across diffusion-step counts, which is why the default is chosen by architecture preference plus generated-profile metrics rather than by loss alone. The 110,401 vs. 688,408 parameter comparison quoted in the docs is from the Week 6 research configuration (6-dim conditioning), not these bundled models.

See the [category README](../README.md) and [`docs/models/pjm_load.md`](../../../docs/models/pjm_load.md) §5.
