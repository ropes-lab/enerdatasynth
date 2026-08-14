# Hyperparameter Sweep Summary

Sweep type: `one_factor_at_a_time_9`
Epochs per configuration: 60

## Ranked Configurations (by best validation loss)

| Rank | Config | T | Hidden | Layers | Params | Best Val Loss | Train Time (s) | Gen Time /64 (s) |
|------|--------|---|--------|--------|--------|----------------|----------------|-------------------|
| 1 | T100_H256_L4 | 100 | 256 | 4 | 224,024 | 0.08164 | 6.7 | 0.03 |
| 2 | T200_H128_L4 | 200 | 128 | 4 | 62,872 | 0.09167 | 3.6 | 0.08 |
| 3 | T100_H128_L4 | 100 | 128 | 4 | 62,872 | 0.10624 | 4.0 | 0.02 |
| 4 | T100_H128_L2 | 100 | 128 | 2 | 29,848 | 0.12487 | 2.0 | 0.02 |
| 5 | T50_H128_L4 | 50 | 128 | 4 | 62,872 | 0.13748 | 4.9 | 0.03 |
| 6 | T100_H64_L4 | 100 | 64 | 4 | 19,160 | 0.19624 | 6.2 | 0.02 |
| 7 | T100_H128_L6 | 100 | 128 | 6 | 95,896 | 0.38388 | 5.9 | 0.03 |

## Best Configuration

**T100_H256_L4** — diffusion_steps=100, hidden_dim=256, n_layers=4
- Best validation loss: 0.08164 (epoch 40)
- Parameters: 224,024
- Training time: 6.7s

## Notes on Sweep Design

This sweep isolates one hyperparameter axis at a time (diffusion steps, then hidden dimension, then layer count) rather than running the full 27-cell grid, to keep baseline compute reasonable. The full grid can be run with `run_sweep(full_grid=True)` if a more exhaustive search is needed later.

## Trade-offs Observed

**Diffusion steps** (hidden_dim and n_layers held at defaults):
- T=50: val_loss=0.13748, generation time per 64 samples = 0.03s
- T=100: val_loss=0.10624, generation time per 64 samples = 0.02s
- T=200: val_loss=0.09167, generation time per 64 samples = 0.08s

More diffusion steps generally costs more generation time linearly (more reverse steps to run) — check whether the validation loss improvement justifies the slower sampling.

**Hidden dimension** (T and n_layers held at defaults):
- hidden_dim=64: val_loss=0.19624, params=19,160
- hidden_dim=128: val_loss=0.10624, params=62,872
- hidden_dim=256: val_loss=0.08164, params=224,024

**Number of layers** (T and hidden_dim held at defaults):
- n_layers=2: val_loss=0.12487, params=29,848
- n_layers=4: val_loss=0.10624, params=62,872
- n_layers=6: val_loss=0.38388, params=95,896

Note: deeper is not always better on this dataset — more layers can slow or destabilize convergence for a problem this small, as the L6 result shows.