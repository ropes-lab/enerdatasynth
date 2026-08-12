# Results Summary

## Dataset Summary

- Zone: PS
- Training samples: 2785
- Validation samples: 600
- Total raw days before cleaning: 3653
- X shape (train): [2785, 24]
- Z shape (train): [2785, 23], dim=23

## Model Architecture

Simple feedforward MLP denoiser (no attention, no transformers, no Fourier features). Inputs: noisy profile x_t, diffusion timestep t (sinusoidal embedding), condition vector z (concatenated). Output: predicted noise. Loss: standard diffusion MSE only.

## Training Results

Best configuration: **T100_H256_L4**
- Diffusion steps: 100
- Hidden dimension: 256
- Layers: 4
- Parameters: 224,024
- Best validation loss: 0.08164 (epoch 40)
- Training time: 6.7s
- Generation time (64 samples): 0.03s

## Best Hyperparameters

See `hyperparameter_results.csv` and `hyperparameter_summary.md` for the full ranked sweep.

## Validation Results

Three validation plots generated per configuration:
1. `hourly_distribution_comparison_<run_id>.png` — real vs generated boxplots per hour
2. `first_difference_comparison_<run_id>.png` — hour-over-hour delta distributions
3. `autocorrelation_comparison_<run_id>.png` — ACF comparison

All comparisons use IN-SAMPLE validation: generated profiles are produced from the SAME conditioning vectors as the real validation set, as requested.

## Failure Modes

# Failure Analysis

Run: `T100_H256_L4`

## Ramp rate mismatch (too large)

**Observed issue:** Real ramp std = 0.0274, generated ramp std = 0.1070 (+289.8% relative error)

**Possible cause:** Generated profiles are noisier than real load — incomplete denoising or too few reverse steps.

**Potential future fix:** Train longer; consider whether hidden_dim is large enough to capture hour-to-hour transition structure; verify noise schedule (beta_start/beta_end) is appropriate for this data's scale.

## Seasonal mismatch

**Observed issue:** Winter: real mean=0.3285, generated mean=0.2931 (-10.8%); Summer: real mean=0.5118, generated mean=0.4356 (-14.9%)

**Possible cause:** Model is not fully utilizing the one-hot month conditioning — possibly due to limited training epochs, or the conditioning signal being too weak relative to the noise schedule at the early-training stage.

**Potential future fix:** Train to convergence; verify the conditioning vector is reaching the network correctly; consider per-season validation metrics during training to catch this earlier.


## Recommended Next Experiments

See `reports/future_experiments.md` for the full list. Per the PI's direction, these are NOT implemented yet:
- One-hot vs cyclic encoding
- Attention layers
- MW anchor (multi-day continuity)
- Unified PJM model (all zones)
- Advanced conditioning (humidity, hourly temperatures)
