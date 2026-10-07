# Results Summary

## Dataset Summary

- Zone: PS
- Training samples: 2785
- Validation samples: 600
- Total raw days before cleaning: 3653
- X shape (train): [2785, 24]
- Z shape (train): [2785, 23], dim=23

## Shipped Default Checkpoint

Run: **transformer_T1000_H64_L2** (architecture: transformer)
- Diffusion steps: 1000
- Hidden dimension: 64, layers: 2
- Parameters: 111,489
- Best validation loss: 0.00946 (epoch 123 of 163 trained)
- Calibration: scalar offset +0.02342, fit on 1200 training days

## Model Architecture

Transformer-encoder denoiser. Each of the 24 hours is one token with a learned positional embedding; the diffusion timestep (sinusoidal embedding) and the condition vector are projected through a shared MLP and added to every token; a standard `nn.TransformerEncoder` (GELU, 4x feedforward) and a linear head predict the noise. Loss: standard diffusion MSE only.

## Hyperparameter Sweep

Best configuration in `hyperparameter_results.csv`, ranked by validation loss. This is the sweep's own winner and is separate from the shipped default described above (a Transformer is selected over any MLP run regardless of validation loss, since losses are not comparable across diffusion-step counts).

Best sweep configuration: **T100_H256_L4**
- Diffusion steps: 100
- Hidden dimension: 256
- Layers: 4
- Parameters: 224,024
- Best validation loss: 0.08164 (epoch 40)
- Training time: 6.7s
- Generation time (64 samples): 0.03s

See `hyperparameter_results.csv` and `hyperparameter_summary.md` for the full ranked sweep.

## Validation Results

Three validation plots generated per configuration:
1. `hourly_distribution_comparison_<run_id>.png` — real vs generated boxplots per hour
2. `first_difference_comparison_<run_id>.png` — hour-over-hour delta distributions
3. `autocorrelation_comparison_<run_id>.png` — ACF comparison

All comparisons use IN-SAMPLE validation: generated profiles are produced from the SAME conditioning vectors as the real validation set, as requested.

## Failure Modes

# Failure Analysis

Run: `transformer_T1000_H64_L2`

## Seasonal mismatch

**Observed issue:** Winter: real mean=0.3285, generated mean=0.2925 (-11.0%)

**Possible cause:** Model is not fully utilizing the one-hot month conditioning — possibly due to limited training epochs, or the conditioning signal being too weak relative to the noise schedule at the early-training stage.

**Potential future fix:** Train to convergence; verify the conditioning vector is reaching the network correctly; consider per-season validation metrics during training to catch this earlier.


## Recommended Next Experiments

See `reports/future_experiments.md` for the full list. Not yet implemented in this package:
- Re-sweeping hyperparameters for the transformer architecture (the current defaults come from research_history, not a from-scratch sweep against the package's own conditioning pipeline)
- One-hot vs the research lineage's raw (non-one-hot) conditioning encoding
- MW anchor (multi-day continuity)
- Unified PJM model (all zones)
- Advanced conditioning (humidity, hourly temperatures)
