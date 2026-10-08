# Failure Analysis

Run: `transformer_T1000_H64_L2`

## Seasonal mismatch

**Observed issue:** Winter: real mean=0.3285, generated mean=0.2925 (-11.0%)

**Possible cause:** Model is not fully utilizing the one-hot month conditioning — possibly due to limited training epochs, or the conditioning signal being too weak relative to the noise schedule at the early-training stage.

**Potential future fix:** Train to convergence; verify the conditioning vector is reaching the network correctly; consider per-season validation metrics during training to catch this earlier.
