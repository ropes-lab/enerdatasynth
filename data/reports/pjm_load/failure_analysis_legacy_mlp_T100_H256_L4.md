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
