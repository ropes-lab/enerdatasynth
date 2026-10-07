# Wind Results Summary

Generated from `resources/wind/manifest.json` — real numbers measured by the upstream project, not a new comparison run in this repository (there is no real wind data here to compare against; see [`docs/models/wind.md`](../../../docs/models/wind.md) §2).

## Preset `1ch`

- Channels: windspeed only
- Sequence length: 288 steps @ 5min
- Measured: 2026-07-21
- Training recipe: Post-LN (norm_first=False), hidden_dim=128, warmup_epochs=8, 60 epochs, cond_dropout_prob=0.3, cond_dropout_scope=wind_regime
- Generation settings: sampler=ddim, steps=50, eta=1.0, guidance_scale=40.0, guidance_scope=wind_regime, dynamic_threshold=true, dynamic_threshold_percentile=99.0

| Regime | marginal_wasserstein | acf_rmse | regime_accuracy |
|---|---|---|---|
| low | 2.247 | 0.060 | 0.69 |
| mid | 1.999 | 0.108 | 0.41 |
| high | 2.109 | 0.101 | 0.55 |

**Flagged issues:** 2 (see `failure_analysis_1ch.md`)

## Preset `2ch`

- Channels: windspeed decomposed into wind-vector U/V components
- Sequence length: 288 steps @ 5min
_No structured metrics for this preset in the manifest — its real, measured history is only in the known-limitation narrative:_

> Substantially improved 2026-07-21, but still not a fully solved preset -- do not treat as equivalent to 1ch/3ch. History: originally measured BROKEN (synthetic speed ~2.8x inflated vs real, 100% of days landing in the 'high' bucket regardless of request) and root-caused to a sampling-time clip-boundary instability specific to jointly modeling a 2D vector magnitude (U/V) -- see reports/2CH_CLAMP_INVESTIGATION.md. A training-time joint-consistency loss (joint_consistency_weight=1.0) gave a partial fix (2.8x -> 1.4x inflation, plateaued). Dynamic thresholding, layered on top of that same checkpoint, closes most of the remaining gap: marginal_wasserstein 8.0/4.2/3.2 (joint-loss alone) -> 4.5/1.5/4.4 (+ dynamic thresholding), now in the same range as 1ch/3ch. Guidance still does not compose with this checkpoint even with dynamic thresholding on: at guidance_scale=40, ramp_wasserstein explodes (3.5/1.9/0.3) and 'mid' requests resolve to 'low' 83% of the time. Use this preset only without guidance. Separately, eta=1.0 (the stochasticity fix that broadly helped 1ch/3ch) was tested here and is MIXED: low/mid improved (marginal_wasserstein 4.5/1.5 -> 3.0/1.0, regime_accuracy 0.31/0.43 -> 0.35/0.56) but high got clearly WORSE (marginal_wasserstein 4.4 -> 5.9, regime_accuracy 0.32 -> 0.12, confusion shows 'high' requests increasingly resolving to 'mid'). Not adopted as this preset's default pending further investigation -- recommended_generation keeps eta=0.0 for 2ch. See reports/BASELINE_2CH_DYNTHRESH_JOINTLOSS_GS1.md vs reports/BASELINE_2CH_DYNTHRESH_JOINTLOSS_GS40.md (guidance comparison) and reports/BASELINE_2CH_DYNTHRESH_ETA1_GS1.md (eta comparison).

**Flagged issues:** 1 (see `failure_analysis_2ch.md`)

## Preset `3ch`

- Channels: windspeed plus sin/cos direction encoding
- Sequence length: 288 steps @ 5min
- Measured: 2026-07-21
- Training recipe: Post-LN (norm_first=False), hidden_dim=128, warmup_epochs=8, 60 epochs, cond_dropout_prob=0.3, cond_dropout_scope=wind_regime (same recipe as 1ch)
- Generation settings: sampler=ddim, steps=50, eta=1.0, guidance_scale=40.0, guidance_scope=wind_regime, dynamic_threshold=true, dynamic_threshold_percentile=99.0

| Regime | marginal_wasserstein | acf_rmse | regime_accuracy |
|---|---|---|---|
| low | 2.665 | 0.074 | 0.41 |
| mid | 2.213 | 0.109 | 0.43 |
| high | 1.158 | 0.118 | 0.82 |

**Flagged issues:** 3 (see `failure_analysis_3ch.md`)
