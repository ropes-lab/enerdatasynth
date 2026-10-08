# Wind Failure Analysis

Preset: `2ch`

**Not a new measurement** — flags issues already present in this preset's `manifest.json` entry (real numbers from the upstream project's own validation), against documented thresholds. See [`docs/models/wind.md`](../../../docs/models/wind.md) §8 for the full
per-regime table this is derived from.

## Documented known limitation (from manifest.json)

**Observed issue:** Substantially improved 2026-07-21, but still not a fully solved preset -- do not treat as equivalent to 1ch/3ch. History: originally measured BROKEN (synthetic speed ~2.8x inflated vs real, 100% of days landing in the 'high' bucket regardless of request) and root-caused to a sampling-time clip-boundary instability specific to jointly modeling a 2D vector magnitude (U/V) -- see reports/2CH_CLAMP_INVESTIGATION.md. A training-time joint-consistency loss (joint_consistency_weight=1.0) gave a partial fix (2.8x -> 1.4x inflation, plateaued). Dynamic thresholding, layered on top of that same checkpoint, closes most of the remaining gap: marginal_wasserstein 8.0/4.2/3.2 (joint-loss alone) -> 4.5/1.5/4.4 (+ dynamic thresholding), now in the same range as 1ch/3ch. Guidance still does not compose with this checkpoint even with dynamic thresholding on: at guidance_scale=40, ramp_wasserstein explodes (3.5/1.9/0.3) and 'mid' requests resolve to 'low' 83% of the time. Use this preset only without guidance. Separately, eta=1.0 (the stochasticity fix that broadly helped 1ch/3ch) was tested here and is MIXED: low/mid improved (marginal_wasserstein 4.5/1.5 -> 3.0/1.0, regime_accuracy 0.31/0.43 -> 0.35/0.56) but high got clearly WORSE (marginal_wasserstein 4.4 -> 5.9, regime_accuracy 0.32 -> 0.12, confusion shows 'high' requests increasingly resolving to 'mid'). Not adopted as this preset's default pending further investigation -- recommended_generation keeps eta=0.0 for 2ch. See reports/BASELINE_2CH_DYNTHRESH_JOINTLOSS_GS1.md vs reports/BASELINE_2CH_DYNTHRESH_JOINTLOSS_GS40.md (guidance comparison) and reports/BASELINE_2CH_DYNTHRESH_ETA1_GS1.md (eta comparison).

**Possible cause:** See the manifest entry's own account of what was tried.

**Potential future fix:** See the manifest entry's own account of what was tried.
