# Wind Failure Analysis

Preset: `3ch`

**Not a new measurement** — flags issues already present in this preset's `manifest.json` entry (real numbers from the upstream project's own validation), against documented thresholds. See [`docs/models/wind.md`](../../../docs/models/wind.md) §8 for the full
per-regime table this is derived from.

## Regime steering weak or absent (low)

**Observed issue:** regime_accuracy=0.41 for 'low' (chance floor for 3-way conditioning is 0.33)

**Possible cause:** Conditioning signal for wind_regime is too weak relative to the generation settings, or (for 'mid' specifically) the bucket sits at the boundary between 'low' and 'high' in a way the model consistently confuses — a documented, unresolved issue across every generation-setting configuration tried (see manifest's own known_limitation field for this preset).

**Potential future fix:** Try guidance_scope='wind_regime' with a higher guidance_scale specifically for this regime, or retrain with a stronger cond_dropout_prob so classifier-free guidance has more signal to amplify. Not attempted in this repository.

## Regime steering weak or absent (mid)

**Observed issue:** regime_accuracy=0.43 for 'mid' (chance floor for 3-way conditioning is 0.33)

**Possible cause:** Conditioning signal for wind_regime is too weak relative to the generation settings, or (for 'mid' specifically) the bucket sits at the boundary between 'low' and 'high' in a way the model consistently confuses — a documented, unresolved issue across every generation-setting configuration tried (see manifest's own known_limitation field for this preset).

**Potential future fix:** Try guidance_scope='wind_regime' with a higher guidance_scale specifically for this regime, or retrain with a stronger cond_dropout_prob so classifier-free guidance has more signal to amplify. Not attempted in this repository.

## Documented known limitation (from manifest.json)

**Observed issue:** Superseded 2026-07-21 by adding eta=1.0 on top of dynamic_threshold -- see 1ch's known_limitation for the shared root cause (DDIM eta=0.0 is fully deterministic, suppressing real data's natural persistence structure; fixed by reintroducing DDPM-style stochasticity). Every gate improved substantially: marginal_wasserstein 3.31/3.49/3.52 -> 2.67/2.21/1.16, ACF RMSE 0.107/0.146/0.146 -> 0.074/0.109/0.118, high regime_accuracy jumped 0.51 -> 0.82. mid regime_accuracy stayed flat (0.44 -> 0.43) -- still the one open problem across every configuration tried this session.

**Possible cause:** See the manifest entry's own account of what was tried.

**Potential future fix:** See the manifest entry's own account of what was tried.
