# Wind Failure Analysis

Preset: `1ch`

**Not a new measurement** — flags issues already present in this preset's `manifest.json` entry (real numbers from the upstream project's own validation), against documented thresholds. See [`docs/models/wind.md`](../../../docs/models/wind.md) §8 for the full
per-regime table this is derived from.

## Regime steering weak or absent (mid)

**Observed issue:** regime_accuracy=0.41 for 'mid' (chance floor for 3-way conditioning is 0.33)

**Possible cause:** Conditioning signal for wind_regime is too weak relative to the generation settings, or (for 'mid' specifically) the bucket sits at the boundary between 'low' and 'high' in a way the model consistently confuses — a documented, unresolved issue across every generation-setting configuration tried (see manifest's own known_limitation field for this preset).

**Potential future fix:** Try guidance_scope='wind_regime' with a higher guidance_scale specifically for this regime, or retrain with a stronger cond_dropout_prob so classifier-free guidance has more signal to amplify. Not attempted in this repository.

## Documented known limitation (from manifest.json)

**Observed issue:** Superseded 2026-07-21 by adding eta=1.0 on top of dynamic_threshold -- the eta=0.0 (deterministic DDIM) + dynamic_threshold measurement is preserved in reports/BASELINE_1CH_DYNTHRESH_GS40.md for comparison. Root cause: DDIM's default eta=0.0 makes the reverse process fully deterministic given its starting noise -- no per-step stochastic injection at all. This measurably suppresses the natural persistence/autocorrelation structure real turbulent wind data has; it is NOT fixed by more steps (tried steps=200, no change) or by dynamic_threshold alone. Setting eta=1.0 (matching ancestral DDPM's own stochasticity) fixed most of it: ACF RMSE 0.100/0.142/0.135 -> 0.060/0.108/0.101, and every other gate improved substantially alongside it (marginal_wasserstein 3.62/2.72/3.60 -> 2.25/2.00/2.11, low regime_accuracy jumped 0.58 -> 0.69). mid regime_accuracy stayed flat (0.43 -> 0.41) -- still the one open problem across every configuration tried this session. See presentation_plots/ for the mechanism trace behind this (DDIM steps=50/200/eta=0 vs eta=1 vs ancestral T=200 comparison).

**Possible cause:** See the manifest entry's own account of what was tried.

**Potential future fix:** See the manifest entry's own account of what was tried.
