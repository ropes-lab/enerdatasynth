# Calibration — removing the level offset

Bias measured on 1200 training days (validation never used for fitting).

- Scalar offset applied: **+0.00920** per hour
- Per-hour offsets: min -0.0027, max +0.0210, std 0.0075

| Variant | Hourly | FirstDiff | ACF | RMSE | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.0296 | 0.0031 | 0.0317 | 0.0210 | 0.0084 | 0.1644 | 0.0025 |
| scalar | 0.0296 | 0.0031 | 0.0317 | 0.0200 | 0.0008 | 0.0563 | 0.0025 |
| per-hour | 0.0333 | 0.0034 | 0.0501 | 0.0249 | 0.0046 | 0.0563 | 0.0028 |

Energy error 0.1644 -> 0.0563 with a single scalar correction.

A constant shift cannot change first-difference, ramp or autocorrelation, so the shape metrics are unaffected. The correction is stored in `calibration.json` next to the checkpoint and should be applied at generation time.