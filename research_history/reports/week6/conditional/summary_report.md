# Week 6 — Conditional Subset Evaluation

Diffusion models evaluated on validation slices (temperature, calendar, season).
Lower is better on every metric. All values in normalized [0,1] load units.


## Temp

| Slice | n | RMSE | FirstDiff | ACF | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| Cold (bottom 25%) | 152 | 0.0368 | 0.0148 | 0.0273 | 0.0211 | 0.6445 | 0.0133 |
| Mild (middle 50%) | 296 | 0.0266 | 0.0179 | 0.0606 | 0.0317 | 0.2603 | 0.0162 |
| Hot (top 25%) | 152 | 0.0329 | 0.0216 | 0.0480 | 0.0063 | 0.0316 | 0.0171 |

*Hardest temp slice by RMSE:* **Cold (bottom 25%)** (0.0368).  *Easiest:* **Mild (middle 50%)** (0.0266).

## Calendar

| Slice | n | RMSE | FirstDiff | ACF | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| Weekday | 414 | 0.0266 | 0.0172 | 0.0499 | 0.0116 | 0.0018 | 0.0154 |
| Weekend | 169 | 0.0264 | 0.0202 | 0.0322 | 0.0124 | 0.0719 | 0.0164 |
| Holiday | 17 | 0.0226 | 0.0192 | 0.0676 | 0.0141 | 0.1895 | 0.0156 |

*Hardest calendar slice by RMSE:* **Weekday** (0.0266).  *Easiest:* **Holiday** (0.0226).

## Season

| Slice | n | RMSE | FirstDiff | ACF | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| Winter (DJF) | 150 | 0.0402 | 0.0152 | 0.0261 | 0.0236 | 0.7763 | 0.0134 |
| Spring (MAM) | 182 | 0.0337 | 0.0170 | 0.0447 | 0.0353 | 0.4694 | 0.0158 |
| Summer (JJA) | 178 | 0.0318 | 0.0217 | 0.0392 | 0.0010 | 0.1679 | 0.0172 |
| Fall (SON) | 90 | 0.0299 | 0.0179 | 0.0946 | 0.0452 | 0.4978 | 0.0163 |

*Hardest season slice by RMSE:* **Winter (DJF)** (0.0402).  *Easiest:* **Fall (SON)** (0.0299).