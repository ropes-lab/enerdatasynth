# Week 7 — Transformer Promoted to Baseline

The transformer denoiser (Week-6 architecture winner) is trained to convergence and
adopted as the primary model. Full re-evaluation below.

## Head-to-head (lower is better)

| Model | Hourly | FirstDiff | ACF | RMSE | Peak | Energy | Ramp |
|---|---|---|---|---|---|---|---|
| AR | 1.36 | 0.423 | 0.1464 | 0.1887 | 0.7557 | 4.4371 | 0.1252 |
| ARMA | 1.3611 | 0.4191 | 0.1084 | 0.1918 | 0.7557 | 4.4636 | 0.1223 |
| ARIMA | 1.3535 | 0.1633 | 0.327 | 0.1808 | 0.3124 | 4.2818 | 0.0057 |
| SARIMA | 1.3614 | 0.6345 | 0.2478 | 0.1905 | 0.6764 | 4.4705 | 0.2927 |
| MLP (old baseline) | 0.0399 | 0.0185 | 0.0429 | 0.0259 | 0.0115 | 0.0008 | 0.016 |
| Transformer (new baseline) | 0.0243 | 0.0031 | 0.0325 | 0.0195 | 0.0089 | 0.1405 | 0.0025 |

Transformer beats the best classical baseline on **7 of 7** metrics.
Transformer beats the old MLP baseline on **6 of 7** metrics, at ~6x fewer parameters (110K vs 688K).

## Consistency across regimes (transformer)

| Group | Slice | n | RMSE | Peak | Ramp |
|---|---|---|---|---|---|
| temp | Cold (bottom 25%) | 152 | 0.026 | 0.0256 | 0.0004 |
| temp | Mild (mid 50%) | 296 | 0.0209 | 0.0053 | 0.0023 |
| temp | Hot (top 25%) | 152 | 0.0205 | 0.001 | 0.0049 |
| calendar | Weekday | 414 | 0.023 | 0.0184 | 0.0021 |
| calendar | Weekend | 169 | 0.0195 | 0.01 | 0.0032 |
| calendar | Holiday | 17 | 0.0306 | 0.0359 | 0.0044 |
| season | Winter | 150 | 0.0283 | 0.0297 | 0.0002 |
| season | Spring | 182 | 0.0209 | 0.0017 | 0.0019 |
| season | Summer | 178 | 0.0266 | 0.0124 | 0.005 |
| season | Fall | 90 | 0.0193 | 0.0115 | 0.0023 |

Hardest slice for the new baseline: **calendar/Holiday** (RMSE 0.0306).