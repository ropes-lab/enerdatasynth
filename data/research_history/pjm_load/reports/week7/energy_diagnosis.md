# Week 7 follow-up — Energy bias & error analysis

## Part 1: why daily-energy error regressed

- Per-hour level offset (gen − real): **-0.00596**
- Signed daily energy bias: **-0.14305**  (offset × 24 h = -0.14305)
- Reported |energy error|: **0.14305**
- |energy error| after removing the constant offset: **0.00000**
- Constant offset explains **100.0%** of the energy error.
- Per-hour bias spread: min -0.0307, max +0.0229, std 0.0173

Interpretation: the transformer reproduces daily *shape* better than the MLP (first-difference, ramp, peak all improved), but carries a small systematic level offset. Because daily energy sums all 24 hours, that offset compounds ~24× and shows up as a large energy error while shape metrics stay excellent.

## Part 2: best / worst generated days

- Median per-day RMSE: **0.0680**; p90 0.1335; max 0.4593

| Rank | RMSE | Conditions |
|---|---|---|
| best | 0.0169 | Dec Tue  Tmax_z=-0.69 |
| best | 0.0200 | Nov Tue  Tmax_z=-0.20 |
| best | 0.0218 | May Fri  Tmax_z=+0.46 |
| worst | 0.4593 | Jun Thu  Tmax_z=+1.10 |
| worst | 0.3835 | Jul Thu  Tmax_z=+1.38 |
| worst | 0.3381 | Jun Sat  Tmax_z=+1.27 |

Share of worst-10% days by condition (vs share of all days):

- hot: 68.3% vs 25.3%
- cold: 10.0% vs 25.3%
- holiday: 3.3% vs 2.8%
- weekend: 26.7% vs 28.2%