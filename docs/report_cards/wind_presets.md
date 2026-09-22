# Report Card — Wind Channel Presets (1ch / 2ch / 3ch)

**Status: `1ch` and `3ch` are recommended and perform about equally. `2ch` has improved
but hasn't been validated to the same standard (see LIMITATIONS).**

[`transformer.md`](transformer.md) and [`mlp.md`](mlp.md) compare two architectures for
one resource. This card instead compares three channel presets that share one
architecture and training recipe; [`README.md`](README.md) explains why the difference
matters. All numbers come from `src/enerdatasynth/resources/wind/manifest.json`,
measured on 2026-07-21 against each preset's validation split with 100 synthetic
realizations per regime. For the full write-up, see
[`docs/models/wind.md`](../models/wind.md).

---

**MODEL**
Conditional DDPM with a transformer denoiser (`hidden_dim=128, num_heads=4`), a
`T=200` cosine schedule, and DDIM sampling with classifier-free guidance and dynamic
thresholding. All three presets use this architecture. They differ in which wind
features they model and, for `2ch`, in the training recipe.

**CHANNELS**

| Preset | Feature columns | Channel description |
|---|---|---|
| `1ch` | `windspeed_100m` | Windspeed only |
| `2ch` | `windspeed_100m`, `winddirection_100m` | Windspeed decomposed into wind-vector U/V components |
| `3ch` | `windspeed_100m`, `winddirection_100m` | Windspeed plus sin/cos-encoded direction |

**TRAINING RECIPE**

| Preset | Recipe |
|---|---|
| `1ch` | Post-LN transformer, `warmup_epochs=8`, 60 epochs, `cond_dropout_prob=0.3, cond_dropout_scope=wind_regime` |
| `3ch` | Same recipe as `1ch` |
| `2ch` | Same recipe plus `joint_consistency_channels=[0,1], joint_consistency_weight=1.0`, added to fix `2ch`'s original failures (see FAILURE MODES) |

**RECOMMENDED GENERATION SETTINGS** (`eta` and `dynamic_threshold` apply at generation time and work with any checkpoint)

| Preset | sampler | steps | eta | guidance_scale | dynamic_threshold |
|---|---|---|---|---|---|
| `1ch` | ddim | 50 | 1.0 | 40.0 (scope: `wind_regime`) | true (p99) |
| `3ch` | ddim | 50 | 1.0 | 40.0 (scope: `wind_regime`) | true (p99) |
| `2ch` | ddim | 50 | **0.0** | **1.0 (disabled)** | true (p99) |

`2ch` can't use the `eta=1.0` and guidance settings that help `1ch` and `3ch` (see
FAILURE MODES).

**RESULTS BY REGIME** (lower is better except for `regime_accuracy`; with three
regimes, chance is about 0.33)

`1ch`:

| Regime | n (real) | `marginal_wasserstein` | `acf_rmse` | `ramp_wasserstein` | `regime_accuracy` |
|---|---|---|---|---|---|
| Low | 156 | 2.25 | 0.060 | 0.017 | 0.69 |
| Mid | 300 | 2.00 | 0.108 | 0.029 | 0.41 (near chance) |
| High | 147 | 2.11 | 0.101 | 0.066 | 0.55 |

`3ch`:

| Regime | n (real) | `marginal_wasserstein` | `acf_rmse` | `ramp_wasserstein` | `regime_accuracy` |
|---|---|---|---|---|---|
| Low | 156 | 2.67 | 0.074 | 0.074 | 0.41 |
| Mid | 300 | 2.21 | 0.109 | 0.161 | 0.43 (near chance) |
| High | 147 | 1.16 | 0.118 | 0.110 | 0.82 |

`2ch` (dynamic thresholding, no guidance): `marginal_wasserstein` is 4.5 / 1.5 / 4.4
for low / mid / high. Its manifest entry has no `gates_by_regime` table, only a written
note in `known_limitation`, so it isn't shown as a full table here. See FAILURE MODES
for its history.

**THE SHARED FAILURE MODE**

For `1ch` and `3ch`, `mid` regime accuracy never went above about 41–43% under any
generation setting tried, barely better than the one-in-three chance rate. Switching
from DDIM's default `eta=0.0` to `eta=1.0` improved `low` and `high` a lot but didn't
change `mid`. (`eta=0.0` makes sampling fully deterministic, which weakens the
autocorrelation that real wind turbulence has.)

**FAILURE MODES**

- **`1ch` and `3ch`:** `mid` conditioning stays near chance (see above). Otherwise,
  `eta=1.0` with dynamic thresholding clearly beats the DDIM defaults. For example,
  `1ch` `acf_rmse` went from 0.100 / 0.142 / 0.135 to 0.060 / 0.108 / 0.101, and `3ch`
  `marginal_wasserstein` went from 3.31 / 3.49 / 3.52 to 2.67 / 2.21 / 1.16.
- **`2ch`:** the first version produced windspeeds about 2.8× too high, and 100% of
  days landed in the `high` bucket whatever was requested. The cause was the sampler's
  clipping interacting badly with a jointly modeled U/V vector. The joint-consistency
  training loss cut the inflation to 1.4×, and dynamic thresholding fixed most of the
  rest. Guidance still breaks this checkpoint (`ramp_wasserstein` rises to 3.5 / 1.9
  at `guidance_scale=40`), so use `2ch` with guidance off. `eta=1.0` gave mixed results
  for `2ch` (`high` got clearly worse), so it isn't the default.

**LIMITATIONS**

1. **`2ch` hasn't been validated to the same standard as `1ch` and `3ch`.** It has
   improved a lot, but its manifest entry says not to treat it as equivalent.
2. **`mid` regime conditioning is unreliable** for `1ch` and `3ch` under every
   generation setting tried. Nothing so far has fixed it.
3. No preset has a real-vs-generated figure made from this repository, since it has
   no real wind data (`docs/data_card.md` §Wind Data). `docs/models/wind.md` §9 shows
   upstream plots for `1ch` only, with unknown generation settings.
4. All three presets share one vendored architecture and training pipeline. This card
   compares channel representations and generation settings, not separately designed
   models.
