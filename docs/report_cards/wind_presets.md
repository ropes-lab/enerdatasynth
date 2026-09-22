# Report Card — Wind Channel Presets (1ch / 2ch / 3ch)

**Status: `1ch` and `3ch` recommended, comparable quality. `2ch` improved but not
validated to the same standard — see LIMITATIONS.**

Unlike [`transformer.md`](transformer.md)/[`mlp.md`](mlp.md), which compare two
competing *architectures* for one resource, this card compares three channel
*presets* of the same architecture and training recipe for the wind resource — see
[`README.md`](README.md) for why that distinction matters. All numbers below are
real, from `src/enerdatasynth/resources/wind/manifest.json`, measured 2026-07-21
against each preset's own validation split (100 synthetic realizations per regime).
Full narrative and methodology: [`docs/models/wind.md`](../models/wind.md).

---

**MODEL**
Conditional DDPM, Transformer-style denoiser (`hidden_dim=128, num_heads=4`),
`T=200` cosine schedule, DDIM sampling with classifier-free guidance and dynamic
thresholding. Same architecture family across all three presets; they differ in
which wind features are modeled and (for `2ch`) the training recipe.

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
| `2ch` | Same base recipe **plus** `joint_consistency_channels=[0,1], joint_consistency_weight=1.0` — a non-default addition specific to fixing `2ch`'s originally-broken behavior (see FAILURE MODES) |

**RECOMMENDED GENERATION SETTINGS** (generation-time only, no training dependency for `eta`/`dynamic_threshold`)

| Preset | sampler | steps | eta | guidance_scale | dynamic_threshold |
|---|---|---|---|---|---|
| `1ch` | ddim | 50 | 1.0 | 40.0 (scope: `wind_regime`) | true (p99) |
| `3ch` | ddim | 50 | 1.0 | 40.0 (scope: `wind_regime`) | true (p99) |
| `2ch` | ddim | 50 | **0.0** | **1.0 (disabled)** | true (p99) |

`2ch` cannot use the `eta=1.0` / guidance combination that helps `1ch`/`3ch` — see
FAILURE MODES.

**RESULTS BY REGIME** (lower is better except `regime_accuracy`; chance floor for a
3-way regime bucket is ≈0.33)

`1ch`:

| Regime | n (real) | `marginal_wasserstein` | `acf_rmse` | `ramp_wasserstein` | `regime_accuracy` |
|---|---|---|---|---|---|
| Low | 156 | 2.25 | 0.060 | 0.017 | 0.69 |
| Mid | 300 | 2.00 | 0.108 | 0.029 | **0.41 (near chance)** |
| High | 147 | 2.11 | 0.101 | 0.066 | 0.55 |

`3ch`:

| Regime | n (real) | `marginal_wasserstein` | `acf_rmse` | `ramp_wasserstein` | `regime_accuracy` |
|---|---|---|---|---|---|
| Low | 156 | 2.67 | 0.074 | 0.074 | 0.41 |
| Mid | 300 | 2.21 | 0.109 | 0.161 | **0.43 (near chance)** |
| High | 147 | 1.16 | 0.118 | 0.110 | 0.82 |

`2ch` (with dynamic thresholding, no guidance): `marginal_wasserstein` 4.5 / 1.5 / 4.4
for low/mid/high — in the same range as `1ch`/`3ch` after its joint-consistency fix,
but its manifest entry has no structured `gates_by_regime` table (only the narrative
in `known_limitation`); see FAILURE MODES for the full history and why it's not
presented as an equivalent numeric comparison here.

**THE SHARED FAILURE MODE**

`mid`-regime steering accuracy never exceeded ~41–43% for `1ch`/`3ch`, across every
generation setting tried — barely above the one-third no-conditioning chance floor.
`low` and `high` both improved substantially from `eta=1.0` (DDPM-style stochastic
DDIM, replacing DDIM's default fully-deterministic `eta=0.0`, which measurably
suppresses real wind turbulence's autocorrelation structure); `mid` did not move.
This is a real, unresolved limitation, not a documentation gap.

**FAILURE MODES**

- **`1ch`/`3ch`**: `mid`-regime conditioning stuck near chance (above). Otherwise
  `eta=1.0` + dynamic thresholding is a clear win over the DDIM default: e.g. `1ch`
  `acf_rmse` 0.100/0.142/0.135 → 0.060/0.108/0.101, `3ch` `marginal_wasserstein`
  3.31/3.49/3.52 → 2.67/2.21/1.16.
- **`2ch`**: originally measured broken (~2.8× inflated windspeed, 100% of days
  landing in the `high` bucket regardless of request) — a sampling-time
  clip-boundary instability specific to jointly modeling a 2D U/V vector magnitude.
  The joint-consistency training loss gave a partial fix (2.8× → 1.4× inflation);
  dynamic thresholding closed most of the rest. Guidance still does not compose with
  this checkpoint (`ramp_wasserstein` explodes to 3.5/1.9 at `guidance_scale=40`) —
  use `2ch` only with guidance disabled. `eta=1.0` (which helps `1ch`/`3ch`) is mixed
  for `2ch` (`high`-regime got clearly worse) and is not adopted as its default.

**LIMITATIONS**

1. **`2ch` is not validated to the same standard as `1ch`/`3ch`** — substantially
   improved but its own manifest entry explicitly says not to treat it as equivalent.
2. **`mid`-regime conditioning is unreliable across every preset** and every
   generation setting tried — the one problem nothing fixed this round.
3. No real-vs-generated visual comparison **generated by or for this repository**
   exists for any preset — there is no raw wind data here to independently verify
   against (`docs/data_card.md` §Wind Data). `docs/models/wind.md` §9 now shows the
   upstream `wind_diffusion` project's own envelope plots for `1ch`, borrowed and
   explicitly attributed, as the closest available evidence — not a substitute for
   an independently-generated comparison.
4. All three presets share one architecture/training pipeline (vendored, not built by
   this project) — this card compares channel representations and generation-time
   settings, not independently-designed models.
