# Report Card — Transformer Denoiser

**Status: Recommended / primary architecture. Not currently the package's default
loaded checkpoint** (see LIMITATIONS).

---

**MODEL**
Conditional DDPM denoiser using a Transformer encoder over the 24 hourly tokens of a
daily load profile. `src/enerdatasynth/models/transformer.py`.

**DATA**
PJM zone PS, 2016–2023 train / 2024–2025 validation (research-lineage split), 2,785
train / 600 validation days for the Week 7 result. See `docs/data_card.md`.

**CONDITIONING**
Research-validated on the 6-dimensional "Z5" representation (month, day-of-week,
holiday, temperature, spread packed into 6 raw columns) — **not** the current package's
23-dimensional one-hot representation. See `docs/research_status.md` §C.

**ARCHITECTURE**
Per-hour tokenization; learned positional embedding (`N(0, 0.02²)` init); sinusoidal
timestep embedding; conditioning injected via a 2-layer SiLU MLP added to every token;
standard `nn.TransformerEncoder` (`dropout=0.0`, `activation="gelu"`); default
`model_dim=64, num_heads=4, num_layers=2`. **110,401 parameters** (Week 6 comparison
configuration, 6-dim conditioning).

**TRAINING**
Standard DDPM noise-prediction MSE loss, `T=1000` diffusion steps, linear β schedule
(`1e-4 → 0.02`, selected over cosine/quadratic — see `docs/results.md` §5). Trained to
convergence in Week 7 (`scripts/week7.py --task train`). Optimizer/LR/batch size not
independently re-verified in this audit.

**SAMPLING**
Corrected ancestral DDPM reverse sampling (Ho et al., 2020) — recomputes the noise
prediction fresh at every reverse step. This is the fix for a historical sampler bug
(see `docs/research_status.md` §E); a regression test guards against reintroducing it.

**EVALUATION**
7-metric suite (Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp) — see
`docs/research_status.md` §G for exact definitions.

**RESULTS** (Week 7 head-to-head, research conditioning)

| Metric | Transformer | MLP (old) | Best classical |
|---|---|---|---|
| Hourly | **0.0243** | 0.0399 | 1.3535 (ARIMA) |
| FirstDiff | **0.0031** | 0.0185 | 0.1633 (ARIMA) |
| ACF | **0.0325** | 0.0429 | 0.1084 (ARMA) |
| RMSE | **0.0195** | 0.0259 | 0.1808 (ARIMA) |
| Peak | **0.0089** | 0.0115 | 0.3124 (ARIMA) |
| Energy | 0.1405 | **0.0008** | 4.2818 (ARIMA) |
| Ramp | **0.0025** | 0.016 | 0.0057 (ARIMA) |

Beats every classical baseline 7/7; beats MLP 6/7 (loses only on Energy, which
calibration fixes — see below). Full conditional/zone breakdowns in `docs/results.md`.

After scalar calibration (fit on 1,200 training days): Energy error 0.1644 → 0.0563 with
zero cost to shape metrics (`docs/results.md` §6 — note this specific before/after pair
is from the calibration report's own uncalibrated baseline, which differs slightly from
the head-to-head table above due to a different evaluation run; both are genuine,
traceable numbers, not a discrepancy to paper over).

**FAILURE MODES**
Hot-condition days over-represented in the worst-generated decile (68.3% vs. 25.3%
baseline rate — `docs/results.md` §7). Holiday slice (n=17) is the hardest conditional
slice, likely partly a small-sample effect.

**LIMITATIONS**
1. **No checkpoint compatible with the current package's 23-dim conditioning has been
   trained.** All results above use the 6-dim research conditioning.
2. **Not the checkpoint `enerdatasynth.model("pjm_load", ...)` loads today** — the
   package falls back to the best legacy MLP run because no transformer-tagged
   checkpoint exists in `data/checkpoints/`.
3. Zone generalization (16/17 zones, DOM fails) was validated with this architecture but
   not re-exposed through the package (`pjm_load` ships zone `PS` only).
4. Calibration method is validated for this architecture but no calibrated checkpoint is
   bundled.
