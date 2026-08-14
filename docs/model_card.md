# Model Card — `pjm_load` (Electric Load Diffusion Generator)

This card describes the `pjm_load` resource served by `enerdatasynth.model("pjm_load", ...)`,
written during this repository's documentation audit against the fixed template
requested for that pass. **`../MODEL_CARD.md` (repository root) is the pre-existing,
broader model card covering both `pjm_load` and wind** — read that one for wind's card;
this page goes deeper on `pjm_load` specifically, cross-checked directly against the
current code as of this audit. The two are consistent wherever they overlap; neither
supersedes the other.

## Model Summary

A conditional Denoising Diffusion Probabilistic Model (DDPM) that generates synthetic
24-hour PJM electric load profiles conditioned on calendar and temperature context. Two
architectures exist: a Transformer denoiser (recommended, validated in research) and an
MLP denoiser (original baseline, what the installed package currently defaults to — see
"Known Limitations").

**This is a synthetic-data generator. It is not a forecasting system.** It does not
predict what load *will* be on a specific future day; it samples a plausible daily
profile consistent with given conditions, for research and data-augmentation purposes.

## Intended Use

- Generating synthetic daily load profiles for research on load-shape modeling,
  scenario analysis, or data augmentation where real metered data is limited or
  sensitive.
- Studying conditional generative modeling techniques (diffusion, conditioning
  strategies, calibration) on a real-world time-series domain.

## Non-Intended Use

- **Not a forecasting model.** Does not predict actual future load values.
- **Not an operational dispatch or grid-control system.** No output from this model
  should inform real-time grid operations.
- **Not a causal model.** Conditioning variables (temperature, calendar) shape the
  generated distribution; they are not validated as causal drivers, and the model
  cannot answer "what would load be if temperature changed by X" as a causal claim.
- **Not a guarantee of physical feasibility.** Generated profiles are statistically
  plausible, not verified against grid physics or operational constraints.

## Architecture

Two denoiser architectures share the same DDPM training/sampling machinery
(`src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py`):

**Transformer** (`src/enerdatasynth/resources/pjm_load/models/transformer.py`, recommended/
`DEFAULT_ARCHITECTURE`): per-hour tokenization, learned positional embedding, sinusoidal
timestep embedding, conditioning injected via a 2-layer MLP added to every token,
standard `nn.TransformerEncoder` (default `model_dim=64, num_heads=4, num_layers=2`).
110,401 parameters in the Week 6 controlled comparison (6-dim conditioning).

**MLP** (`src/enerdatasynth/resources/pjm_load/models/denoiser.py`'s `SimpleMLPDenoiser`): the original
baseline, 4 residual blocks, width 256, ~688,408 parameters (Week 6 comparison
configuration) / 224,024 parameters in the specific bundled checkpoint (`T100_H256_L4`,
23-dim conditioning — different config, not directly comparable parameter counts).

## Inputs

A conditioning vector `z` plus a diffusion timestep; see "Conditioning" below for how
`z` is constructed from human-readable arguments via `enerdatasynth.model(...).set(...)`.

## Conditioning

**Current package (23-dimensional, one-hot):** month (12), day-of-week (12+7),
holiday (1), Tmax/Tmin/spread (3, z-scored). See `docs/research_status.md` §C.

**Research-validated configuration ("Z5", 6-dimensional):** the representation the
Week 6/7 Transformer results were actually produced with — 5 conceptual fields packed
into 6 raw numeric columns, not one-hot. **These two representations are not
interchangeable and no checkpoint bridges them** — see "Known Limitations".

## Outputs

A `(N, 24)` array of load values, normalized `[0,1]` per-hour by default (`units="normalized"`)
or rescaled to MW (`units="MW"`) via `enerdatasynth.model(...).sample(..., units=...)`.

## Training Procedure

Standard DDPM training: sample a random timestep per batch item, add noise via the
forward process, train the denoiser to predict that noise with plain MSE loss
(`DDPM.training_loss`, `src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py`). Optimizer, learning rate,
batch size, and epoch count are configurable via `RunConfig`
(`src/enerdatasynth/utils/config.py`) and the `enerdatasynth train` CLI command; specific
default values were not independently re-verified against `training/train.py` in this
audit pass.

## Diffusion Objective

`L = MSE(ε̂, ε)` — standard DDPM noise-prediction loss, no auxiliary terms. Linear beta
schedule (`beta_start=1e-4, beta_end=0.02`), chosen empirically over cosine/quadratic
(see `docs/results.md` §5). See `docs/research_status.md` §E for the full forward/reverse
process definition and the historical sampler-bug correction this model's sampler
implements.

## Sampling

Standard ancestral DDPM reverse sampling (Ho et al., 2020), `T` steps (`DEFAULT_T=1000`),
recomputing the noise prediction fresh at every reverse step — the corrected replacement
for an earlier, buggy sampler that reused predictions across steps (see
`docs/research_status.md` §E, and the regression test added in `tests/pjm_load/test_diffusion.py`
during this audit). Output is clamped to `[-3, 3]` and NaN/Inf-sanitized per step
(`DDPM.sample`).

## Calibration

Optional post-processing scalar bias correction, auto-applied when a `calibration.json`
exists next to the loaded checkpoint. Cuts daily-energy error ~66% at no cost to shape
metrics when present (see `docs/research_status.md` §F). **No bundled checkpoint in this
repository currently ships a `calibration.json`** — default sampling is uncalibrated.

## Evaluation

Seven metrics (`Hourly`, `FirstDiff`, `ACF`, `RMSE`, `Peak`, `Energy`, `Ramp`), all
lower-is-better, defined precisely in `docs/research_status.md` §G.

## Performance

See `docs/results.md` for full tables. Headline (research-configuration, Week 7):
Transformer beats classical baselines 7/7 metrics and the MLP baseline 6/7 metrics at
~6× fewer parameters. **These numbers are not yet reproduced against the package's
current 23-dimensional conditioning** — see "Known Limitations".

## Known Limitations

1. **Conditioning mismatch (critical):** reported Transformer performance used the
   research-history 6-dim "Z5" conditioning, not the package's current 23-dim one-hot
   conditioning. No checkpoint has been trained/evaluated on the current conditioning.
2. **The installed package defaults to the MLP baseline, not the Transformer.** No
   `"architecture": "transformer"` checkpoint is bundled; `enerdatasynth.model("pjm_load", ...)`
   resolves to the best legacy MLP run today (see `docs/research_status.md` §D).
3. **Default sampling is uncalibrated** (no bundled `calibration.json`).
4. **Only zone `PS` is packaged**, despite 17-zone research validation.
5. **DOM zone fails badly** in the 17-zone research result (RMSE ~23× the best zone) and
   this is unresolved.
6. **Hot days are disproportionately represented among worst-generated days** (68.3% of
   the worst decile vs. 25.3% of all days) — see `docs/results.md` §7.
7. **Calibration is train-fit**, not independently re-validated beyond the 1,200-day
   sample it was measured on.
8. **In-sample validation methodology**: comparisons use the same conditioning vectors as
   the real validation set, which tests conditional fidelity, not out-of-distribution
   generalization.
9. An unexplained **118-minute CNN training-time anomaly** appears in the architecture
   comparison and was never investigated (`docs/results.md` §2).

## Failure Modes

- Hot-condition days (high `Tmax`) — see limitation 6.
- Small-sample conditioning slices (e.g. Holiday, n=17 in the validation set) show higher
  variance in reported metrics; treat as noisier estimates, not necessarily worse
  generation.
- DOM zone (limitation 5) — likely a data or scaling issue specific to that zone, not
  re-diagnosed in this audit.

## Zone Generalization

16 of 17 researched PJM zones generalize well (RMSE range 0.0086–0.0436); DOM is a clear,
unresolved outlier (RMSE 0.1998). See `docs/results.md` §8. **Not currently exposed by
the installed package** (`pjm_load` is packaged for zone `PS` only).

## Reproducibility

`enerdatasynth preprocess` → `train` → `calibrate` → `evaluate` → `report`, or
`enerdatasynth pipeline` end-to-end, given raw CSVs under `data/raw/pjm_load/`.
See `docs/quickstart.md`. Reproducing the exact Week 7 research numbers additionally
requires the research-lineage 6-dim conditioning, not the current package preprocessing
— not currently automated.

## Ethical / Operational Considerations

Do not use generated profiles as a substitute for real metered data in any context where
the distinction matters operationally (billing, settlement, grid planning, forecasting).
Generated data should be clearly labeled as synthetic in any downstream use.

## Versioning

Package version `0.5.0` (`enerdatasynth.__version__`, unpublished as of this audit — the
live PyPI release is `0.4.1` and predates the wind integration and conditioning
refactor). See `CHANGELOG.md`.
