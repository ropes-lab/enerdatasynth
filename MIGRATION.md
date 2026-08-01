# Migration Report — Promoting Final Research to Production

This documents the audit-and-promote pass that replaced the package's
outdated default architecture (`SimpleMLPDenoiser`) with the one
research actually validated as best (`TransformerDenoiser`), and
brought calibration into the package for the first time. Evidence for
every claim below is a specific file under `research_history/reports/`
or `research_history/scripts/` — nothing here is inferred from
filenames or chronology alone.

## 1. Production modules changed

| Module | Change |
|---|---|
| `src/powerdiffusion/models/__init__.py` | New. `build_denoiser(architecture, ...)` factory — training/sampling/API code now selects architecture by name instead of importing a specific class. |
| `src/powerdiffusion/models/transformer.py` | New. `TransformerDenoiser`, extracted from `research_history/scripts/week7.py`. |
| `src/powerdiffusion/models/denoiser.py` | Docstring only — reframed as the `architecture="mlp"` legacy option; no logic changed. |
| `src/powerdiffusion/evaluation/calibration.py` | New. Scalar bias correction, extracted from `research_history/scripts/week7_calibrate.py`. |
| `src/powerdiffusion/api.py` | New. `LoadDiffusionModel` — the public high-level interface (Phase 4 API). |
| `src/powerdiffusion/utils/config.py` | `RunConfig` gained `architecture` (default `"transformer"`) and `n_heads` fields; `DEFAULT_T` 100→1000, `DEFAULT_HIDDEN_DIM` 128→64, `DEFAULT_N_LAYERS` 4→2, `MAX_EPOCHS` 200→400, `EARLY_STOP_PATIENCE` 10→40 — all now match the actual validated transformer/convergence run (`week7.py --epochs 400 --patience 40`) instead of the earlier MLP-sweep defaults. `run_id` format changed from `T{steps}_H{hidden}_L{layers}` to `{architecture}_T{steps}_H{hidden}_L{layers}` to avoid collisions between architectures. |
| `src/powerdiffusion/training/train.py` | Calls `build_denoiser(cfg.architecture, ...)` instead of hardcoding `SimpleMLPDenoiser`. `result.json` now also records `architecture`, `x_dim`, `cond_dim`, `n_heads` (needed by `LoadDiffusionModel.from_run_id`). |
| `src/powerdiffusion/sampling/generate.py` | Same factory swap; gained optional `architecture`, `n_heads`, and `seed` parameters (the last fixes a previously-flagged reproducibility gap — opt-in, doesn't change default behavior for existing callers). |
| `src/powerdiffusion/cli.py` | Added `--architecture`/`--n-heads` flags to `train`/`generate`; added a new `calibrate` subcommand; `pipeline` now runs calibration as a stage. |
| `src/powerdiffusion/__init__.py` | Exports `LoadDiffusionModel`, `TransformerDenoiser`, `build_denoiser`, `calibrate_checkpoint`. Version bumped 0.1.0 → 0.2.0 (new default behavior, not just additive). |

## 2. Outdated implementations replaced

- **Default denoiser architecture**: `SimpleMLPDenoiser` → `TransformerDenoiser`. Evidence: `research_history/reports/week6/arch/summary_report.md` (transformer has the best generated-profile RMSE of 4 architectures compared under identical conditions) and `research_history/reports/week7/summary_report.md` (transformer beats the MLP on 6/7 metrics at ~6x fewer parameters — 110,401 vs 688,408 — and beats every classical baseline on 7/7). `SimpleMLPDenoiser` is retained, not deleted, as `architecture="mlp"`.
- **Training/convergence hyperparameters**: `max_epochs`/`patience` defaults updated to match the actual convergence run that produced the validated transformer (400/40, not the earlier MLP sweep's 200/10).
- **Diffusion step count default**: 100 → 1000, matching every successful experiment in the validated research lineage (architecture comparison, noise-schedule test, transformer promotion, calibration, 17-zone replication all fixed T=1000).

## 3. Research scripts promoted into production

- `research_history/scripts/week7.py` → `TransformerDenoiser` extracted into `models/transformer.py`. Submodule attribute names (`cond`, `val`, `pos`, `enc`, `out`) were kept identical to the original specifically so trained checkpoints remain loadable — **verified directly**: I loaded the actual research checkpoint (`research_history/outputs/week3_outputs/transformer_baseline_z5.pt`) into the new class with `strict=True` and got zero missing/unexpected keys, 110,401 parameters, exactly matching the validated result. One fix applied during extraction: the original hardcoded the positional-embedding length to a module-level global instead of the constructor's `x_dim` argument — corrected to use `x_dim` properly (no effect on current behavior, since `x_dim` is always 24 here; matters for reusability as a library class).
- `research_history/scripts/week7_calibrate.py` → `evaluation/calibration.py`. The **scalar** mode was chosen as the default deliberately, not by taking "the newer/more elaborate one": the report shows per-hour calibration actually *degrades* ACF (0.0317→0.0501) and RMSE (0.0210→0.0249) relative to no calibration, while scalar matches or improves every metric — and the original script's own JSON output already recorded `"mode_recommended": "scalar"`. This is the one place in this migration where "prefer the more sophisticated-looking option" would have been the wrong call, and the evidence was checked rather than assumed.
- `research_history/reports/week6/noise/summary_report.md` → **no change made**. Linear beta schedule beat both cosine and quadratic on validation loss and generated RMSE. `diffusion/schedules.py` (added in an earlier pass) makes the alternate schedules available but linear correctly remains the default in `diffusion/ddpm.py` — this was verified, not assumed, before leaving it alone.

## 4. Research code that still needs manual review

- **`research_history/scripts/zone_expansion.py`** (multi-zone support): re-implements preprocessing independently per zone rather than reusing `data/preprocess.py`, and confirms the full validated pipeline (transformer + linear schedule + corrected sampler + scalar calibration) generalizes across 17 PJM zones. This is a capability extension, not a bug-fix supersession of anything currently in `src/`, so I did not fold it in automatically — promoting it would mean designing a `zone=` parameter through the preprocessing/training/API surface, a real feature addition someone should scope deliberately.
- **`research_history/scripts/week7_erroranalysis.py`**: diagnostic-only (explains *why* the energy bias exists; the fix is already captured in `evaluation/calibration.py`). No further code to promote, but its finding — the transformer's hardest slice is Holiday conditioning (n=17, sparse) — is worth keeping in mind if conditioning is ever revisited.
- **`n_heads=4`, `model_dim=64`, `num_layers=2`**: fixed constants in every research run that used the transformer; never swept. `training/sweep.py`'s grid (`SWEEP_HIDDEN_DIM`, `SWEEP_N_LAYERS`, `SWEEP_DIFFUSION_STEPS`) was designed for the MLP and has not been re-run against the transformer. Running it now would be a legitimate, currently-unstarted next step.

## 5. Where the final implementation could not be determined with certainty

**This is the most important item in this report.** Every validated result cited above (architecture comparison, noise-schedule test, transformer promotion, calibration, 17-zone replication) was produced using the research lineage's **6-dimensional raw "Z5" conditioning vector** (month and day-of-week as plain integers, not one-hot). The package's own preprocessing, `powerdiffusion.data.preprocess.build_conditioning_vector`, produces a **23-dimensional one-hot conditioning vector** instead — a different, deliberate, documented encoding, but one that was never itself run through the architecture/schedule/calibration experiments.

I did not have evidence to determine which conditioning scheme is actually "final":
- The one-hot encoding is a defensible, arguably more principled choice for categorical variables (no false ordinal distance between e.g. December and January), and was clearly a deliberate design decision (documented in `config.py`, described as the package's baseline).
- But it was never validated against the transformer/calibration pipeline the way the raw Z5 encoding was, repeatedly, across multiple research phases.

**What I did, given that ambiguity**: kept the current one-hot preprocessing unchanged (did not silently rewrite it to match Z5), and built the transformer/calibration code generically (accepting whatever `cond_dim` the active preprocessing produces) rather than hardcoding either scheme. **Consequence**: the exact numbers quoted throughout this report and the README (RMSE 0.0215, 6/7 metrics, 66% energy-error reduction) are proven for the Z5 6-dim conditioning, not guaranteed to reproduce identically under the current 23-dim one-hot conditioning. No pretrained checkpoint compatible with the current preprocessing pipeline exists — `checkpoints/` has no transformer runs at all yet. Training one (`powerdiffusion train`) is required, and re-running the evaluation suite against it would confirm or refute whether the one-hot conditioning preserves the transformer's advantage.

I'm flagging this explicitly rather than picking one silently, since changing `preprocess.py`'s conditioning scheme is a much larger-blast-radius decision (it would affect the MLP baseline's existing checkpoints too, and the exact conditioning contract several other modules assume) than swapping a model class.

## Verification performed

- `TransformerDenoiser` loads the real research checkpoint bit-for-bit (`strict=True`, zero missing/unexpected keys, 110,401 params).
- A real (if tiny) end-to-end run through the new default path succeeded: train → calibrate → `LoadDiffusionModel.from_run_id` → `.sample()`.
- Calibration verified to actually change output, and same-seed sampling verified reproducible (previously it was not).
- Package rebuilt (`python -m build`) and both the wheel and sdist pass `twine check` at the new version (0.2.0).
- `powerdiffusion --help` and all subcommands (including the new `calibrate`) verified via `--help`.
