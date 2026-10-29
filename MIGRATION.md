# Migration Report — Promoting Final Research to Production

> **Update (multi-zone):** §4's note about `zone_expansion.py` is now done. The `zone` parameter runs through preprocessing, training, calibration and generation (`--zone` on the CLI, `zone=` in Python), and all 17 PJM zones ship one Transformer each, trained on the 23-dim conditioning (`docs/models/pjm_load.md` §11). Two data issues found on the way: AE, DPL and PL report several load areas per zone, which the research script averaged (wrong for MW) and the original preprocessing would have flagged as daylight-saving days; preprocessing now sums them (`sum_load_areas`).

This documents the audit-and-promote pass that replaced the package's
outdated default architecture (`SimpleMLPDenoiser`) with the one
research actually validated as best (`TransformerDenoiser`), and
brought calibration into the package for the first time. Evidence for
every claim below is a specific file under `research_history/reports/`
or `research_history/scripts/` — nothing here is inferred from
filenames or chronology alone.

**Addendum (later pass, package v0.6.0):** every `src/enerdatasynth/<module>/` path
below (`models/`, `diffusion/`, `training/`, `sampling/`, `evaluation/`, `data/`,
`visualization/`) was subsequently moved to
`src/enerdatasynth/resources/pjm_load/<module>/`, and the vendored wind pipeline moved
from a separate top-level `src/wind_diffusion/` package to
`src/enerdatasynth/resources/wind/` — both nested under a shared `resources/<name>/`
taxonomy so a third resource can be added the same way. Paths in the sections below are
historical (accurate as of the migration described here, not the current tree); see
[`docs/architecture.md`](docs/architecture.md) for the current layout and why it's
structured this way.

**Addendum (later pass, package v0.7.0):** `pjm_load` gained the parity pieces `wind`
already had (`api.py`, `exceptions.py`, `manifest.json`, `config.py`); `wind` gained the
parity pieces `pjm_load` already had (`sampling/generate.py`,
`evaluation/failure_analysis.py`, `evaluation/results_report.py`); the CLI's
`generate`/`evaluate`/`report` gained a `--resource {pjm_load,wind}` flag. See
[`docs/api_reference.md`](docs/api_reference.md) and
[`docs/report_cards/wind_presets.md`](docs/report_cards/wind_presets.md) for the detail.

**Addendum (package v0.8.0):** the repository no longer contains any real data, and the wheel now bundles the shipped checkpoints (load Transformer plus the three wind presets, via `setup.py` and `MANIFEST.in`) so `pip install enerdatasynth` can generate from any folder. `enerdatasynth.utils.config` looks in the project's `data/` first and falls back to the bundled copy. See [`docs/packaging.md`](docs/packaging.md).

**One structural event worth recording here specifically**: this v0.7.0 work was
developed in a separate local clone whose git history shares no common ancestor with
`ropes-lab/enerdatasynth`'s `main` at the time. Rather than force-push (which would have
made `main`'s prior commits unreachable), `main` was updated via a single merge commit
whose tree was forced to exactly match the new work (`git read-tree --reset`) but whose
two parents are old-`main`'s tip and the new work's tip — old history stays fully
reachable and `git log`-able, not discarded. If you see a merge commit on `main` with a
second parent that shares no history with the first, this is why. The pre-merge state of
`main` is also preserved directly as the `main-before-restructure` branch.

## 1. Production modules changed

| Module | Change |
|---|---|
| `src/enerdatasynth/models/__init__.py` | New. `build_denoiser(architecture, ...)` factory — training/sampling/API code now selects architecture by name instead of importing a specific class. |
| `src/enerdatasynth/models/transformer.py` | New. `TransformerDenoiser`, extracted from `research_history/scripts/week7.py`. |
| `src/enerdatasynth/models/denoiser.py` | Docstring only — reframed as the `architecture="mlp"` legacy option; no logic changed. |
| `src/enerdatasynth/evaluation/calibration.py` | New. Scalar bias correction, extracted from `research_history/scripts/week7_calibrate.py`. |
| `src/enerdatasynth/api.py` | New. `LoadDiffusionModel` — the public high-level interface (Phase 4 API). |
| `src/enerdatasynth/utils/config.py` | `RunConfig` gained `architecture` (default `"transformer"`) and `n_heads` fields; `DEFAULT_T` 100→1000, `DEFAULT_HIDDEN_DIM` 128→64, `DEFAULT_N_LAYERS` 4→2, `MAX_EPOCHS` 200→400, `EARLY_STOP_PATIENCE` 10→40 — all now match the actual validated transformer/convergence run (`week7.py --epochs 400 --patience 40`) instead of the earlier MLP-sweep defaults. `run_id` format changed from `T{steps}_H{hidden}_L{layers}` to `{architecture}_T{steps}_H{hidden}_L{layers}` to avoid collisions between architectures. |
| `src/enerdatasynth/training/train.py` | Calls `build_denoiser(cfg.architecture, ...)` instead of hardcoding `SimpleMLPDenoiser`. `result.json` now also records `architecture`, `x_dim`, `cond_dim`, `n_heads` (needed by `LoadDiffusionModel.from_run_id`). |
| `src/enerdatasynth/sampling/generate.py` | Same factory swap; gained optional `architecture`, `n_heads`, and `seed` parameters (the last fixes a previously-flagged reproducibility gap — opt-in, doesn't change default behavior for existing callers). |
| `src/enerdatasynth/cli.py` | Added `--architecture`/`--n-heads` flags to `train`/`generate`; added a new `calibrate` subcommand; `pipeline` now runs calibration as a stage. |
| `src/enerdatasynth/__init__.py` | Exports `LoadDiffusionModel`, `TransformerDenoiser`, `build_denoiser`, `calibrate_checkpoint`. Version bumped 0.1.0 → 0.2.0 (new default behavior, not just additive). |

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

**This is the most important item in this report.** Every validated result cited above (architecture comparison, noise-schedule test, transformer promotion, calibration, 17-zone replication) was produced using the research lineage's **6-dimensional raw "Z5" conditioning vector** (month and day-of-week as plain integers, not one-hot). The package's own preprocessing, `enerdatasynth.resources.pjm_load.data.preprocess.build_conditioning_vector`, produces a **23-dimensional one-hot conditioning vector** instead — a different, deliberate, documented encoding, but one that was never itself run through the architecture/schedule/calibration experiments.

I did not have evidence to determine which conditioning scheme is actually "final":
- The one-hot encoding is a defensible, arguably more principled choice for categorical variables (no false ordinal distance between e.g. December and January), and was clearly a deliberate design decision (documented in `config.py`, described as the package's baseline).
- But it was never validated against the transformer/calibration pipeline the way the raw Z5 encoding was, repeatedly, across multiple research phases.

**What I did, given that ambiguity**: kept the current one-hot preprocessing unchanged (did not silently rewrite it to match Z5), and built the transformer/calibration code generically (accepting whatever `cond_dim` the active preprocessing produces) rather than hardcoding either scheme. **Consequence**: the exact numbers quoted throughout this report and the README (RMSE 0.0215, 6/7 metrics, 66% energy-error reduction) are proven for the Z5 6-dim conditioning, not guaranteed to reproduce identically under the current 23-dim one-hot conditioning. No pretrained checkpoint compatible with the current preprocessing pipeline exists — `checkpoints/` has no transformer runs at all yet. Training one (`enerdatasynth train`) is required, and re-running the evaluation suite against it would confirm or refute whether the one-hot conditioning preserves the transformer's advantage.

I'm flagging this explicitly rather than picking one silently, since changing `preprocess.py`'s conditioning scheme is a much larger-blast-radius decision (it would affect the MLP baseline's existing checkpoints too, and the exact conditioning contract several other modules assume) than swapping a model class.

## Verification performed

- `TransformerDenoiser` loads the real research checkpoint bit-for-bit (`strict=True`, zero missing/unexpected keys, 110,401 params).
- A real (if tiny) end-to-end run through the new default path succeeded: train → calibrate → `LoadDiffusionModel.from_run_id` → `.sample()`.
- Calibration verified to actually change output, and same-seed sampling verified reproducible (previously it was not).
- Package rebuilt (`python -m build`) and both the wheel and sdist pass `twine check` at the new version (0.2.0).
- `enerdatasynth --help` and all subcommands (including the new `calibrate`) verified via `--help`.

---

## Addendum — Production-Quality SDK Pass (v0.3.0)

A second pass, requested explicitly as "production-quality open-source
research software... comparable to mature research projects." Two
decisions here were confirmed with the user rather than assumed:

1. **Package renamed** `powerdiffusion` → `pjm_diffusion` (import name),
   PyPI distribution name `pjm-diffusion`, CLI command `pjm-diffusion`.
   The prior name was fully built and packaging-verified before this
   rename; nothing about the rename itself changed behavior, only
   names (verified: reinstalled editable, rebuilt wheel/sdist, both
   pass `twine check` again under the new name). **This name did not
   last** — see the next addendum below for a second rename to
   `enerdatasynth`, driven by evidence found in the repository itself.
2. **"Unified SDK exposing both load and wind generation"**: built the
   *interface* (`enerdatasynth.generators.BaseGenerator`, an
   `ABC` — genuinely enforced, not just documentation, see
   `tests/test_generators.py`) and made `LoadDiffusionModel` implement
   it. Did **not** fabricate a wind model, wind data, or wind results —
   none exist anywhere in this project. See `docs/extending.md`.

### New in this pass

- `src/enerdatasynth/generators/` — `base.py` (`BaseGenerator`), `load.py`
  (`LoadDiffusionModel`, moved from the old `api.py` and now inheriting
  `BaseGenerator`).
- `configs/transformer.yaml`, `configs/mlp.yaml` + `RunConfig.from_yaml()`/
  `.to_yaml()` — the original Phase 5 ask ("move every hyperparameter
  into YAML"), not previously done. Unknown keys raise rather than
  being silently ignored. Wired into the CLI as `--config`.
- `tests/` — 38 pytest tests (~2s), covering model shapes, DDPM/schedule
  math, metrics correctness, calibration math, YAML config I/O, and a
  real (tiny) end-to-end train → checkpoint → `LoadDiffusionModel` →
  sample round trip. Deliberately synthetic/self-contained — no
  dependency on the real dataset being present.
- `docs/` — installation, quickstart, architecture, evaluation,
  extending (the wind-interface question, answered honestly),
  contributing, and research_notes (**strictly limited to claims
  traceable to a specific file under `research_history/`** — the
  "Complete Reconstruction Report" the user supplied alongside this
  request contains substantial detail — PI email content, a Week-2
  slide deck, specific meeting feedback — that I have no way to verify
  against anything in this repository, and none of it is repeated here
  as fact).
- `benchmarks/run_benchmark.py` — classical-vs-diffusion comparison
  script, built entirely from library calls (`evaluation.baselines`,
  `evaluation.metrics`, `generators.LoadDiffusionModel`), verified to
  run end-to-end against a real checkpoint.

### A regression caught and fixed during this pass

`LoadDiffusionModel.from_run_id` originally defaulted a missing
`architecture` field in `result.json` to `DEFAULT_ARCHITECTURE`
("transformer"). The 7 checkpoints already in `checkpoints/` (from the
original MLP sweep, trained before the `architecture` field existed)
have no such field — loading any of them would have tried to
instantiate a transformer and load MLP weights into it, failing. Fixed
to default missing-field checkpoints to `"mlp"` specifically (the only
architecture that existed when those checkpoints were written), not to
whatever the current default happens to be. Verified against a real
legacy checkpoint (`T100_H128_L2`) and covered by a regression test
(`test_from_run_id_defaults_missing_architecture_to_mlp_not_current_default`).

### Explicitly out of scope for this pass, and why

The user's supplied reconstruction report proposes several additional
artifacts. These were deliberately not built:

- **A LaTeX paper draft** (`paper/`) — writing manuscript prose isn't
  a software-engineering deliverable, and fabricating paper content
  risks misrepresenting what's actually been validated.
- **A hosted documentation site** (Sphinx/MkDocs + CI deploy) — a real
  infrastructure decision (hosting, CI, domain) beyond "write good
  docs," which `docs/*.md` does provide.
- **Resume bullets** — not a repository artifact.
- **Fixing the DOM zone failure, the CNN 118-minute runtime anomaly,
  or reconciling any AE-zone metric discrepancy** — these are research/
  debugging questions requiring actual compute and investigation, not
  something to resolve by editing code structure. They're recorded as
  open items in `docs/results.md` instead of silently "fixed"
  with a fabricated explanation.
- **`assets/` figure regeneration** — would require rerunning full
  training; the existing figures under `reports/` and
  `research_history/reports/` remain the source of truth.

---

## Addendum — Research SDK Redesign (v0.4.0)

A third pass: redesign the public API as a fluent, implementation-hiding
SDK (`model()` / `.set()` / `.sample()`), rewrite the README as a
landing page, add a model card, and round out documentation and test
coverage for a package meant to accompany a paper submission.

### Second package rename: `pjm_diffusion` → `enerdatasynth`

While fixing README.md for this pass, I found it had an **unresolved
git merge conflict since the repository's very first commit** —
literal `<<<<<<<`/`=======`/`>>>>>>>` markers, despite a commit in the
log titled "Resolved merge conflict in README.md". The losing side of
that conflict read: `# enerdatasynth — A python package to generate
energy system data from pre-trained generator models.` That's
suspiciously specific and close to what this project actually is, and
it directly conflicted with the `edatasynth` name I'd started applying
based on this pass's `import edatasynth` example. I stopped, fixed the
conflict markers (kept the actively-maintained HEAD content, since the
other side was a one-line stub), and asked the user which name was
correct before doing any further work that would need it — confirmed:
`enerdatasynth`. Renamed `pjm_diffusion` → `enerdatasynth` throughout
(import name, PyPI distribution name, CLI command — no hyphenation
needed this time, unlike `pjm-diffusion`).

### Public API

`enerdatasynth.model(name, zone=...)` returns a `ResourceModel` that
never exposes tensors, architecture classes, or checkpoint paths.
`.set(**kwargs)` accepts human-readable conditioning (Tmax, Tmin,
month, weekday, holiday); `.sample(n=...)` generates. See
`docs/api_reference.md` for the exact contract and the specific
UX decisions this required (weekday sampling semantics, the
unavoidable "spread" default) — those are real design calls made
without research to back a "correct" answer, documented rather than
hidden.

### A real packaging bug caught by actually running `pip install dist/*.whl`

`PROJECT_ROOT` (and everything derived from it: `RAW_DIR`,
`PROCESSED_DIR`, `CHECKPOINT_DIR`, ...) was computed as `Path(__file__)
.resolve().parents[3]`. That only works for an editable install, where
the source tree stays in place next to `data/`/`checkpoints/`. A real
`pip install .` or `pip install dist/*.whl` copies the code into
site-packages, nowhere near any project directory — `parents[3]` from
`site-packages/enerdatasynth/utils/config.py` silently resolved to
`/opt/homebrew/lib/python3.11/`, and `enerdatasynth.model(...)` failed
looking for data under there. This is exactly the kind of bug that
only surfaces by actually installing the built wheel and running
something, not by reading the code — verified only because this pass
did all three real installs (`pip install -e .`, `pip install .`,
`pip install dist/*.whl`) rather than treating one as a stand-in for
the others.

Fixed: `PROJECT_ROOT` now comes from `$ENERDATASYNTH_PROJECT_ROOT` if
set, else the current working directory — matching how the CLI is
actually meant to be used (`cd` into a project directory, run
`enerdatasynth ...`), independent of where pip put the code. Verified
via a real subprocess launched from a different `cwd`
(`tests/test_config.py::test_project_root_is_not_derived_from_installed_package_location`),
plus manual verification of all three install modes from both the
project directory and elsewhere with/without the override set.

### Verification performed this pass

See the final verification section the assistant reported at the end
of this pass's turn for the concrete commands run (tests, `pip install
.`, `pip install dist/*.whl`, CLI, fluent API round trip).

---

## Addendum — Repo Root Consolidation (v0.4.0, same release)

Requested directly: reduce the repo root to ~4-5 folders so a first
look isn't a dozen artifact directories deep. Purely a physical
reorganization — no behavior changed, verified by the full test suite
and a fluent-API smoke test passing identically before and after.

**Note on reading the rest of this document**: everywhere above this
point, bare paths like `checkpoints/`, `outputs/`, `reports/`,
`configs/`, `research_history/`, `scripts/run_all.py`, and
`benchmarks/run_benchmark.py` describe the repository structure *as it
was at that point in the project's history* — accurate when written,
not retroactively rewritten here. As of this addendum, the actual
locations are:

| Then (described above) | Now |
|---|---|
| `checkpoints/` | `data/checkpoints/pjm_load/` |
| `outputs/` | `data/outputs/pjm_load/` |
| `reports/` | `data/reports/pjm_load/` |
| `configs/` | `data/configs/pjm_load/` |
| `research_history/` | `data/research_history/pjm_load/` |
| `scripts/run_all.py` | `examples/pjm_load/run_all.py` |
| `benchmarks/run_benchmark.py` | `examples/pjm_load/run_benchmark.py` |

Root is now exactly 5 folders: `src/`, `tests/`, `docs/`, `examples/`,
`data/`. `data/` holds everything that isn't source code, tests, docs,
or example scripts: `raw/`, `processed/`, `processed_backup/`,
`checkpoints/`, `outputs/`, `reports/`, `configs/`, and
`research_history/` (nested here deliberately — it's reference/archival
material, not installable software, consistent with why it was already
excluded from the package).

Also folded in: the orphaned top-level `temperature data/` folder (a
single leftover `README.md` from before `pjm_hourly_temps.py` was
promoted into `src/enerdatasynth/data/temperature.py` — content moved
to `docs/temperature_data.md`, folder removed).

`src/enerdatasynth/utils/config.py`'s path constants were updated
accordingly (`CHECKPOINT_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR`,
`CONFIGS_DIR` now nest under a new `DATA_ROOT = PROJECT_ROOT / "data"`;
new `RESEARCH_HISTORY_DIR` constant added) — this is the single source
of truth every other module already read from, so no other production
code needed to change. Every current-state doc (README, MODEL_CARD,
`docs/*.md`) was swept to the new paths; this file's historical
sections were deliberately left describing what was true when written.
