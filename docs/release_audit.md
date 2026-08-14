# Release Audit

Produced by a full repository audit and documentation/packaging pass. No model
architecture, conditioning representation, diffusion process, or calibration logic was
changed — this is a documentation, testing, and packaging-hygiene pass only.

**Addendum:** raw PJM CSVs (`data/raw/`) and the cached NOAA
temperature data (`data/cache/`) were removed in a follow-up pass, per an explicit
decision that this repository should not distribute raw source data at all. `data/processed/`
(derived tensors + scaler metadata) and `data/checkpoints/` (trained models) were kept —
the package still generates out-of-the-box. Verified: full test suite (230/230) and the
README quickstart both still pass with zero raw data present. This closes blocker #10
below.

## Repository Status

Clean. No stray hardcoded local paths in the installed package's source code (only in
`data/research_history/scripts/`, which is frozen historical record by design — see that
directory's README). Bundled checkpoint metadata (`data/checkpoints/*/result.json`) had
machine-specific paths; fixed during this audit. No secrets, no TODO/FIXME/HACK markers
found anywhere in the current package code.

## Package Status

Installs cleanly and works standalone: verified via a from-scratch virtualenv install of
the freshly-built wheel, then `import enerdatasynth` + a real `pjm_load` sample + a real
`wind` sample, with **no optional dependencies installed**. A real bug was found and
fixed during this verification: moving `statsmodels`/`matplotlib` to an optional extra
broke `import enerdatasynth` entirely, because `training/train.py` imported
`visualization.plots` eagerly at module level. Fixed by making that import lazy (see
`src/enerdatasynth/training/train.py`). This is exactly the kind of regression a clean-
install test is for — caught, not shipped.

## Documentation Status

Complete for everything requested: `docs/research_status.md`, `docs/results.md`,
`docs/data_card.md`, `docs/model_card.md`, `docs/experiments.md`,
`docs/model_report_card.md` + two report cards, `data/research_history/README.md`, plus
folder READMEs for `docs/`, `src/enerdatasynth/` and its `data/`, `models/`,
`diffusion/`, `training/` (+ `sampling/`), `evaluation/`, `generators/` subpackages,
`examples/`, `tests/`, and `data/configs/`. Main `README.md` updated with a "What This
Is / What This Is Not" section, an explicit research question, a Zone Generalization
section, and links to all new pages. No internal doc links found broken (checked
directly, not assumed).

**Not written:** dedicated READMEs for `visualization/` and `utils/` (small enough to be
covered in `src/enerdatasynth/README.md`'s subpackage table instead, per the stated
"too small to justify its own README" allowance).

## Test Status

**230 / 230 passing.** Two new tests added specifically as a regression guard for the
historical sampler bug (`tests/test_diffusion.py::test_sampler_recomputes_noise_prediction_every_step`,
`::test_sampler_noise_prediction_differs_across_steps`) — both assert real behavior
(call counts, prediction variation across steps), not constants. All 8 CLI subcommands'
`--help` verified working by direct invocation.

**Not done** (flagged, not silently skipped): the full Part 17 test-expansion list
(deterministic-seed test, checkpoint-loading edge cases, condition-shape fuzzing, etc.)
was not exhaustively implemented — the sampler regression test was prioritized as the
one explicitly marked critical. No CLI-level integration test exists yet exercising each
subcommand against real data end-to-end (verified manually instead, this session).

## PyPI Status

**Not published.** The live PyPI release remains `enerdatasynth==0.4.1`, which predates
the wind integration and the conditioning-refactor merge. This repository's version is
`0.5.0`, unpublished. No `twine upload` was run, per this project's standing rule that
this step is always run by the human, never automated by an assistant. The built wheel
(`dist/enerdatasynth-0.5.0-py3-none-any.whl`) passes `twine check` and installs cleanly
in a fresh virtualenv with zero optional dependencies required for basic generation.

## Reproducibility Status

`enerdatasynth preprocess → train → calibrate → evaluate → report` (or `pipeline`)
reproduces a checkpoint from raw data, verified by direct execution during this audit
(a tiny synthetic training smoke test, not the full sweep). **Reproducing the exact
cited Week 7 Transformer numbers is not currently automated or verified** — see
"Scientific Claims Verified" below.

## Scientific Claims Verified (during this audit, by direct inspection/execution)

- The historical sampler bug and its fix are real, documented, and now guarded by a
  regression test (verified by reading `sampler_bugfix.py`, the before/after CSVs, and
  the current `ddpm.py` code).
- No bundled checkpoint has an `"architecture": "transformer"` tag; `_discover_best_checkpoint()`
  therefore resolves to the legacy MLP baseline today (verified by reading every bundled
  `result.json` and the resolution code directly, and by live-testing `enerdatasynth.model("pjm_load", ...)`).
- No bundled checkpoint ships a `calibration.json` (verified: searched
  `data/checkpoints/` directly).
- The Week 3 "Z5" conditioning tensor is actually 6-dimensional, not 5 as its own
  filename/header comment implies (verified by loading the `.npy` file directly).
- 17-zone Phase 2 results and the DOM outlier are real, traceable numbers (verified by
  reading `phase2/summary_report.md` directly).
- The package's train/validation split is chronological (2016–2023/2024–2025), not
  random (verified by reading `week3.py`'s own pipeline description).
- Wind's checkpoints load and produce plausible-shaped output for all three presets
  (verified by direct execution against a clean install).

## Claims That Are Historical Only (not re-verified against current code/data)

- The exact optimizer/learning-rate/batch-size values used in the Week 7 Transformer
  training run (cited in `docs/model_card.md`/`docs/report_cards/transformer.md` as "not
  independently re-verified").
- The exact COVID-period date range excluded during preprocessing.
- Whether the current `preprocess.py`'s per-hour MinMax / z-score / MinMax-spread
  normalization is bit-for-bit identical to `week3.py`'s Step 7 description.
- Wind's generation quality against any ground-truth wind observation — no raw wind
  dataset exists in this repository to validate against (unchanged finding from the
  prior wind-specific audit this session).

## Known Limitations (full list: `docs/research_status.md`, `docs/results.md` §9, `docs/model_card.md`)

1. Conditioning mismatch between validated research results and the package's current
   preprocessing (critical).
2. Package defaults to the legacy MLP checkpoint, not the recommended Transformer.
3. Default sampling is uncalibrated.
4. Only 1 of 17 researched zones is packaged.
5. DOM zone fails badly and is unresolved.
6. Hot-day generation quality is measurably weaker than aggregate metrics suggest.
7. CNN's 118-minute training-time anomaly in the architecture comparison was never
   explained.
8. Calibration is train-fit, not independently re-validated beyond its fitting sample.
9. No wind model is scientifically validated against real wind observations (no raw
   wind data exists in this repository).
10. ~~Raw PJM/NOAA-derived data redistribution rights not verified.~~ **Resolved** —
    the raw CSVs and NOAA cache were removed from this repository entirely (see
    `docs/data_card.md` "Licensing / Redistribution"); the question is now moot rather
    than open.

## Remaining Blockers

| Item | Severity | Notes |
|---|---|---|
| No Transformer checkpoint compatible with current 23-dim conditioning | HIGH | Package doesn't yet demonstrate its own headline result. Fix: `enerdatasynth train` (defaults to transformer already) + `enerdatasynth calibrate`. |
| No calibration bundled for any checkpoint | MEDIUM | Default sampling is uncalibrated; run `enerdatasynth calibrate --run-id <run>` once a checkpoint exists. |
| Only 1 of 17 zones packaged | MEDIUM | `generators/load_conditioning.py`'s `zones=["PS"]` would need extending, plus per-zone checkpoints/scalers bundled or trainable. |
| DOM zone failure undiagnosed | MEDIUM (scientific, not a release blocker) | Worth a dedicated investigation before any paper claims "17/17" or omits DOM. |
| `run_benchmark.py` docstring claims a `benchmarks/` location that doesn't exist | LOW | Cosmetic inconsistency; not moved during this audit (see `examples/README.md`). |
| CNN 118-minute training anomaly unexplained | LOW (documentation completeness) | Not re-investigated; flagged rather than hidden. |
| Full Part 17 test-expansion list not exhaustively implemented | LOW | Sampler regression test (the explicitly critical one) is done; broader coverage is a reasonable follow-up, not a blocker. |

## Recommended Next Steps

1. **Resolve the data-licensing flag** before any public GitHub push — this is the one
   true blocker found in this audit.
2. Train and calibrate a Transformer checkpoint against the current 23-dim conditioning
   (`enerdatasynth train && enerdatasynth calibrate --run-id <new-run-id>`), so the
   package's default output actually matches its documented recommendation.
3. Decide whether to extend `pjm_load`'s packaged `zones` list beyond `PS`, and if so,
   bundle or generate the other 16 zones' checkpoints/scalers.
4. Investigate DOM's failure and the CNN training-time anomaly if either is going into a
   paper — both are currently "observed, not explained."
5. Do not describe this repository as "publication ready" — it is **documentation-
   complete and packaging-verified**, with the specific scientific gaps above still
   open. That distinction is deliberate, not a hedge.
