# Contributing

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project layout

- `src/pjm_diffusion/` — the installable package. Production code
  here never imports from `research_history/`.
- `research_history/` — frozen historical research scripts, kept for
  provenance/reproducibility. Don't edit these; if research there
  needs to become production code, extract and clean it into `src/`
  instead (see MIGRATION.md for examples of how prior extractions were
  done and verified).
- `tests/` — pytest suite; runs in a couple of seconds, no dependency
  on the full dataset being downloaded/processed.
- `configs/` — YAML configs for `RunConfig.from_yaml`.
- `docs/` — this documentation.
- `benchmarks/` — reproducible comparison scripts against classical
  baselines.
- `examples/` — small, runnable demonstrations of the public API.

## Before submitting a change

1. Run the test suite: `pytest tests/ -v`.
2. If you changed default hyperparameters or added a new architecture/
   schedule/calibration mode, cite the evidence (a results file, a
   report) the way `MIGRATION.md` does — this project's standard is
   "every default is backed by a specific number in a specific file,"
   not "this seemed reasonable."
3. If you promote a `research_history/` script into `src/`, verify
   checkpoint/behavior compatibility explicitly where it matters (see
   `MIGRATION.md` §3 for the pattern: loading the actual legacy
   checkpoint with `strict=True` and confirming zero mismatched keys).
4. Rebuild and sanity-check packaging if you touched `pyproject.toml`
   or added new modules: `python -m build && twine check dist/*`.

## Reporting issues

Known gaps and open questions are tracked in `MIGRATION.md` (see
especially §4 "research code that still needs manual review" and §5
"where the final implementation could not be determined with
certainty") rather than restated here — check there first.
