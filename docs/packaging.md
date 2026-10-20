# Packaging

How `enerdatasynth` is built, versioned, and verified — useful if you're modifying `pyproject.toml`, cutting a release, or debugging an install issue.

## Build system

Modern `pyproject.toml` (PEP 517/518), `setuptools` backend, `src/`-layout package discovery (`[tool.setuptools.packages.find] where = ["src"]`). No `setup.py` — not needed with a modern `pip`/`setuptools`.

## Versioning

Single source of truth: `src/enerdatasynth/__init__.py`'s `__version__`. `pyproject.toml` reads it dynamically (`[tool.setuptools.dynamic] version = {attr = "enerdatasynth.__version__"}`) — bump one place, not two. Semantic versioning; pre-1.0, so breaking changes still bump the minor version (see [MIGRATION.md](../MIGRATION.md) for the version history and what changed at each bump).

## Three install modes — all verified, not just claimed

```bash
pip install -e .              # editable, for development
pip install .                 # real install from source
pip install dist/*.whl        # real install from a built wheel
```

All three are exercised as part of this project's own verification process (see [MIGRATION.md](../MIGRATION.md)) — not just the editable one. This matters: a real bug (`PROJECT_ROOT` resolving to a nonsense path under a non-editable install) was only caught by actually running `pip install dist/*.whl` and importing the package, not by reading the code or testing only the editable install.

## Building distributions

```bash
pip install ".[dev]"          # includes `build` and `twine`
python -m build                # writes dist/*.tar.gz and dist/*.whl
twine check dist/*             # validates PyPI metadata without uploading
```

## Dependencies

- **Core** (`[project.dependencies]`): numpy, pandas, torch, scikit-learn, statsmodels, holidays, matplotlib, requests, tqdm, pyarrow, pyyaml — everything `src/enerdatasynth/` actually imports.
- **`viz` extra**: seaborn — only needed for `data/research_history/pjm_load/scripts/eda_exploration.py`, not the installed package itself.
- **`dev` extra**: pytest, pytest-cov, ruff, mypy, build, twine.
- **`requirements.txt`**: pinned to the exact versions that produced the data/checkpoints/pjm_load/results currently committed to this repo — use this instead of `pyproject.toml`'s looser ranges when you need bit-for-bit reproduction (see [`docs/installation.md`](installation.md)).

## What's inside the built package (and what isn't)

The wheel contains `src/enerdatasynth/`'s Python files plus `enerdatasynth/_bundled/`: the shipped load Transformer (`best.pt`, `result.json`, `calibration.json`), the three wind presets (`model_checkpoint.pt`, `norm_stats.json`) and `metadata.json`, about 13 MB in total. There is no real data, no legacy MLP runs and no `data/research_history/`. `setup.py` copies the files in at build time and `MANIFEST.in` keeps them in the sdist; both lists must be kept in sync. Check a built wheel with `python -m zipfile -l dist/*.whl | grep -E '_bundled|\.pt|\.npy|parquet'`: you should see exactly those ten files and nothing else.

## The `enerdatasynth` CLI

Installed via `[project.scripts] enerdatasynth = "enerdatasynth.cli:main"`. Subcommands: `preprocess`, `train`, `sweep`, `generate`, `calibrate`, `evaluate`, `report`, `pipeline`, `demo`. Pure argparse dispatch to library functions — no logic lives in `cli.py` itself, so behavior matches calling the underlying functions directly.

## `PROJECT_ROOT` and where data/checkpoints live

`src/enerdatasynth/utils/config.py`'s `PROJECT_ROOT` comes from `$ENERDATASYNTH_PROJECT_ROOT` if set, otherwise the current working directory — **not** from the installed package's location. This is deliberate: a pip-installed package's code lives in site-packages, far from any project's `data/`/`data/checkpoints/pjm_load/`; the SDK operates on whatever project directory you `cd` into or point it at, independent of install mode. The one exception is the bundled checkpoints: they are found relative to the installed package, as a fallback. If your project folder has its own `data/checkpoints/...` or `data/processed/pjm_load/metadata.json`, those are used first. See [MIGRATION.md](../MIGRATION.md) for the bug this fixed and how it was caught.

## Publishing

Published on PyPI as `enerdatasynth`, first at 0.4.1 (load only), then
0.5.0/0.6.0 (adds the `wind` resource and the resources-nested restructure
-- see MODEL_CARD.md), 0.7.0 (pjm_load/wind API parity, wind CLI
support), 0.8.0 (the shipped Transformer default, no real data in
the repository, and the trained checkpoints bundled in the wheel -- see
`MIGRATION.md` for the earlier version history), and **0.8.1** (adds `enerdatasynth demo`).

Releases go out via `.github/workflows/publish.yml`, triggered by pushing
a `vX.Y.Z` tag (matching `__version__`) or manually via the Actions tab.
It builds the sdist/wheel, runs `twine check`, then uploads through PyPI
Trusted Publishing (OIDC) -- no stored API token in this repo or its
secrets. This requires a PyPI project owner to link the workflow once,
under the project's **Publishing** settings on pypi.org (owner:
`ropes-lab`, repo: `enerdatasynth`, workflow: `publish.yml`, environment:
`pypi`) -- **done as of the 0.7.0 release**: the first tag push failed
with exactly the OIDC/trusted-publisher error described below, the link
was added, and re-running the same build published successfully. If this
ever needs redoing (e.g. a renamed workflow file), the fallback is a
manual `twine upload dist/*` from a machine that has a PyPI API token for
this project.

Since 0.8.0 the wheel carries the shipped checkpoints (see above), so a plain `pip install enerdatasynth` can generate for both resources from any folder. Releases 0.4.1 to 0.7.0 shipped code only and needed a clone of this repository. Training your own checkpoints still needs a clone and your own data.

Python 3.9 is not supported (`requires-python >= 3.10`). On 3.9, pip silently installs the last compatible release, 0.4.1, and a new release cannot change that.

## Before making the repository public

The working tree contains no real data, but git history does. The pre-restructure history (`main-before-restructure`) holds hundreds of PJM/NOAA raw CSVs, parquet files and `.npy` arrays, and even `main` still has some in older commits. Making the repository public as-is publishes all of it. Either publish a fresh single-commit snapshot as a new repository (or a Zenodo archive), or rewrite history with `git filter-repo` after the whole team agrees. Do not rewrite shared history unilaterally.

