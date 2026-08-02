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
- **`viz` extra**: seaborn — only needed for `data/research_history/scripts/eda_exploration.py`, not the installed package itself.
- **`dev` extra**: pytest, pytest-cov, ruff, mypy, build, twine.
- **`requirements.txt`**: pinned to the exact versions that produced the data/checkpoints/results currently committed to this repo — use this instead of `pyproject.toml`'s looser ranges when you need bit-for-bit reproduction (see [`docs/installation.md`](installation.md)).

## What's inside the built package (and what isn't)

The wheel contains only `src/enerdatasynth/`'s Python files — no data, no checkpoints, no `data/research_history/`. Verified by inspecting the built wheel's contents directly (`python -m zipfile -l dist/*.whl`) as part of this project's packaging checks, not assumed.

## The `enerdatasynth` CLI

Installed via `[project.scripts] enerdatasynth = "enerdatasynth.cli:main"`. Subcommands: `preprocess`, `train`, `sweep`, `generate`, `calibrate`, `evaluate`, `report`, `pipeline`. Pure argparse dispatch to library functions — no logic lives in `cli.py` itself, so behavior matches calling the underlying functions directly.

## `PROJECT_ROOT` and where data/checkpoints live

`src/enerdatasynth/utils/config.py`'s `PROJECT_ROOT` comes from `$ENERDATASYNTH_PROJECT_ROOT` if set, otherwise the current working directory — **not** from the installed package's location. This is deliberate: a pip-installed package's code lives in site-packages, far from any project's `data/`/`data/checkpoints/`; the SDK operates on whatever project directory you `cd` into or point it at, independent of install mode. See [MIGRATION.md](../MIGRATION.md) for the bug this fixed and how it was caught.

## Publishing (not yet done)

This package has not been published to PyPI. Before doing so: verify the name `enerdatasynth` is actually available (not checked as part of this work), and review `pyproject.toml`'s classifiers/URLs for accuracy. `twine upload dist/*` once ready — not run as part of this project's own verification, since publishing is a one-way action outside the scope of local checks.
