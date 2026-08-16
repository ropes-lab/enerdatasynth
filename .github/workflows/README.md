# `.github/workflows/`

CI/CD automation, two workflows:

- **`tests.yml`** — runs on every push/PR: installs the package (CPU-only PyTorch, pinned to match `requirements.txt` since tests load committed checkpoint `state_dict`s), then `pytest tests/ -v`. Matrix: Python 3.10 and 3.12.
- **`publish.yml`** — builds the sdist/wheel and publishes to PyPI via Trusted Publishing (OIDC, no stored API token). It only triggers on pushing a version tag (`v*.*.*`) or a manual dispatch from the Actions tab, not on every push to `main`. It needs the PyPI project (`enerdatasynth`, owner `ropes-lab`) to have this workflow registered as a trusted publisher first — see the workflow file's own header comment for the exact PyPI settings URL.

## What should not be changed casually

`tests.yml`'s pinned CPU-only torch install exists specifically because several tests load real checkpoint weights committed under `data/checkpoints/` — a mismatched torch version can fail to deserialize them. Bumping the pin means re-verifying checkpoint loading, not just checking that `pip install` succeeds.

Any change to `publish.yml` that could make it upload to PyPI automatically should be treated as high-risk and reviewed carefully — this project's standing convention is that PyPI publishing is triggered by a person, not by CI on its own.
