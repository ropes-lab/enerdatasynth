# `.github/workflows/`

Two workflows.

- **`tests.yml`**: runs on every push and pull request. It installs the package with CPU-only PyTorch, pinned to match `requirements.txt` because the tests load the committed checkpoint `state_dict`s, then runs `pytest tests/ -v`. The matrix is Python 3.10 and 3.12.
- **`publish.yml`**: builds the sdist and wheel and publishes to PyPI through Trusted Publishing (OIDC, so no API token is stored). It runs only when a version tag (`v*.*.*`) is pushed or when started by hand from the Actions tab, never on an ordinary push to `main`. PyPI must already list this workflow as a trusted publisher for the `enerdatasynth` project (owner `ropes-lab`); the header comment in the workflow file gives the exact settings URL.

## Be careful with

The pinned CPU-only torch in `tests.yml` is there because several tests load real checkpoint weights from `data/checkpoints/`, and a different torch version can fail to deserialize them. If you change the pin, re-check checkpoint loading, not just that `pip install` works.

Treat any change to `publish.yml` that could upload to PyPI automatically as high-risk and review it carefully.
