# FAQ

**Do I need to know anything about diffusion models to use this?**
No. `enerdatasynth.model(...).set(...).sample(...)` never exposes tensors, architecture classes, or checkpoint paths. See the [README Quick Start](../README.md#-quick-start).

**Which model does `enerdatasynth.model("pjm_load")` use?**
The SDK auto-discovers the best *available* checkpoint, preferring transformer runs, and falls back to the best MLP run only if no transformer checkpoint exists. A calibrated Transformer (`transformer_T1000_H64_L2`) ships in `data/checkpoints/pjm_load/`, so that is what loads; the seven older MLP checkpoints stay bundled as the comparison baseline. Train another one with `enerdatasynth train` and it will be picked up automatically. See [`docs/api_reference.md`](api_reference.md)'s `_discover_best_checkpoint` note and [`docs/models/pjm_load.md`](models/pjm_load.md) §5.

**What temperature units does `.set(Tmax=, Tmin=)` expect?**
Degrees Fahrenheit — matching the training data. Not silently converted from Celsius. See [`docs/api_reference.md`](api_reference.md).

**Can I generate load for a zone other than PS?**
Not currently — `PS` (PSE&G, New Jersey) is the only zone the shipped pipeline has data/checkpoints for, even though 17 PJM zones were researched and 16 generalize well. See [`docs/models/pjm_load.md`](models/pjm_load.md#11-multi-zone-results) for the full multi-zone result, including the unresolved `DOM` outlier.

**Can I generate wind data?**
Yes — `enerdatasynth.model("wind", zone="1ch")` (or `"2ch"`/`"3ch"`), with real trained checkpoints bundled in this repository. It wraps a separately-developed diffusion pipeline (`wind_diffusion`, vendored from a Data-For-Good research project) behind the same `.set()`/`.sample()` interface load uses. See [`docs/models/wind.md`](models/wind.md) for what's actually validated (and what isn't — there's no raw wind data in this repository, so nothing here can be independently re-verified against ground truth yet) and [`docs/api_reference.md`](api_reference.md) for the full parameter reference.

**Why does `spread` default to the mean instead of being computed from `Tmax`/`Tmin`?**
`spread` is a property of the load profile being generated, not something derivable from weather alone with any validated method here. See the "Why does `spread` default to the mean?" section in [`docs/api_reference.md`](api_reference.md).

**Is the output real PJM data?**
No — always synthetic. Output should never be presented as real, historical, or utility-reported data. See [MODEL_CARD.md](../MODEL_CARD.md#out-of-scope-uses).

**Does this package provide privacy/anonymization guarantees?**
No formal guarantee was evaluated. See [MODEL_CARD.md](../MODEL_CARD.md#out-of-scope-uses).

**Why MIT and not Apache-2.0?**
A deliberate, low-risk default for open-source research tooling — see [MIGRATION.md](../MIGRATION.md) for when/how this was decided.

**`pip install .` / `pip install dist/*.whl` fails to find my data — why?**
`PROJECT_ROOT` (where `data/`, `data/checkpoints/pjm_load/`, etc. are expected) comes from the current working directory or `$ENERDATASYNTH_PROJECT_ROOT`, not from where pip installed the code. Run `enerdatasynth ...` / import the package from inside your project directory, or set the environment variable explicitly. See [`docs/packaging.md`](packaging.md).

**Why does the README cite different benchmark numbers than `data/research_history/pjm_load/reports/week7/`?**
Because they're two different (real) things: the `data/research_history/pjm_load/` numbers were validated under a different conditioning encoding never ported into this package; the README's numbers are a real run against the currently-shipped pipeline. Both are accurate for what they measure — see the caveat in [`docs/architecture.md`](architecture.md).

**How do I reproduce the exact results in this repo?**
`pip install -r requirements.txt` (pinned to the exact versions used) instead of `pyproject.toml`'s looser ranges — see [`docs/installation.md`](installation.md).

**Where do I report a bug or propose a change?**
See [`docs/developer_guide.md`](developer_guide.md) — known gaps are already tracked in [`docs/release_audit.md`](release_audit.md) and [MIGRATION.md](../MIGRATION.md); check there before opening something new.
