# `data/`

Everything the package reads from or writes into at runtime, plus the frozen research
record. **No raw or source data is bundled here** — see `docs/data_card.md` for why and
where to get it yourself.

| Folder | What's in it | README |
|---|---|---|
| `raw/` | Where you place raw PJM/NOAA files to run `enerdatasynth preprocess` yourself — empty by default | [`raw/README.md`](raw/README.md) |
| `raw_data/` | A convention-only placeholder location for wind's (currently nonexistent) raw data — not read by any code path automatically | [`raw_data/README.md`](raw_data/README.md) |
| `cache/` | Regenerable NOAA temperature fetch cache — empty by default | [`cache/README.md`](cache/README.md) |
| `processed/`, `processed_backup/` | Bundled, small (~1.3MB), derived tensors + scaler metadata — **required** for `enerdatasynth.model("pjm_load", ...)` to construct | [`processed/README.md`](processed/README.md) |
| `configs/` | YAML mirrors of `RunConfig` hyperparameters | [`configs/README.md`](configs/README.md) |
| `checkpoints/` | The actual trained models this package ships | [`checkpoints/README.md`](checkpoints/README.md) |
| `outputs/` | Where `enerdatasynth generate`/training write their own output | [`outputs/README.md`](outputs/README.md) |
| `reports/` | Current-package benchmark/evaluation reports (distinct from historical ones) | — (see `docs/results.md`, which supersedes this as the canonical read) |
| `research_history/` | Frozen research provenance — not the product | [`research_history/README.md`](research_history/README.md) |

## What a developer should look at first

`checkpoints/` and `processed/` if you're wondering what the package actually ships and
runs on; `research_history/` only if you're auditing *why* the package looks the way it
does, not to run anything from it directly.

## What should not be placed here

Raw or cached third-party source data (PJM CSVs, NOAA data) — see `docs/data_card.md`
"Licensing / Redistribution" for why this was deliberately removed and should not be
re-added without checking that section first.
