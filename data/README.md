# `data/`

Everything the package reads from or writes into at runtime, plus the frozen research record. No raw or source data is bundled here — see `docs/data_card.md` for why, and where to get it yourself. Every category below is resource-nested, with one `pjm_load/` and one `wind/` subfolder each. See [`docs/architecture.md`](../docs/architecture.md) for why, and what adding a third resource here would look like.

| Folder | What's in it | README |
|---|---|---|
| `raw/{pjm_load,wind}/` | Where you'd place raw source files to run `enerdatasynth preprocess` yourself — both empty by default | [`raw/README.md`](raw/README.md) |
| `cache/{pjm_load,wind}/` | Regenerable NOAA temperature fetch cache, empty by default | [`cache/README.md`](cache/README.md) |
| `processed/{pjm_load,wind}/`, `processed/pjm_load_backup/` | Bundled, small (~1.3MB), derived tensors and scaler metadata. Required for `enerdatasynth.model("pjm_load", ...)` to construct at all | [`processed/README.md`](processed/README.md) |
| `configs/{pjm_load,wind}/` | YAML mirrors of `RunConfig` hyperparameters — load only, since wind's config lives in code | [`configs/README.md`](configs/README.md) |
| `checkpoints/{pjm_load,wind}/` | The actual trained models this package ships | [`checkpoints/README.md`](checkpoints/README.md) |
| `outputs/{pjm_load,wind}/` | Where `enerdatasynth generate`/training write their own output | [`outputs/README.md`](outputs/README.md) |
| `reports/{pjm_load,wind}/` | Current-package benchmark and evaluation reports, distinct from the historical ones | [`reports/README.md`](reports/README.md) — see also `docs/results.md`, the canonical cross-checked read |
| `research_history/{pjm_load,wind}/` | Frozen research provenance, not the product itself | [`research_history/README.md`](research_history/README.md) |

## What a developer should look at first

`checkpoints/` and `processed/` if you're wondering what the package actually ships and runs on. `research_history/` only if you're trying to understand why the package looks the way it does — not to run anything from it directly.

## What should not be placed here

Raw or cached third-party source data — PJM CSVs, NOAA data. See "Licensing / Redistribution" in `docs/data_card.md` for why this was deliberately removed, and check that section before adding any of it back.
