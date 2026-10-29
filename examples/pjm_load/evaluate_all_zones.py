"""
examples/pjm_load/evaluate_all_zones.py
=======================================
Score the shipped Transformer, with and without its calibration, on each zone's
2024-2025 validation days, and write one row per zone to
`data/reports/pjm_load/zone_results.csv`.

For each zone it generates one profile per validation day (using that day's own
conditioning vector), three times with different seeds, and averages the package's
seven metrics (Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp; normalized units,
lower is better). It needs the processed validation arrays, which are built from
real PJM data and are not in this repository: point `--workdir` at the folder
`train_all_zones.py` wrote them to.

Reproduce:
    python examples/pjm_load/train_all_zones.py --raw-dir /path/to/raw --workdir /tmp/eds_zones
    python examples/pjm_load/evaluate_all_zones.py --workdir /tmp/eds_zones
    python examples/pjm_load/evaluate_all_zones.py --workdir /tmp/eds_zones --zones PS DOM

Two columns describe the data rather than the model: `val_over_train_mean` (mean
validation load divided by mean training load, in MW) and `val_days_above_train_max_pct`
(share of validation days with any hour above the training maximum). A zone whose load
grew a lot between 2016-2023 and 2024-2025, like DOM, is partly an extrapolation task for a
model trained on the first period. `--shift-only` adds just those two columns to an
existing result file, without generating anything.

Takes about five minutes per zone on a CPU. The result file holds only aggregate scores,
no data.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from enerdatasynth.generators import LoadDiffusionModel
from enerdatasynth.resources.pjm_load.evaluation.calibration import apply_calibration
from enerdatasynth.resources.pjm_load.evaluation.metrics import compute_metrics, METRIC_KEYS
from enerdatasynth.utils.config import PJM_ZONES

RUN_ID = "transformer_T1000_H64_L2"
SEEDS = [101, 102, 103]


def _zone_dir(base: Path, zone: str) -> Path:
    return base if zone == "PS" else base / "zones" / zone


def level_shift(proc: Path) -> dict:
    """How much the validation days' load level moved away from the training days'."""
    meta = json.loads((proc / "metadata.json").read_text())
    lo = np.array(meta["load_scaler"]["data_min_"])
    span = np.array(meta["load_scaler"]["data_max_"]) - lo
    tr = np.load(proc / "X_daily_profiles.npy").astype(float) * span + lo
    va = np.load(proc / "X_daily_profiles_val.npy").astype(float) * span + lo
    top = np.array(meta["load_scaler"]["data_max_"])
    return {"val_over_train_mean": round(float(va.mean() / tr.mean()), 3),
            "val_days_above_train_max_pct": round(100 * float((va > top).any(axis=1).mean()), 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workdir", required=True, type=Path, help="Folder train_all_zones.py wrote to")
    p.add_argument("--zones", nargs="*", default=PJM_ZONES)
    p.add_argument("--out", default="data/reports/pjm_load/zone_results.csv")
    p.add_argument("--shift-only", action="store_true",
                   help="only add the level-shift columns to an existing --out file")
    a = p.parse_args()

    if a.shift_only:
        with open(a.out, newline="") as f:
            old = list(csv.DictReader(f))
        for row in old:
            row.update(level_shift(_zone_dir(a.workdir / "data" / "processed" / "pjm_load", row["zone"])))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(old[0]))
            w.writeheader()
            w.writerows(old)
        print("updated", a.out)
        return

    rows = []
    for zone in a.zones:
        proc = _zone_dir(a.workdir / "data" / "processed" / "pjm_load", zone)
        ckpt_root = _zone_dir(a.workdir / "data" / "checkpoints" / "pjm_load", zone)
        X = np.load(proc / "X_daily_profiles_val.npy").astype(float)
        Z = np.load(proc / "Z_conditions_val.npy").astype(np.float32)
        meta = json.loads((proc / "metadata.json").read_text())
        res = json.loads((ckpt_root / RUN_ID / "result.json").read_text())
        gen = LoadDiffusionModel.from_run_id(RUN_ID, checkpoint_root=ckpt_root)
        cal = gen.calibration or {}

        row = {
            "zone": zone,
            "train_days": meta["sample_count"]["train"],
            "val_days": meta["sample_count"]["val"],
            "best_val_loss": round(res["best_val_loss"], 5),
            "best_epoch": res.get("best_epoch"),
            "calibration_scalar": round(float(cal.get("scalar", 0.0)), 5),
            **level_shift(proc),
        }
        # Calibration is applied on top of the raw samples, so one generation pass per seed
        # gives both versions (identical to sampling twice with the same seed).
        raw = {s: gen.sample(Z, seed=s, apply_calibration=False) for s in SEEDS}
        mode = cal.get("mode_recommended", "scalar")
        versions = {"raw": raw, "cal": {s: apply_calibration(x, cal, mode=mode) for s, x in raw.items()}}
        for tag, samples in versions.items():
            scores = [compute_metrics(X, samples[s]) for s in SEEDS]
            for k in METRIC_KEYS:
                row[f"{k}_{tag}"] = round(float(np.mean([m[k] for m in scores])), 4)
        rows.append(row)
        print(f"{zone:5s} val_loss={row['best_val_loss']:.5f}  RMSE cal={row['RMSE_cal']:.4f}  "
              f"Energy raw/cal={row['Energy_raw']:.3f}/{row['Energy_cal']:.3f}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
