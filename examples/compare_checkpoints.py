"""
examples/compare_checkpoints.py
================================
Multi-seed comparison of trained PJM load checkpoints on the validation
set, using the package's 7-metric suite. This is the evidence behind
which checkpoint ships as the default (see data/checkpoints/pjm_load/README.md).

Each checkpoint generates one sample per real validation condition vector,
once per seed, and every run's metrics are scored against the real
validation profiles. A checkpoint with a `calibration.json` is scored both
uncalibrated and calibrated. Only generation noise varies across seeds;
each checkpoint was trained once.

Usage:
    python examples/compare_checkpoints.py \\
        --run-ids T100_H256_L4 transformer_T1000_H64_L2 --seeds 101 102 103 104 105

Writes data/reports/pjm_load/checkpoint_comparison.csv.
"""

from __future__ import annotations
import argparse
import csv
import json

import numpy as np
import torch

from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR, REPORTS_DIR, set_seed, ensure_dirs
from enerdatasynth.resources.pjm_load.models import build_denoiser
from enerdatasynth.resources.pjm_load.diffusion.ddpm import DDPM
from enerdatasynth.resources.pjm_load.evaluation.calibration import apply_calibration, load_calibration
from enerdatasynth.resources.pjm_load.evaluation.metrics import compute_metrics, METRIC_KEYS


def _load_checkpoint(run_id: str):
    ckpt_dir = CHECKPOINT_DIR / run_id
    with open(ckpt_dir / "result.json") as f:
        res = json.load(f)
    model = build_denoiser(
        res.get("architecture", "mlp"), x_dim=res.get("x_dim", 24), cond_dim=res.get("cond_dim", 23),
        hidden_dim=res["hidden_dim"], n_layers=res["n_layers"], n_heads=res.get("n_heads") or 4,
    )
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu"))
    return model.eval(), DDPM(T=res["diffusion_steps"], device="cpu"), load_calibration(ckpt_dir / "calibration.json")


def compare(run_ids: list[str], seeds: list[int]) -> list[dict]:
    real = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    Z = torch.tensor(np.load(PROCESSED_DIR / "Z_conditions_val.npy"), dtype=torch.float32)

    scores: dict[str, list[dict]] = {}
    for run_id in run_ids:
        model, ddpm, calibration = _load_checkpoint(run_id)
        for seed in seeds:
            print(f"{run_id}  seed {seed}")
            set_seed(seed)
            with torch.no_grad():
                generated = ddpm.generate_batch(model, Z, real.shape[1]).cpu().numpy()
            scores.setdefault(run_id, []).append(compute_metrics(real, generated))
            if calibration is not None:
                scores.setdefault(f"{run_id} (calibrated)", []).append(
                    compute_metrics(real, apply_calibration(generated, calibration))
                )

    rows = []
    for name, runs in scores.items():
        row = {"model": name, "n_seeds": len(runs)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs]
            row[f"{key}_mean"], row[f"{key}_std"] = round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    args = ap.parse_args()

    rows = compare(args.run_ids, args.seeds)
    ensure_dirs()
    out_path = REPORTS_DIR / "checkpoint_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")
