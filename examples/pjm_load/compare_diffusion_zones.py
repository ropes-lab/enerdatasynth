"""
examples/pjm_load/compare_diffusion_zones.py
============================================
Diffusion-vs-diffusion comparison across all 17 PJM zones: the shipped Transformer denoiser
(transformer_T1000_H64_L2, taken from the repository's own checkpoints) against a tuned MLP denoiser trained here, per zone, on the same data, conditioning,
optimizer, early stopping and calibration step. Three MLP configurations per zone:
  A  T=100,  H=256, L=4   (the legacy PS winner)
  B  T=200,  H=128, L=4   (legacy runner-up)
  C  T=1000, H=256, L=4   (same number of diffusion steps as the Transformer)
The best of the three per zone, chosen by validation RMSE, is an oracle choice that favors the
baseline; the Transformer is not tuned. Scores: the package's 7 metrics plus W1, MMD^2 and paired RMSE,
each with and without calibration, averaged over 3 generation seeds. Writes aggregate scores only,
to data/reports/pjm_load/diffusion_zone_comparison.csv.

The 60-epoch MLP of the research phase (RMSE 0.0465 on PS) was under-trained: trained with the same
protocol as the Transformer it reaches about 0.022, so comparisons against it overstate the gap.

Reproduce:
    # raw PJM and NOAA files in R (see data/raw/pjm_load/README.md); W is a scratch folder
    python examples/pjm_load/compare_diffusion_zones.py --stage prep  --raw-dir R --workdir W
    python examples/pjm_load/compare_diffusion_zones.py --stage train --raw-dir R --workdir W
    python examples/pjm_load/compare_diffusion_zones.py --stage eval  --raw-dir R --workdir W

Needs the repository's per-zone Transformer checkpoints (data/checkpoints/pjm_load/); the train stage
takes about 2 minutes per zone and the eval stage about 5 minutes per zone on a CPU.
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance

from enerdatasynth.resources.pjm_load.evaluation.metrics import compute_metrics, METRIC_KEYS
ZONES = ["PS", "AE", "AP", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ", "EKPC", "JC", "ME", "PE", "PL", "PN", "RECO"]
MLP_CONFIGS = {"A": (100, 256, 4), "B": (200, 128, 4), "C": (1000, 256, 4)}
SEEDS = [101, 102, 103]
REPO = Path(__file__).resolve().parents[2]  # the repository root (checkpoints are read from here)


def _w1(real, gen):                     # mean over hours of the 1-D Wasserstein distance
    return float(np.mean([wasserstein_distance(real[:, h], gen[:, h]) for h in range(real.shape[1])]))


def _mmd2(real, gen):                   # biased RBF MMD^2, bandwidth = median pairwise distance of real
    def d2(a, b): return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    bw2 = np.median(d2(real, real)[np.triu_indices(len(real), 1)])
    k = lambda a, b: np.exp(-d2(a, b) / bw2)
    return float(k(real, real).mean() + k(gen, gen).mean() - 2 * k(real, gen).mean())


def score(real, gen):
    m = compute_metrics(real, gen)
    return {**{k: m[k] for k in METRIC_KEYS}, "W1": _w1(real, gen), "MMD2": _mmd2(real, gen),
            "CondRMSE": float(np.sqrt(((real - gen) ** 2).mean()))}


def zdir(base, zone):
    return base if zone == "PS" else base / "zones" / zone


def eds(workdir, zone, *args):
    env = dict(os.environ, ENERDATASYNTH_PROJECT_ROOT=str(workdir), PYTHONPATH=str(REPO / "src"))
    subprocess.run([sys.executable, "-m", "enerdatasynth", *args, "--zone", zone], check=True, env=env)


def run_id(cfg):
    t, h, l = cfg
    return f"mlp_T{t}_H{h}_L{l}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "train", "eval"])
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--zones", nargs="*", default=ZONES)
    ap.add_argument("--out", default="data/reports/pjm_load/diffusion_zone_comparison.csv")
    a = ap.parse_args()
    W = a.workdir
    proc_base, ck_base = W / "data/processed/pjm_load", W / "data/checkpoints/pjm_load"

    if a.stage == "prep":
        raw = W / "data/raw/pjm_load"; raw.mkdir(parents=True, exist_ok=True)
        for f in sorted(a.raw_dir.glob("*.csv")):
            if not (raw / f.name).exists():
                (raw / f.name).symlink_to(f.resolve())
        for z in a.zones:
            if not (zdir(proc_base, z) / "X_daily_profiles.npy").exists():
                eds(W, z, "preprocess")
        return

    if a.stage == "train":
        for z in a.zones:
            for name, cfg in MLP_CONFIGS.items():
                rid = run_id(cfg)
                if (zdir(ck_base, z) / rid / "calibration.json").exists():
                    continue
                t, h, l = cfg
                t0 = time.time()
                eds(W, z, "train", "--architecture", "mlp", "--diffusion-steps", str(t), "--hidden-dim", str(h), "--n-layers", str(l))
                eds(W, z, "calibrate", "--run-id", rid, "--architecture", "mlp", "--diffusion-steps", str(t), "--hidden-dim", str(h), "--n-layers", str(l))
                print(f"[{z}] {rid} done in {time.time() - t0:.0f}s", flush=True)
        return

    # eval
    from enerdatasynth.generators import LoadDiffusionModel
    from enerdatasynth.resources.pjm_load.evaluation.calibration import apply_calibration
    KEYS = ["Hourly", "FirstDiff", "ACF", "RMSE", "Peak", "Energy", "Ramp", "W1", "MMD2", "CondRMSE"]
    rows = []
    for z in a.zones:
        proc = zdir(proc_base, z)
        X = np.load(proc / "X_daily_profiles_val.npy").astype(float)
        Z = np.load(proc / "Z_conditions_val.npy").astype(np.float32)
        models = {"Transformer": (LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2", checkpoint_root=zdir(REPO / "data/checkpoints/pjm_load", z)))}
        for name, cfg in MLP_CONFIGS.items():
            models[f"MLP-{name}"] = LoadDiffusionModel.from_run_id(run_id(cfg), checkpoint_root=zdir(ck_base, z))
        for mname, gen in models.items():
            raw = {s: gen.sample(Z, seed=s, apply_calibration=False) for s in SEEDS}
            cal = gen.calibration
            variants = {"raw": raw}
            if cal is not None:
                variants["cal"] = {s: apply_calibration(x, cal, mode=cal.get("mode_recommended", "scalar")) for s, x in raw.items()}
            for vname, smp in variants.items():
                sc = [score(X, smp[s].astype(float)) for s in SEEDS]
                rows.append({"zone": z, "model": mname, "variant": vname, "n_val": len(X),
                             **{k: round(float(np.mean([r[k] for r in sc])), 4) for k in KEYS}})
        print(z, {(r["model"], r["variant"]): r["RMSE"] for r in rows if r["zone"] == z}, flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", a.out)


if __name__ == "__main__":
    main()
