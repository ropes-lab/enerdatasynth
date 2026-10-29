"""
examples/pjm_load/train_all_zones.py
====================================
Train the shipped Transformer (`transformer_T1000_H64_L2`: T=1000, hidden 64,
2 layers, 4 heads, lr 3e-4, batch 64, seed 42) once per PJM zone and fit each
zone's scalar calibration. This is how the 17 bundled zone models were built.

For every zone it runs the same three commands you would run by hand:

    enerdatasynth preprocess --zone <ZONE>
    enerdatasynth train      --zone <ZONE>
    enerdatasynth calibrate  --zone <ZONE> --run-id transformer_T1000_H64_L2

inside a separate working folder (`--workdir`), so the processed arrays, which
are derived from real PJM data, never end up in your repository. Only the files
the package ships (`metadata.json`, `best.pt`, `result.json`, `calibration.json`)
are meant to be copied out of it.

Reproduce:
    pip install -e .                      # from a clone of this repository
    # put the PJM and NOAA files in <raw-dir> (see data/raw/pjm_load/README.md):
    #   hrl_load_metered_2016.csv ... hrl_load_metered_2025.csv, pjm_hourly_temps_wide.csv
    python examples/pjm_load/train_all_zones.py --raw-dir /path/to/raw --workdir /tmp/eds_zones
    python examples/pjm_load/train_all_zones.py --raw-dir /path/to/raw --workdir /tmp/eds_zones --zones DOM JC

Each zone takes a few minutes on a CPU. A zone that already has a trained
checkpoint in the working folder is skipped.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN_ID = "transformer_T1000_H64_L2"
ZONES = ["PS", "AE", "AP", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ",
         "EKPC", "JC", "ME", "PE", "PL", "PN", "RECO"]


def _eds(zone: str, workdir: Path, *args: str) -> None:
    env = dict(os.environ, ENERDATASYNTH_PROJECT_ROOT=str(workdir))
    cmd = [sys.executable, "-m", "enerdatasynth", *args, "--zone", zone]
    subprocess.run(cmd, check=True, env=env)


def _zone_checkpoint(workdir: Path, zone: str) -> Path:
    base = workdir / "data" / "checkpoints" / "pjm_load"
    return (base if zone == "PS" else base / "zones" / zone) / RUN_ID


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-dir", required=True, type=Path, help="Folder with the PJM and NOAA files")
    p.add_argument("--workdir", required=True, type=Path, help="Scratch project folder (created)")
    p.add_argument("--zones", nargs="*", default=ZONES, help="Zones to train (default: all 17)")
    a = p.parse_args()

    raw = a.workdir / "data" / "raw" / "pjm_load"
    raw.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.raw_dir.glob("*.csv")):
        link = raw / f.name
        if not link.exists():
            link.symlink_to(f.resolve())

    summary = {}
    for zone in a.zones:
        ckpt = _zone_checkpoint(a.workdir, zone)
        if (ckpt / "calibration.json").exists():
            print(f"[{zone}] already trained, skipping", flush=True)
        else:
            t0 = time.time()
            print(f"\n===== {zone}: preprocess =====", flush=True)
            _eds(zone, a.workdir, "preprocess")
            print(f"===== {zone}: train =====", flush=True)
            _eds(zone, a.workdir, "train")
            print(f"===== {zone}: calibrate =====", flush=True)
            _eds(zone, a.workdir, "calibrate", "--run-id", RUN_ID)
            print(f"[{zone}] done in {time.time() - t0:.0f}s", flush=True)
        res = json.loads((ckpt / "result.json").read_text())
        summary[zone] = {"best_val_loss": res["best_val_loss"], "best_epoch": res.get("best_epoch")}
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
