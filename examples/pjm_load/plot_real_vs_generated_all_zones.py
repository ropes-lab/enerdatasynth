"""
examples/pjm_load/plot_real_vs_generated_all_zones.py
=====================================================
One validation day per PJM zone, real versus generated, for all 17 zones.

The day is chosen by a fixed rule, never by looking at generated output: the
validation day whose Tmax is nearest the zone's median validation Tmax. For each
zone the shipped, calibrated Transformer generates one profile for that day's exact
condition vector (seed 0, dashed line) and 100 more (seed 1, band: 5th to 95th
percentile). Each panel has its own vertical scale. Produces
docs/assets/pjm_load_real_vs_generated_all_zones.png.

Usage:
    python examples/pjm_load/plot_real_vs_generated_all_zones.py

Reproduce:
    1. From a clone of this repository: `pip install -e ".[research]"`.
    2. Get the PJM and NOAA source files yourself (sources: data/README.md) and put
       them in data/raw/pjm_load/. The repository ships no real data.
    3. `enerdatasynth preprocess --zone Z` for every zone Z (`examples/pjm_load/train_all_zones.py`
       does this and more).
    4. `python examples/pjm_load/plot_real_vs_generated_all_zones.py`
    Output: docs/assets/pjm_load_real_vs_generated_all_zones.png. Same seed, same days, same figure.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from enerdatasynth.generators import LoadDiffusionModel
from enerdatasynth.utils.config import DATA_ROOT, PJM_ZONES, PROJECT_ROOT, zone_dir

RUN = "transformer_T1000_H64_L2"
a = argparse.ArgumentParser()
a.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs/assets/pjm_load_real_vs_generated_all_zones.png")
args = a.parse_args()
FS = 6
plt.rcParams.update({"font.size": FS, "font.family": "serif", "axes.titlesize": FS + .5, "axes.labelsize": FS,
                     "xtick.labelsize": FS - .5, "ytick.labelsize": FS - .5, "legend.fontsize": FS})
from matplotlib.ticker import FormatStrFormatter
fig, axes = plt.subplots(4, 5, figsize=(7.16, 4.0), sharex=True, gridspec_kw=dict(left=.075, right=.995, top=.96, bottom=.075, wspace=.42, hspace=.42))
rows = []
for ax, z in zip(axes.ravel(), PJM_ZONES):
    d = zone_dir(DATA_ROOT / "processed" / "pjm_load", z)
    df = pd.read_parquet(d / "df_val.parquet"); X = np.load(d / "X_daily_profiles_val.npy"); Z = np.load(d / "Z_conditions_val.npy")
    tmax = df["Tmax"].to_numpy(); i = int(np.argmin(np.abs(tmax - np.median(tmax))))
    m = LoadDiffusionModel.from_run_id(RUN, zone=z)
    one = np.asarray(m.sample(Z[[i]], seed=0))[0]
    many = np.asarray(m.sample(np.repeat(Z[[i]], 100, axis=0), seed=1))
    lo, hi = np.percentile(many, 5, 0), np.percentile(many, 95, 0)
    ax.fill_between(range(24), lo, hi, color="#4466ee", alpha=.15, lw=0)
    ax.plot(range(24), X[i], color="black", lw=1.2, label="real")
    ax.plot(range(24), one, color="#4466ee", lw=1.0, ls="--", label="generated")
    ax.set_title(f"{z} ({pd.Timestamp(df['date'].iloc[i]).strftime('%m/%d')}, {tmax[i]:.0f}\u00b0F)", pad=2)
    ax.set_xticks([0, 12, 23]); ax.grid(alpha=.25); ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f")); ax.locator_params(axis="y", nbins=4)
    cover = float(np.mean((X[i] >= lo) & (X[i] <= hi)))
    rows.append((z, str(pd.Timestamp(df["date"].iloc[i]).date()), round(float(tmax[i]), 1), round(float(np.sqrt(np.mean((one - X[i]) ** 2))), 4), round(cover, 3)))
    print(z, rows[-1], flush=True)
for ax in axes.ravel()[len(PJM_ZONES):-1]: ax.axis("off")
h, l = axes[0, 0].get_legend_handles_labels()
axes[-1, -1].axis("off"); axes[-1, -1].legend(h + [plt.Rectangle((0, 0), 1, 1, color="#4466ee", alpha=.15)], l + ["5th–95th pct (100 draws)"], loc="center", frameon=False)
for ax in axes[:, 0]: ax.set_ylabel("normalized load")
for ax in axes[-1]: ax.set_xlabel("hour of day (h)")
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=200, bbox_inches="tight")
r = pd.DataFrame(rows)
print(f"Saved: {args.out}")
print("median share of real hours inside the band:", float(np.median(r[4])))
