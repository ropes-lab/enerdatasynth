"""
examples/pjm_load/plot_all_zones.py
===================================
One generated July weekday (Tmax 86 F, Tmin 70 F) for each of the 17 PJM zone models,
saved as docs/assets/pjm_load_all_zones_generated.png. Each panel shows the mean of 100
generated days and the 5th to 95th percentile band, in the zone's own megawatts, so the
panels have different scales (CE and DOM are around 11 GW, RECO around 0.2 GW).

This is generated output only, from the models shipped with the package: it needs no data
and shows that every zone has a working model. It is not a comparison with real load; for
that see examples/pjm_load/plot_real_vs_generated.py (zone PS) and
data/reports/pjm_load/zone_results.csv (all zones, scored on real validation days).

Reproduce:
    pip install enerdatasynth            # or `pip install -e .` from a clone
    python examples/pjm_load/plot_all_zones.py
    python examples/pjm_load/plot_all_zones.py --out my_figure.png --month 1 --tmax 30 --tmin 18

Takes about a minute on a CPU. Same seed, same figure.
"""

from __future__ import annotations
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enerdatasynth
from enerdatasynth.utils.config import PJM_ZONES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="docs/assets/pjm_load_all_zones_generated.png")
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--tmax", type=float, default=86.0, help="daily maximum temperature, F")
    p.add_argument("--tmin", type=float, default=70.0, help="daily minimum temperature, F")
    p.add_argument("--n", type=int, default=100, help="days per zone")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    hours = np.arange(24)
    fig, axes = plt.subplots(4, 5, figsize=(13, 8.2))
    for ax, zone in zip(axes.flat, PJM_ZONES):
        m = enerdatasynth.model("pjm_load", zone=zone)
        m.set(month=a.month, Tmax=a.tmax, Tmin=a.tmin, weekday=True)
        x = np.asarray(m.sample(n=a.n, seed=a.seed))
        ax.fill_between(hours, np.percentile(x, 5, axis=0), np.percentile(x, 95, axis=0), alpha=0.25, lw=0)
        ax.plot(hours, x.mean(axis=0), lw=1.6)
        ax.set_title(zone, fontsize=10)
        ax.set_xlim(0, 23)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(PJM_ZONES):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("hour of day", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("load (MW)", fontsize=8)
    fig.suptitle(f"Generated load, all 17 PJM zones (month {a.month}, weekday, "
                 f"Tmax {a.tmax:g} F, Tmin {a.tmin:g} F; mean and 5th-95th percentile of {a.n} days)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=130)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
