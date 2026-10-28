"""
examples/wind/plot_generated_samples.py
=======================================
Representative generated wind days for the three regimes (low, mid, high), from
the bundled 1ch checkpoint, saved as docs/assets/wind_generated_examples.png.

This is generated output only. The repository has no real wind data, so unlike
examples/pjm_load/plot_real_vs_generated.py there is no real line to compare
against; the real-vs-synthetic plots in docs/models/wind.md section 9 come from
the upstream project. Each panel shows five days from one fixed seed (not a
best-of-N pick). The regime is a request, not a guarantee: docs/models/wind.md
section 8 reports how often it is actually met.

Reproduce:
    pip install enerdatasynth            # or `pip install -e .` from a clone
    python examples/wind/plot_generated_samples.py
    python examples/wind/plot_generated_samples.py --out my_figure.png

Needs no data. Takes about a minute on a CPU. Same seed, same figure.
"""

from __future__ import annotations
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enerdatasynth

REGIMES = ("low", "mid", "high")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="docs/assets/wind_generated_examples.png")
    p.add_argument("--preset", default="1ch", choices=["1ch", "2ch", "3ch"])
    p.add_argument("--n", type=int, default=5, help="days per regime")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    hours = np.arange(288) * 5 / 60
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        m = enerdatasynth.model("wind", zone=a.preset)
        m.set(wind_regime=regime, month=7, temp_min=10, temp_max=22)
        days = np.asarray(m.sample(n=a.n, seed=a.seed))[:, :, 0]
        for d in days:
            ax.plot(hours, d, alpha=0.8, lw=1)
        ax.set(title=f"requested regime: {regime}", xlabel="hour of day", xlim=(0, 24))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("wind speed (m/s)")
    fig.suptitle(f"Generated wind days, {a.preset} preset, July (generated only; no real data in this repository)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
