"""One-call demo: print a generated day of PJM load and wind side by side.

    python3 -m enerdatasynth demo
    python3 -c "import enerdatasynth; enerdatasynth.demo()"

Uses the models shipped inside the package, so it works from any folder.
"""

from __future__ import annotations

import numpy as np


def demo(plot: str | None = None, seed: int = 0) -> None:
    """Generate one day of PJM load and one day of wind and print both by hour.

    Load is one generated July weekday (Tmax 90 / Tmin 74 F) in MW. Wind is
    one generated July day from the 1ch preset (wind speed only) in m/s,
    averaged over each hour's twelve 5-minute steps. `plot`, if given, is a
    PNG path to save a picture of both days to.
    """
    import enerdatasynth

    load = enerdatasynth.model("pjm_load", zone="PS")
    load.set(month=7, Tmax=90, Tmin=74, weekday=True)
    load_day = np.asarray(load.sample(n=1, seed=seed))[0]            # (24,) MW

    wind = enerdatasynth.model("wind", zone="1ch")
    wind.set(wind_regime="high", month=7, temp_min=10, temp_max=22)
    wind_day = np.asarray(wind.sample(n=1, seed=seed))[0, :, 0]      # (288,) m/s
    wind_hourly = wind_day.reshape(24, 12).mean(axis=1)

    print(f"enerdatasynth {enerdatasynth.__version__}")
    print("\nOne generated day: PJM load (zone PS, July weekday, 90/74 F) and wind (1ch, July)\n")
    print("hour   load (MW)   wind speed (m/s)")
    for h in range(24):
        print(f"{h:4d}   {load_day[h]:9.0f}   {wind_hourly[h]:16.2f}")
    print(f"\nLoad ranges {load_day.min():.0f} to {load_day.max():.0f} MW; "
          f"wind averages {wind_day.mean():.1f} m/s (max {wind_day.max():.1f}).")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(range(24), load_day)
        ax[0].set(title="PJM load, one generated day", xlabel="hour", ylabel="MW")
        ax[1].plot(np.arange(288) * 5 / 60, wind_day)
        ax[1].set(title="Wind speed (1ch), one generated day", xlabel="hour", ylabel="m/s")
        plt.tight_layout()
        plt.savefig(plot, dpi=120)
        print(f"\nSaved plot: {plot}")
