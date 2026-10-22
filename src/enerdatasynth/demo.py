"""The simplest way in: one call, one printed day.

    import enerdatasynth
    enerdatasynth.load(month=7, Tmax=90, Tmin=74)   # PJM load, one day, MW
    enerdatasynth.wind(month=7, wind_regime="high") # wind, one day, m/s

Each call generates one day with the models shipped inside the package,
prints it hour by hour and returns the numbers as a numpy array. For many
samples, or more control, use `enerdatasynth.model(...)`.
"""

from __future__ import annotations

import numpy as np


def show(samples) -> None:
    """Print the 24-hour profile of the first day in what `.sample()` returned.

        samples = enerdatasynth.model("pjm_load", zone="PS").set(...).sample(n=1)
        enerdatasynth.show(samples)

    Load samples are (n, 24) in MW: prints the 24 hourly values. Wind samples
    are (n, 288, 2) 5-minute values: prints the hourly average wind speed in
    m/s (plus direction in degrees when the preset has it). If you generated
    more than one day, the first one is shown; the rest stay in `samples`.
    """
    a = np.asarray(samples, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    direction = None
    if a.ndim == 2 and a.shape[1] == 24:
        values, label, fmt = a[0], "load (MW)", "{:.0f}"
    elif a.ndim == 3 and a.shape[1] == 288:
        values = a[0, :, 0].reshape(24, 12).mean(axis=1)
        label, fmt = "wind speed (m/s)", "{:.2f}"
        if a.shape[2] > 1 and np.any(a[0, :, 1] != 0):
            rad = np.radians(a[0, :, 1]).reshape(24, 12)
            direction = np.degrees(np.arctan2(np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1))) % 360
    else:
        raise ValueError(
            f"show() expects load samples of shape (n, 24) or wind samples of shape (n, 288, 2); got {a.shape}."
        )
    note = f" (first of {a.shape[0]} generated days)" if a.shape[0] > 1 else ""
    print(f"24-hour profile{note}\n")
    head = f"{'hour':>4}   {label:>16}" + (f"   {'direction (deg)':>15}" if direction is not None else "")
    print(head)
    for h in range(24):
        row = f"{h:4d}   " + fmt.format(values[h]).rjust(16)
        if direction is not None:
            row += f"   {direction[h]:15.0f}"
        print(row)


def load(month: int, Tmax: float, Tmin: float, *, seed: int = 0, **conditions) -> np.ndarray:
    """Generate and print one day of PJM load (zone PS).

    `month` is 1-12; `Tmax` and `Tmin` are the day's temperatures in degrees
    Fahrenheit. Optional conditions: `weekday` (True/False), `holiday`,
    `day_of_week` (0=Mon..6=Sun), `spread`. Prints 24 hourly values in MW and
    returns them as a (24,) array. The same `seed` gives the same day.
    """
    import enerdatasynth

    m = enerdatasynth.model("pjm_load", zone="PS")
    m.set(month=month, Tmax=Tmax, Tmin=Tmin, **conditions)
    day = np.asarray(m.sample(n=1, seed=seed))[0]
    print(f"PJM load, one generated day (zone PS, month {month}, Tmax {Tmax} F, Tmin {Tmin} F)\n")
    print("hour   load (MW)")
    for h, v in enumerate(day):
        print(f"{h:4d}   {v:9.0f}")
    print(f"\nRange {day.min():.0f} to {day.max():.0f} MW.")
    return day


def wind(preset: str = "1ch", *, seed: int = 0, **conditions) -> np.ndarray:
    """Generate and print one day of wind.

    `preset` is "1ch" (wind speed only), "2ch" or "3ch" (speed and direction).
    Optional conditions: `wind_regime` ("low", "mid" or "high"), `month` (1-12),
    `day_of_week` (0=Mon..6=Sun), `temp_min` and `temp_max` (degrees Celsius).
    Prints the hourly average speed in m/s (and direction in degrees for 2ch
    and 3ch) and returns the full (288, 2) array of 5-minute values
    ([:, 0] speed, [:, 1] direction). The same `seed` gives the same day.
    """
    import enerdatasynth

    m = enerdatasynth.model("wind", zone=preset)
    if conditions:
        m.set(**conditions)
    day = np.asarray(m.sample(n=1, seed=seed))[0]                      # (288, 2)
    speed = day[:, 0].reshape(24, 12).mean(axis=1)
    shown = ", ".join(f"{k} {v}" for k, v in conditions.items()) or "default conditions"
    print(f"Wind {preset}, one generated day ({shown})\n")
    if preset == "1ch":
        print("hour   wind speed (m/s)")
        for h in range(24):
            print(f"{h:4d}   {speed[h]:16.2f}")
    else:
        direction = np.degrees(np.angle(np.exp(1j * np.radians(day[:, 1])).reshape(24, 12).mean(axis=1))) % 360
        print("hour   wind speed (m/s)   direction (deg)")
        for h in range(24):
            print(f"{h:4d}   {speed[h]:16.2f}   {direction[h]:15.0f}")
    print(f"\nAverage {day[:, 0].mean():.1f} m/s, max {day[:, 0].max():.1f} m/s.")
    return day


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
