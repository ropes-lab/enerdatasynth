"""
examples/wind/generate_samples.py
=================================
Generate synthetic wind days from a bundled checkpoint and print the first one
as a 24-hour profile (hourly average wind speed, plus direction for the 2ch and
3ch presets). Wind equivalent of examples/pjm_load/generate_samples.py.

Conditions used to generate: the channel preset (1ch = wind speed only, 2ch and
3ch = speed and direction), `wind_regime` (low, mid or high), `month`, and
`temp_min` / `temp_max` in degrees Celsius. All have defaults.

Reproduce:
    pip install enerdatasynth            # or `pip install -e .` from a clone
    python examples/wind/generate_samples.py
    python examples/wind/generate_samples.py --preset 3ch --regime high --month 1 --n 5

Needs no data: the checkpoints ship inside the package. Output is printed, not
saved. The same --seed gives the same days.
"""

from __future__ import annotations
import argparse

import enerdatasynth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--preset", default="1ch", choices=["1ch", "2ch", "3ch"])
    p.add_argument("--regime", default="mid", choices=["low", "mid", "high"])
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--temp-min", type=float, default=10.0, help="degrees Celsius")
    p.add_argument("--temp-max", type=float, default=22.0, help="degrees Celsius")
    p.add_argument("--n", type=int, default=3, help="number of days to generate")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    wind = enerdatasynth.model("wind", zone=a.preset)
    wind.set(wind_regime=a.regime, month=a.month, temp_min=a.temp_min, temp_max=a.temp_max)
    samples = wind.sample(n=a.n, seed=a.seed)        # (n, 288, 2): m/s, degrees
    print(f"Generated {samples.shape[0]} days, shape {samples.shape} "
          f"(preset {a.preset}, regime {a.regime}, month {a.month})\n")
    enerdatasynth.show(samples)


if __name__ == "__main__":
    main()
