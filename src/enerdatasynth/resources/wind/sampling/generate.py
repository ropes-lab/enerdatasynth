"""
enerdatasynth.resources.wind.sampling.generate
==================================================
Wind's counterpart to `resources.pjm_load.sampling.generate` — added for structural
parity (PJM load had a dedicated `sampling/` module; wind didn't). No new generation
capability lives here: `generate_for_regimes` is a thin batch-convenience wrapper
around the already-existing `resources.wind.api.generate_wind_profile`, useful for
producing one batch per wind regime in a single call (e.g. for the fidelity-gate-style
per-regime comparisons `docs/models/wind.md` §8 tables use).
"""

from __future__ import annotations

from enerdatasynth.resources.wind.api import WindProfile, generate_wind_profile


def generate_for_regimes(
    preset: str = "3ch",
    regimes: tuple[str, ...] = ("low", "mid", "high"),
    n_per_regime: int = 100,
    seed: int | None = None,
    **kwargs,
) -> dict[str, WindProfile]:
    """Generate one `WindProfile` batch per requested wind regime, from the same
    preset/checkpoint. Equivalent to calling `generate_wind_profile` once per regime;
    exists so a caller comparing across regimes (the way this project's own per-regime
    tables in `docs/models/wind.md` §8 do) doesn't have to write that loop themselves.

    `seed`, if given, is offset per regime (`seed + i`) so each regime's batch is
    reproducible individually without every regime generating identical noise.
    """
    profiles: dict[str, WindProfile] = {}
    for i, regime in enumerate(regimes):
        regime_seed = None if seed is None else seed + i
        profiles[regime] = generate_wind_profile(
            channel_method=preset,
            wind_regime=regime,
            n_realizations=n_per_regime,
            seed=regime_seed,
            **kwargs,
        )
    return profiles
