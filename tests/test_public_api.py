"""
Tests for the top-level fluent SDK (enerdatasynth.model / .set / .sample)
— the interface external users are meant to actually use. Runs against
the real data/processed/ and checkpoints/ already in this repo (skips
if absent), since the whole point of these tests is to prove the
auto-discovery and end-to-end wiring work for real, not mocked.
"""

from __future__ import annotations
import numpy as np
import pytest

from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "metadata.json").exists() or not any(CHECKPOINT_DIR.iterdir()),
    reason="requires data/processed/metadata.json and at least one trained checkpoint",
)


def test_model_unknown_name_raises():
    import enerdatasynth
    with pytest.raises(ValueError, match="Unknown model"):
        enerdatasynth.model("solar_farm")


def test_model_unsupported_zone_raises():
    import enerdatasynth
    with pytest.raises(ValueError, match="not supported"):
        enerdatasynth.model("pjm_load", zone="DOM")


def test_model_defaults_to_first_supported_zone():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    assert load._zone == "PS"


def test_sample_before_set_raises_listing_missing_fields():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    with pytest.raises(ValueError, match="month"):
        load.sample(n=1)


def test_set_is_chainable_and_incremental():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    result = load.set(month=7).set(Tmax=86, Tmin=70)
    assert result is load
    assert load._conditions == {"month": 7, "Tmax": 86, "Tmin": 70}


def test_reset_clears_accumulated_conditions():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=7, Tmax=86, Tmin=70)
    result = load.reset()
    assert result is load
    assert load._conditions == {}
    with pytest.raises(ValueError, match="month"):
        load.sample(n=1)


def test_last_metadata_none_before_first_sample():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    assert load.last_metadata() is None


def test_sample_with_metadata_returns_tuple_and_matches_last_metadata():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=7, Tmax=86, Tmin=70)
    samples, meta = load.sample(n=3, seed=0, with_metadata=True)
    assert samples.shape == (3, 24)
    assert meta["name"] == "pjm_load"
    assert meta["n"] == 3
    assert meta["seed"] == 0
    assert meta["units"] == "MW"
    assert meta["conditions"] == {"month": 7, "Tmax": 86, "Tmin": 70}
    assert load.last_metadata() == meta


def test_sample_without_with_metadata_still_updates_last_metadata():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=7, Tmax=86, Tmin=70)
    samples = load.sample(n=2, seed=0)
    assert isinstance(samples, np.ndarray)  # bare array, not a tuple
    meta = load.last_metadata()
    assert meta is not None
    assert meta["n"] == 2


def test_sample_returns_mw_scale_by_default():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
    samples = load.sample(n=3, seed=0)
    assert samples.shape == (3, 24)
    # Real PJM load is in the hundreds-to-thousands of MW, not [0,1]
    assert samples.max() > 100
    assert np.isfinite(samples).all()


def test_sample_normalized_units_in_zero_one_ish_range():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=1, Tmax=30, Tmin=15)
    samples = load.sample(n=3, seed=0, units="normalized")
    assert samples.max() <= 1.5  # generation can slightly overshoot [0,1]; should not be MW-scale
    assert samples.min() >= -0.5


def test_sample_invalid_units_raises():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=1, Tmax=30, Tmin=15)
    with pytest.raises(ValueError, match="units"):
        load.sample(n=1, units="Celsius")


def test_sample_seed_reproducible():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=7, Tmax=86, Tmin=70)
    a = load.sample(n=2, seed=42)
    b = load.sample(n=2, seed=42)
    assert np.array_equal(a, b)


def test_explicit_day_of_week_overrides_weekday_flag():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=1, Tmax=30, Tmin=15, weekday=False, day_of_week="Wednesday")
    # Should not raise, and should produce valid output despite weekday=False
    # being logically overridden by the explicit day_of_week.
    samples = load.sample(n=1, seed=0)
    assert samples.shape == (1, 24)


def test_invalid_month_raises():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    load.set(month=13, Tmax=80, Tmin=60)
    with pytest.raises(ValueError, match="month"):
        load.sample(n=1)


def test_list_resources_reports_pjm_load():
    import enerdatasynth
    info = enerdatasynth.list_resources()
    assert "pjm_load" in info
    assert info["pjm_load"]["zones"] == ["PS"]
    assert info["pjm_load"]["domain"] == "load"
    # Includes "weekday" alongside the ConditionField-modeled names --
    # condition_fields reflects everything .set() actually accepts
    # (condition_spec fields + extra_set_keys), not just the spec-modeled
    # subset (a gap found while wiring up wind's own extra_set_keys).
    assert set(info["pjm_load"]["condition_fields"]) == {
        "month", "day_of_week", "holiday", "Tmax", "Tmin", "spread", "weekday",
    }
    assert info["pjm_load"]["checkpoint_available"] is True
    assert info["pjm_load"]["best_run_id"]


def test_describe_resource_matches_list_resources_entry():
    import enerdatasynth
    assert enerdatasynth.describe_resource("pjm_load") == enerdatasynth.list_resources()["pjm_load"]


def test_describe_resource_unknown_name_raises():
    import enerdatasynth
    with pytest.raises(ValueError, match="Unknown model"):
        enerdatasynth.describe_resource("solar_farm")


def test_repr_does_not_leak_internals():
    import enerdatasynth
    load = enerdatasynth.model("pjm_load")
    text = repr(load)
    assert "ResourceModel" in text
    assert "pjm_load" in text
    # Must not expose implementation details like tensor dtypes, torch, etc.
    assert "torch" not in text.lower()
    assert "tensor" not in text.lower()


def test_load_temperatures_are_optional():
    import numpy as np
    import enerdatasynth
    m = enerdatasynth.model("pjm_load", zone="PS")
    m.set(month=7)                      # no Tmax / Tmin
    out = m.sample(n=3, seed=0)
    assert out.shape == (3, 24) and np.isfinite(out).all()
    # giving temperatures still changes the result
    hot = enerdatasynth.model("pjm_load").set(month=7, Tmax=95, Tmin=78).sample(n=3, seed=0)
    assert not np.allclose(out, hot)
    # month is still required
    with pytest.raises(ValueError, match="month"):
        enerdatasynth.model("pjm_load").set(Tmax=90).sample(n=1)

