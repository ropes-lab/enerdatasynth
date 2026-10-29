"""Tests for per-zone PJM load models: the folder layout, the `--zone` flag, the
SDK's zone lookup, and that every bundled zone loads and generates."""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from enerdatasynth.utils import config
from enerdatasynth.utils.config import (
    DEFAULT_ZONE, PJM_ZONES, available_zones, checkpoint_search_dirs, find_checkpoint_dir,
    processed_metadata_path, zone_dir,
)

RUN_ID = "transformer_T1000_H64_L2"
REPO = Path(__file__).resolve().parents[2]


def test_zone_list_is_the_17_researched_zones_with_ps_first():
    assert len(PJM_ZONES) == 17 == len(set(PJM_ZONES))
    assert PJM_ZONES[0] == DEFAULT_ZONE == "PS"
    assert "DOM" in PJM_ZONES


def test_default_zone_keeps_the_flat_layout_and_others_nest():
    base = Path("/x/data/checkpoints/pjm_load")
    assert zone_dir(base, "PS") == base
    assert zone_dir(base, "dom") == base / "zones" / "DOM"


def test_helpers_take_a_zone_argument():
    ps = processed_metadata_path("PS")
    dom = processed_metadata_path("DOM")
    assert ps != dom and "zones" in dom.parts and "zones" not in ps.parts
    assert find_checkpoint_dir(RUN_ID, "DOM").name == RUN_ID
    assert len(checkpoint_search_dirs("DOM")) == 2


def _run_config_in_subprocess(zone: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ENERDATASYNTH_ZONE"}
    env["PYTHONPATH"] = str(REPO / "src")
    if zone is not None:
        env["ENERDATASYNTH_ZONE"] = zone
    code = ("from enerdatasynth.utils import config as c; "
            "print(c.ZONE, c.PROCESSED_DIR.relative_to(c.PROJECT_ROOT).as_posix())")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO)


def test_environment_variable_selects_the_working_zone():
    out = _run_config_in_subprocess("dom")
    assert out.stdout.split() == ["DOM", "data/processed/pjm_load/zones/DOM"]
    assert _run_config_in_subprocess(None).stdout.split() == ["PS", "data/processed/pjm_load"]


def test_unknown_zone_in_the_environment_is_rejected():
    out = _run_config_in_subprocess("ZZZ")
    assert out.returncode != 0 and "not a PJM zone" in out.stderr


def test_cli_zone_flag_selects_that_zones_folders(tmp_path):
    """`enerdatasynth train --zone JC` must look in zones/JC/, not at the PS files."""
    env = {k: v for k, v in os.environ.items() if k != "ENERDATASYNTH_ZONE"}
    env.update(PYTHONPATH=str(REPO / "src"), ENERDATASYNTH_PROJECT_ROOT=str(tmp_path))
    out = subprocess.run([sys.executable, "-m", "enerdatasynth", "train", "--zone", "JC"],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode != 0
    assert "pjm_load/zones/JC" in out.stderr.replace(os.sep, "/"), out.stderr[-400:]
    ps = subprocess.run([sys.executable, "-m", "enerdatasynth", "train"],
                        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert "zones/" not in ps.stderr.replace(os.sep, "/"), ps.stderr[-400:]


def test_cli_zone_flag_cannot_switch_zones_inside_a_running_process():
    from enerdatasynth import cli
    other = "DOM" if config.ZONE != "DOM" else "JC"
    with pytest.raises(SystemExit, match="ENERDATASYNTH_ZONE"):
        cli.main(["train", "--zone", other])


def test_sdk_zone_lookup_ignores_case_and_rejects_unknown():
    import enerdatasynth
    assert enerdatasynth.model("pjm_load", zone="ps")._zone == "PS"
    with pytest.raises(ValueError, match="not supported"):
        enerdatasynth.model("pjm_load", zone="XYZ")


@pytest.mark.parametrize("zone", PJM_ZONES)
def test_every_zone_has_its_own_scalers_and_checkpoint(zone):
    assert zone in available_zones(), f"zone {zone} has no scaler file or checkpoint"
    meta = json.loads(processed_metadata_path(zone).read_text())
    assert meta["zone"] == zone
    assert meta["condition_dim"] == 23
    res = json.loads((find_checkpoint_dir(RUN_ID, zone) / "result.json").read_text())
    assert res["architecture"] == "transformer" and res["diffusion_steps"] == 1000
    assert (find_checkpoint_dir(RUN_ID, zone) / "calibration.json").exists()


@pytest.mark.parametrize("zone", PJM_ZONES)
def test_every_zone_generates_plausible_load(zone):
    import enerdatasynth
    load = enerdatasynth.model("pjm_load", zone=zone)
    load.set(month=7, Tmax=86, Tmin=70, weekday=True)
    x = np.asarray(load.sample(n=3, seed=0))
    meta = json.loads(processed_metadata_path(zone).read_text())["load_scaler"]
    assert x.shape == (3, 24) and np.isfinite(x).all()
    lo, hi = np.min(meta["data_min_"]), np.max(meta["data_max_"])
    assert lo - 0.5 * (hi - lo) < x.min() and x.max() < hi + 0.5 * (hi - lo)


def test_zones_give_different_load_levels():
    import enerdatasynth
    means = {}
    for zone in ("PS", "DUQ", "CE"):
        m = enerdatasynth.model("pjm_load", zone=zone)
        m.set(month=7, Tmax=86, Tmin=70, weekday=True)
        means[zone] = float(np.asarray(m.sample(n=4, seed=0)).mean())
    assert len({round(v) for v in means.values()}) == 3, means


def test_manifest_lists_every_zone_and_matches_the_result_files():
    from enerdatasynth.resources.pjm_load.api import list_models
    zones = list_models()["zones"]
    assert list(zones) == PJM_ZONES
    for zone, entry in zones.items():
        res = json.loads((find_checkpoint_dir(RUN_ID, zone) / "result.json").read_text())
        cal = json.loads((find_checkpoint_dir(RUN_ID, zone) / "calibration.json").read_text())
        meta = json.loads(processed_metadata_path(zone).read_text())
        assert entry["run_id"] == RUN_ID and entry["n_params"] == res["n_params"]
        assert entry["best_val_loss"] == pytest.approx(res["best_val_loss"])
        assert entry["calibration_scalar"] == pytest.approx(cal["scalar"])
        assert (entry["train_days"], entry["val_days"]) == (meta["sample_count"]["train"], meta["sample_count"]["val"])
