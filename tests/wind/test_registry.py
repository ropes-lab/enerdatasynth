"""Tests for enerdatasynth.resources.wind.registry: manifest loading, weights resolution
precedence (explicit args > env dir > cached download > pending error), and
the sha256-verified download/cache path. Zero real network calls -- the
"download" tests serve fixture files from a local tmp_path via plain
filesystem paths, and offline-mode tests assert (via a monkeypatched call
counter on `_fetch_to_path`) that no download is even attempted.
"""

from __future__ import annotations

import hashlib

import pytest

from enerdatasynth.resources.wind import registry
from enerdatasynth.resources.wind.exceptions import CheckpointNotFoundError, WeightsDownloadError

PRESETS = ("1ch", "2ch", "3ch")


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Manifest content
# ---------------------------------------------------------------------------


def test_manifest_has_exactly_the_three_presets():
    manifest = registry.load_manifest()
    assert set(manifest.keys()) == set(PRESETS)


def test_all_presets_currently_pending():
    manifest = registry.load_manifest()
    for preset in PRESETS:
        assert manifest[preset]["status"] == "pending"


def test_list_models_reflects_manifest():
    models = registry.list_models()
    assert set(models.keys()) == set(PRESETS)
    for preset in PRESETS:
        assert models[preset]["status"] == "pending"


def test_describe_model_includes_cache_state_fields():
    info = registry.describe_model("2ch")
    assert info["status"] == "pending"
    assert "cached_locally" in info
    assert "cache_path" in info
    assert "env_override_dir" in info
    assert "env_override_available" in info
    assert info["cached_locally"] is False


# ---------------------------------------------------------------------------
# Pending manifest -> CheckpointNotFoundError
# ---------------------------------------------------------------------------


def test_resolve_weights_pending_manifest_raises_with_actionable_message(monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)

    with pytest.raises(CheckpointNotFoundError) as excinfo:
        registry.resolve_weights("2ch")

    msg = str(excinfo.value)
    assert "2ch" in msg
    assert "wind-diffusion-train" in msg
    assert "--preset 2ch" in msg


# ---------------------------------------------------------------------------
# Explicit checkpoint=/norm_stats= args
# ---------------------------------------------------------------------------


def test_resolve_weights_explicit_args_bypass_everything(tmp_path, monkeypatch):
    # Even with WIND_DIFFUSION_CHECKPOINTS pointing elsewhere, explicit args win.
    monkeypatch.setenv("WIND_DIFFUSION_CHECKPOINTS", str(tmp_path / "unrelated_dir"))

    ckpt = tmp_path / "my_model.pt"
    norm = tmp_path / "my_norm_stats.json"
    ckpt.write_bytes(b"fake checkpoint bytes")
    norm.write_text("{}", encoding="utf-8")

    resolved_ckpt, resolved_norm = registry.resolve_weights("2ch", checkpoint=ckpt, norm_stats=norm)

    assert resolved_ckpt == ckpt
    assert resolved_norm == norm


def test_resolve_weights_explicit_args_missing_paths_raise_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)

    missing_ckpt = tmp_path / "does_not_exist.pt"
    missing_norm = tmp_path / "also_missing.json"

    with pytest.raises(CheckpointNotFoundError) as excinfo:
        registry.resolve_weights("2ch", checkpoint=missing_ckpt, norm_stats=missing_norm)

    msg = str(excinfo.value)
    assert "does_not_exist.pt" in msg


def test_resolve_weights_only_one_explicit_arg_raises(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(CheckpointNotFoundError):
        registry.resolve_weights("2ch", checkpoint=ckpt, norm_stats=None)


# ---------------------------------------------------------------------------
# WIND_DIFFUSION_CHECKPOINTS directory override
# ---------------------------------------------------------------------------


def test_resolve_weights_env_checkpoints_dir(tmp_path, monkeypatch):
    preset_dir = tmp_path / "2ch"
    preset_dir.mkdir()
    ckpt = preset_dir / "model_checkpoint.pt"
    norm = preset_dir / "norm_stats.json"
    ckpt.write_bytes(b"fake checkpoint")
    norm.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("WIND_DIFFUSION_CHECKPOINTS", str(tmp_path))

    resolved_ckpt, resolved_norm = registry.resolve_weights("2ch")

    assert resolved_ckpt == ckpt
    assert resolved_norm == norm


def test_resolve_weights_env_checkpoints_dir_missing_files_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WIND_DIFFUSION_CHECKPOINTS", str(tmp_path))  # empty dir, no <preset>/ subfolder

    with pytest.raises(CheckpointNotFoundError) as excinfo:
        registry.resolve_weights("2ch")

    msg = str(excinfo.value)
    assert "WIND_DIFFUSION_CHECKPOINTS" in msg
    assert "2ch" in msg


# ---------------------------------------------------------------------------
# WIND_DIFFUSION_OFFLINE
# ---------------------------------------------------------------------------


def test_offline_flag_raises_and_makes_no_download_attempt(monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.setenv("WIND_DIFFUSION_OFFLINE", "1")

    call_count = {"n": 0}

    def counting_fetch(url, dest):
        call_count["n"] += 1
        raise AssertionError("should never be called when offline")

    monkeypatch.setattr(registry, "_fetch_to_path", counting_fetch)

    with pytest.raises(CheckpointNotFoundError) as excinfo:
        registry.resolve_weights("2ch")

    msg = str(excinfo.value)
    assert "offline" in msg.lower() or "OFFLINE" in msg
    assert call_count["n"] == 0


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_offline_flag_truthy_values(monkeypatch, value):
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.setenv("WIND_DIFFUSION_OFFLINE", value)
    assert registry._is_offline() is True


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_offline_flag_falsy_values(monkeypatch, value):
    monkeypatch.setenv("WIND_DIFFUSION_OFFLINE", value)
    assert registry._is_offline() is False


# ---------------------------------------------------------------------------
# Download + cache + sha256 verification (fake manifest entry, local files)
# ---------------------------------------------------------------------------


def _fake_available_entry(checkpoint_src, norm_src, checkpoint_sha, norm_sha):
    return {
        "status": "available",
        "version": "v1",
        "checkpoint_url": str(checkpoint_src),
        "checkpoint_sha256": checkpoint_sha,
        "norm_stats_url": str(norm_src),
        "norm_stats_sha256": norm_sha,
    }


def test_download_and_cache_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)
    cache_root = tmp_path / "cache_root"
    monkeypatch.setenv("WIND_DIFFUSION_CACHE_DIR", str(cache_root))

    # "Remote" source files, served as plain local paths.
    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    ckpt_bytes = b"tiny fake checkpoint payload"
    norm_bytes = b'{"fake": "norm_stats"}'
    ckpt_src = src_dir / "model_checkpoint.pt"
    norm_src = src_dir / "norm_stats.json"
    ckpt_src.write_bytes(ckpt_bytes)
    norm_src.write_bytes(norm_bytes)

    entry = _fake_available_entry(
        ckpt_src, norm_src, _sha256_of_bytes(ckpt_bytes), _sha256_of_bytes(norm_bytes)
    )

    fetch_calls = {"n": 0}
    real_fetch = registry._fetch_to_path

    def counting_fetch(url, dest):
        fetch_calls["n"] += 1
        return real_fetch(url, dest)

    monkeypatch.setattr(registry, "_fetch_to_path", counting_fetch)

    # First call: downloads + caches both artifacts.
    ckpt_path1, norm_path1 = registry._resolve_from_manifest_entry("2ch", entry)
    assert ckpt_path1.exists()
    assert norm_path1.exists()
    assert ckpt_path1.read_bytes() == ckpt_bytes
    assert norm_path1.read_bytes() == norm_bytes
    assert fetch_calls["n"] == 2  # one per artifact

    # Second call: cache hit, no re-download.
    ckpt_path2, norm_path2 = registry._resolve_from_manifest_entry("2ch", entry)
    assert ckpt_path2 == ckpt_path1
    assert norm_path2 == norm_path1
    assert fetch_calls["n"] == 2  # unchanged


def test_download_wrong_sha256_raises_and_leaves_no_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)
    cache_root = tmp_path / "cache_root"
    monkeypatch.setenv("WIND_DIFFUSION_CACHE_DIR", str(cache_root))

    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    ckpt_bytes = b"tiny fake checkpoint payload"
    norm_bytes = b'{"fake": "norm_stats"}'
    ckpt_src = src_dir / "model_checkpoint.pt"
    norm_src = src_dir / "norm_stats.json"
    ckpt_src.write_bytes(ckpt_bytes)
    norm_src.write_bytes(norm_bytes)

    wrong_sha = "0" * 64
    entry = _fake_available_entry(ckpt_src, norm_src, wrong_sha, _sha256_of_bytes(norm_bytes))

    with pytest.raises(WeightsDownloadError):
        registry._resolve_from_manifest_entry("3ch", entry)

    final_path = cache_root / "3ch" / "v1" / "model_checkpoint.pt"
    tmp_leftover = cache_root / "3ch" / "v1" / "model_checkpoint.pt.tmp"
    assert not final_path.exists()
    assert not tmp_leftover.exists()


def test_offline_with_available_status_still_raises_before_download(tmp_path, monkeypatch):
    """Offline should short-circuit even when the manifest entry says
    'available' -- no download attempt should be made."""
    monkeypatch.setenv("WIND_DIFFUSION_OFFLINE", "1")

    call_count = {"n": 0}

    def counting_fetch(url, dest):
        call_count["n"] += 1

    monkeypatch.setattr(registry, "_fetch_to_path", counting_fetch)

    entry = _fake_available_entry(tmp_path / "x.pt", tmp_path / "x.json", "abc", "def")

    with pytest.raises(CheckpointNotFoundError):
        registry._resolve_from_manifest_entry("1ch", entry)

    assert call_count["n"] == 0


def test_unknown_preset_raises_value_error():
    with pytest.raises(ValueError):
        registry.resolve_weights("4ch")
