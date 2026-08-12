"""Weights registry: resolves (checkpoint_path, norm_stats_path) for a preset
from a bundled manifest, with local-override and offline support, plus a
cached/verified download path for when published weights exist.

Manifest schema (`manifest.json`, bundled inside the package and loaded via
`importlib.resources` so this works from an installed wheel, not just a
source checkout). One entry per `PresetName` ("1ch"/"2ch"/"3ch"):

    {
      "<preset>": {
        "status": "pending" | "available",
        "version": str | null,              # cache-directory version tag
        "checkpoint_url": str | null,        # download URL for model_checkpoint.pt
        "checkpoint_sha256": str | null,
        "norm_stats_url": str | null,        # download URL for norm_stats.json
        "norm_stats_sha256": str | null,
        "seq_len": int,
        "freq": str,
        "feature_cols": list[str],           # human-readable physical columns
        "channel_description": str,
        "metrics": dict | null
      },
      ...
    }

This deviates from a single `url`/`sha256` pair (as sketched in the task
spec) because each preset needs *two* artifacts fetched and verified
independently (the model checkpoint and its norm_stats.json) -- splitting
the fields avoids inventing a fake bundling/archive format that doesn't
exist yet.

`resolve_weights` precedence (first match wins):
    1. Explicit `checkpoint=`/`norm_stats=` args (both must be given; both
       must exist on disk).
    2. `WIND_DIFFUSION_CHECKPOINTS` env var: a directory expected to contain
       `<preset>/model_checkpoint.pt` and `<preset>/norm_stats.json`.
    3. `WIND_DIFFUSION_OFFLINE` env var (truthy: "1"/"true"/"yes"/"on"): if
       set, no download is attempted -- raises `CheckpointNotFoundError`
       immediately, before even checking the manifest `status`. This is
       checked *before* the manifest status (see below) since offline is a
       hard user directive that should short-circuit regardless of whether
       a publish would have been possible.
    4. Manifest `status == "available"`: cached/verified download to
       `<cache_root>/<preset>/<version>/`, where `<cache_root>` is
       `Path.home() / ".cache" / "wind_diffusion"` by default, overridable
       via `WIND_DIFFUSION_CACHE_DIR` (mainly for tests).
    5. Manifest `status == "pending"` (true for all three presets right
       now, since no gated checkpoint has been trained yet): raises
       `CheckpointNotFoundError` naming the preset and the local training
       command.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from wind_diffusion.config import PresetName
from wind_diffusion.exceptions import CheckpointNotFoundError, WeightsDownloadError

_PRESETS: tuple[str, ...] = ("1ch", "2ch", "3ch")

_ENV_CHECKPOINTS_DIR = "WIND_DIFFUSION_CHECKPOINTS"
_ENV_OFFLINE = "WIND_DIFFUSION_OFFLINE"
_ENV_CACHE_DIR = "WIND_DIFFUSION_CACHE_DIR"

_TRAIN_CMD_TEMPLATE = "wind-diffusion-train --preset {preset} --data-path <csv> --output-dir <dir>"

_manifest_cache: dict | None = None


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    """Load and cache the bundled manifest.json via importlib.resources.

    Uses `importlib.resources` (not a relative file path) so this resolves
    correctly whether wind_diffusion is imported from a source checkout or
    an installed wheel/sdist.
    """
    global _manifest_cache
    if _manifest_cache is None:
        text = importlib.resources.files("wind_diffusion").joinpath("manifest.json").read_text(encoding="utf-8")
        _manifest_cache = json.loads(text)
    return _manifest_cache


def _get_manifest_entry(preset: PresetName) -> dict:
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {_PRESETS}")
    manifest = load_manifest()
    if preset not in manifest:
        raise CheckpointNotFoundError(
            f"Preset {preset!r} has no manifest entry at all (manifest.json is malformed or out of date)."
        )
    return manifest[preset]


def list_models() -> dict:
    """Return a deep copy of the full manifest: preset -> status/metrics/etc."""
    return copy.deepcopy(load_manifest())


def describe_model(preset: PresetName) -> dict:
    """Return one preset's manifest entry plus current local cache-state info."""
    entry = dict(_get_manifest_entry(preset))  # shallow copy is enough, values are immutable JSON scalars/lists

    version = entry.get("version")
    cached_locally = False
    cache_path: str | None = None
    if version:
        cache_dir = _cache_dir(preset, version)
        ckpt = cache_dir / "model_checkpoint.pt"
        norm = cache_dir / "norm_stats.json"
        if ckpt.exists() and norm.exists():
            cached_locally = True
            cache_path = str(cache_dir)

    env_dir = os.environ.get(_ENV_CHECKPOINTS_DIR)
    env_override_dir = None
    env_override_available = False
    if env_dir:
        base = Path(env_dir) / preset
        env_override_dir = str(base)
        env_override_available = (base / "model_checkpoint.pt").exists() and (base / "norm_stats.json").exists()

    entry["cached_locally"] = cached_locally
    entry["cache_path"] = cache_path
    entry["env_override_dir"] = env_override_dir
    entry["env_override_available"] = env_override_available
    return entry


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _is_offline() -> bool:
    val = os.environ.get(_ENV_OFFLINE, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _cache_root() -> Path:
    override = os.environ.get(_ENV_CACHE_DIR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "wind_diffusion"


def _cache_dir(preset: str, version: str) -> Path:
    return _cache_root() / preset / str(version)


# ---------------------------------------------------------------------------
# Actionable error messages
# ---------------------------------------------------------------------------


def _train_cmd(preset: str) -> str:
    return _TRAIN_CMD_TEMPLATE.format(preset=preset)


def _checked_paths_summary(preset: str) -> str:
    env_dir = os.environ.get(_ENV_CHECKPOINTS_DIR)
    return (
        f"checked: explicit checkpoint=/norm_stats= arguments (none given, or incomplete), "
        f"the {_ENV_CHECKPOINTS_DIR} directory env var "
        f"({env_dir!r} -> {Path(env_dir) / preset if env_dir else 'n/a'}), "
        f"and the bundled manifest for preset {preset!r}"
    )


def _pending_message(preset: str) -> str:
    return (
        f"No published weights yet for preset {preset!r} (manifest status is 'pending' -- "
        f"this is expected, not a bug: no gated checkpoint has been trained/published for "
        f"this preset yet). {_checked_paths_summary(preset)}. "
        f"WIND_DIFFUSION_OFFLINE is {'set' if _is_offline() else 'not set'}. "
        f"Train one locally with: `{_train_cmd(preset)}`, then pass its "
        f"output_dir/model_checkpoint.pt and output_dir/norm_stats.json via "
        f"checkpoint=/norm_stats=, or point {_ENV_CHECKPOINTS_DIR} at a directory "
        f"containing `{preset}/model_checkpoint.pt` and `{preset}/norm_stats.json`."
    )


def _offline_message(preset: str) -> str:
    return (
        f"WIND_DIFFUSION_OFFLINE is set, so no weights download will be attempted for "
        f"preset {preset!r}. {_checked_paths_summary(preset)}. "
        f"Unset {_ENV_OFFLINE} to allow a download (if the manifest has published weights "
        f"for this preset), or train locally with: `{_train_cmd(preset)}` and pass the "
        f"result via checkpoint=/norm_stats= or {_ENV_CHECKPOINTS_DIR}."
    )


def _explicit_path_missing_message(preset: str, kind: str, path: Path) -> str:
    return (
        f"Explicit {kind} path does not exist: {path} (preset {preset!r}). "
        f"{_checked_paths_summary(preset)}. "
        f"WIND_DIFFUSION_OFFLINE is {'set' if _is_offline() else 'not set'}. "
        f"Train locally with: `{_train_cmd(preset)}`."
    )


def _env_dir_missing_message(preset: str, ckpt_path: Path, norm_path: Path) -> str:
    env_dir = os.environ.get(_ENV_CHECKPOINTS_DIR)
    return (
        f"{_ENV_CHECKPOINTS_DIR} is set to {env_dir!r} but {ckpt_path} / {norm_path} "
        f"were not both found for preset {preset!r}. "
        f"WIND_DIFFUSION_OFFLINE is {'set' if _is_offline() else 'not set'}. "
        f"Train locally with: `{_train_cmd(preset)}`, or unset {_ENV_CHECKPOINTS_DIR} to "
        f"fall back to the bundled manifest/download path."
    )


# ---------------------------------------------------------------------------
# Download + cache (sha256-verified, atomic)
# ---------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_to_path(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`. Supports http(s) (streamed via `requests`,
    never loaded fully into memory), `file://` URLs, and plain local
    filesystem paths (the latter two mainly so tests can exercise the full
    download+cache+verify flow without any real network access).

    Module-level (not nested) so tests can monkeypatch it directly to
    assert on call counts.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        import requests

        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    elif parsed.scheme == "file":
        shutil.copyfile(Path(parsed.path), dest)
    else:
        # Plain local filesystem path.
        shutil.copyfile(Path(url), dest)


def _download_and_cache_artifact(url: str, sha256: str, cache_dir: Path, filename: str) -> Path:
    """Download `url` to `cache_dir/filename` if not already cached with a
    matching sha256, verifying the checksum and writing atomically
    (download to `.tmp`, verify, then rename). Raises `WeightsDownloadError`
    on a checksum mismatch and removes the bad file (never leaves a corrupt
    file at the final destination).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename

    if dest.exists() and _sha256_of(dest) == sha256:
        return dest  # already cached and valid; skip re-download

    tmp = cache_dir / f"{filename}.tmp"
    if tmp.exists():
        tmp.unlink()

    try:
        _fetch_to_path(url, tmp)
        actual = _sha256_of(tmp)
        if actual != sha256:
            raise WeightsDownloadError(
                f"sha256 mismatch downloading {url} -> {dest}: expected {sha256}, got {actual}."
            )
        tmp.replace(dest)  # atomic on same filesystem
    finally:
        if tmp.exists():
            tmp.unlink()

    return dest


def _resolve_from_manifest_entry(preset: str, entry: dict) -> tuple[Path, Path]:
    """Resolve (checkpoint_path, norm_stats_path) from a manifest entry dict
    directly (bypassing `load_manifest()`), so tests can exercise the
    download/cache/verify flow against a fake entry without needing a real
    published preset.
    """
    # Offline is checked before manifest status: it's a hard user directive
    # that should short-circuit immediately, and the resulting message
    # should say "offline" rather than "pending" even if status happens to
    # also be pending (matches the current real-manifest state).
    if _is_offline():
        raise CheckpointNotFoundError(_offline_message(preset))

    status = entry.get("status")
    if status != "available":
        raise CheckpointNotFoundError(_pending_message(preset))

    version = entry.get("version")
    cache_dir = _cache_dir(preset, version)

    ckpt_path = _download_and_cache_artifact(
        entry["checkpoint_url"], entry["checkpoint_sha256"], cache_dir, "model_checkpoint.pt"
    )
    norm_path = _download_and_cache_artifact(
        entry["norm_stats_url"], entry["norm_stats_sha256"], cache_dir, "norm_stats.json"
    )
    return ckpt_path, norm_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_weights(
    preset: PresetName,
    checkpoint: str | Path | None = None,
    norm_stats: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve (checkpoint_path, norm_stats_path) for `preset`. See module
    docstring for the full precedence order.
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {_PRESETS}")

    # 1. Explicit args.
    if checkpoint is not None and norm_stats is not None:
        ckpt_path = Path(checkpoint)
        norm_path = Path(norm_stats)
        if not ckpt_path.exists():
            raise CheckpointNotFoundError(_explicit_path_missing_message(preset, "checkpoint", ckpt_path))
        if not norm_path.exists():
            raise CheckpointNotFoundError(_explicit_path_missing_message(preset, "norm_stats", norm_path))
        return ckpt_path, norm_path
    if (checkpoint is None) != (norm_stats is None):
        raise CheckpointNotFoundError(
            f"Both `checkpoint` and `norm_stats` must be given together, or neither, "
            f"for preset {preset!r} (got checkpoint={checkpoint!r}, norm_stats={norm_stats!r})."
        )

    # 2. WIND_DIFFUSION_CHECKPOINTS directory override.
    env_dir = os.environ.get(_ENV_CHECKPOINTS_DIR)
    if env_dir:
        base = Path(env_dir) / preset
        ckpt_path = base / "model_checkpoint.pt"
        norm_path = base / "norm_stats.json"
        if ckpt_path.exists() and norm_path.exists():
            return ckpt_path, norm_path
        raise CheckpointNotFoundError(_env_dir_missing_message(preset, ckpt_path, norm_path))

    # 3-5. Manifest-driven cached download, offline gate, or pending error.
    entry = _get_manifest_entry(preset)
    return _resolve_from_manifest_entry(preset, entry)
