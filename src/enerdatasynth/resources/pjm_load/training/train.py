"""
src/enerdatasynth/resources/pjm_load/training/train.py
======================================================
Training loop for a single configuration. Implements:
  - model checkpoint saving (best + latest)
  - training logs (per-epoch loss to CSV)
  - loss curve plot
  - optional early stopping (patience=10)

No custom losses, no regularization tricks — standard diffusion MSE
loss, trained with AdamW.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from enerdatasynth.utils.config import RunConfig, set_seed, CHECKPOINT_DIR, LOGS_DIR, ensure_dirs, resolve_device, project_relative
from enerdatasynth.resources.pjm_load.models import build_denoiser
from enerdatasynth.resources.pjm_load.diffusion.ddpm import DDPM


def train_one_config(X_train: np.ndarray, Z_train: np.ndarray,
                      X_val: np.ndarray, Z_val: np.ndarray,
                      cfg: RunConfig, device: str | None = None,
                      verbose: bool = True) -> dict:
    """
    Train a single model configuration end-to-end.

    Returns a result dict with training history, timing, and paths
    to the saved checkpoint and loss curve — everything the sweep
    runner and results report need.
    """
    ensure_dirs()
    device = resolve_device(device)
    set_seed(cfg.seed)

    x_dim    = X_train.shape[1]
    cond_dim = Z_train.shape[1]

    model = build_denoiser(
        cfg.architecture, x_dim=x_dim, cond_dim=cond_dim,
        hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
    ).to(device)
    n_params = model.count_parameters()

    ddpm = DDPM(T=cfg.diffusion_steps, device=device)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    Z_val_t   = torch.tensor(Z_val,   dtype=torch.float32, device=device)

    ds = TensorDataset(X_train_t, Z_train_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0

    run_id   = cfg.run_id
    ckpt_dir = CHECKPOINT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n── Training {run_id}  ({n_params:,} params, "
              f"T={cfg.diffusion_steps}, hidden={cfg.hidden_dim}, "
              f"layers={cfg.n_layers}) ──")

    t_start = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, zb in dl:
            xb, zb = xb.to(device), zb.to(device)
            loss = ddpm.training_loss(model, xb, zb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(dl)

        # Validation loss (same MSE objective, no gradient updates)
        model.eval()
        with torch.no_grad():
            val_loss = ddpm.training_loss(model, X_val_t, Z_val_t).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_ctr += 1

        torch.save(model.state_dict(), ckpt_dir / "latest.pt")

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  best_val={best_val_loss:.5f}"
                  f"{'  *' if improved else ''}")

        if cfg.patience is not None and patience_ctr >= cfg.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no val improvement for {cfg.patience} epochs)")
            break

    training_time = time.time() - t_start

    # ── Loss curve plot ──────────────────────────────────────────────
    # Imported here, not at module level: `visualization.plots` needs
    # matplotlib/statsmodels (the optional `research` extra), and
    # `enerdatasynth.resources.pjm_load.training.train` is imported eagerly by the top-level
    # `enerdatasynth/__init__.py` -- a module-level import here would force
    # every `import enerdatasynth` (including a bare `.sample()` call) to
    # require those two packages. Found and fixed during the packaging
    # audit via a clean-venv wheel-install test (see docs/release_audit.md).
    from enerdatasynth.resources.pjm_load.visualization.plots import plot_loss_curve
    loss_curve_path = ckpt_dir / "loss_curve.png"
    plot_loss_curve(history, run_id, best_epoch, loss_curve_path)

    # ── Save training log as CSV ─────────────────────────────────────
    log_path = LOGS_DIR / f"{run_id}_training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for e, tl, vl in zip(history["epoch"], history["train_loss"], history["val_loss"]):
            f.write(f"{e},{tl},{vl}\n")

    # ── Timing for one generation pass (used in results report) ──────
    t_gen_start = time.time()
    with torch.no_grad():
        _ = ddpm.sample(model, Z_val_t[:min(64, len(Z_val_t))], x_dim)
    generation_time_per_batch = time.time() - t_gen_start

    result = {
        "run_id": run_id,
        "architecture": cfg.architecture,
        "x_dim": x_dim,
        "cond_dim": cond_dim,
        "diffusion_steps": cfg.diffusion_steps,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_params": n_params,
        "epochs_trained": history["epoch"][-1],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_time_sec": training_time,
        "generation_time_sec_per_64": generation_time_per_batch,
        "checkpoint_best": project_relative(ckpt_dir / "best.pt"),
        "checkpoint_latest": project_relative(ckpt_dir / "latest.pt"),
        "loss_curve_path": project_relative(loss_curve_path),
        "training_log_path": project_relative(log_path),
    }

    with open(ckpt_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  Done. Best val loss {best_val_loss:.5f} at epoch {best_epoch}. "
              f"Training time {training_time:.1f}s")

    return result


if __name__ == "__main__":
    from enerdatasynth.utils.config import load_processed

    X_train = load_processed("X_daily_profiles.npy")
    Z_train = load_processed("Z_conditions.npy")
    X_val   = load_processed("X_daily_profiles_val.npy")
    Z_val   = load_processed("Z_conditions_val.npy")

    cfg = RunConfig()  # defaults: transformer, T=1000, model_dim=64, layers=2, heads=4
    train_one_config(X_train, Z_train, X_val, Z_val, cfg)
