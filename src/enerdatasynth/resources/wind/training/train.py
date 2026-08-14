"""Training loop for wind_diffusion models: run_epoch + train(config) +
checkpoint saving, plus a console-script entry point.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from enerdatasynth.resources.wind.config import PresetName, WindDiffusionConfig, get_config
from enerdatasynth.resources.wind.data.data import PreparedData, prepare_datasets
from enerdatasynth.resources.wind.diffusion.diffusion import DiffusionSchedule, q_sample
from enerdatasynth.resources.wind.models.models import TransformerDenoiser
from enerdatasynth.resources.wind.data.transforms import wind_persistence_slice, wind_regime_slice
from enerdatasynth.resources.wind.utils import resolve_device, set_seed


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, num_epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear LR warmup for `warmup_epochs`, then cosine annealing decay for
    the remaining epochs. `scheduler.step()` is called once per epoch (not
    per batch). If `warmup_epochs <= 0`, warmup is skipped entirely and plain
    cosine annealing runs over all `num_epochs`.
    """
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def _joint_radius_consistency_loss(
    x0_real: torch.Tensor,
    x_t: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    channels: tuple[int, int],
    clip_std: float,
) -> torch.Tensor:
    """Auxiliary loss penalizing the predicted x0's joint (c0, c1) radius --
    sqrt(c0^2 + c1^2) -- for diverging from the real x0's radius.

    Targets a specific measured failure mode (see
    reports/2CH_CLAMP_INVESTIGATION.md): for channel pairs that jointly
    encode a 2D vector magnitude (2ch's U/V), the primary per-element
    noise-prediction loss can match each channel's marginal distribution
    while still getting the *joint* radius systematically wrong -- 2ch's
    synthetic sqrt(U^2+V^2) measured ~2.8x inflated vs real, and
    generation-time clamping (independent or joint-radius) could not fix it
    because the defect is in what the model learned to predict, not how
    sampling clips it. This loss attacks that directly during training.

    x0_pred is derived from pred_noise the same way every sampler does
    (`(x_t - sqrt(1-ac_t)*pred_noise) / sqrt(ac_t)`) and clamped to
    `[-clip_std, clip_std]` before computing its radius -- unclamped x0_pred
    can blow up for large diffusion timesteps (small sqrt_alphas_cumprod
    denominator), and clamping also matches what generation actually uses.
    """
    c0, c1 = channels
    sqrt_ac_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1)
    sqrt_1m_ac_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    x0_pred = (x_t - sqrt_1m_ac_t * pred_noise) / sqrt_ac_t
    x0_pred = torch.clamp(x0_pred, -clip_std, clip_std)

    radius_pred = torch.sqrt(x0_pred[:, c0] ** 2 + x0_pred[:, c1] ** 2 + 1e-8)
    radius_real = torch.sqrt(x0_real[:, c0] ** 2 + x0_real[:, c1] ** 2 + 1e-8)
    return torch.mean((radius_pred - radius_real) ** 2)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    train: bool = True,
    cond_dropout_prob: float = 0.0,
    cond_dropout_slice: slice | None = None,
    joint_consistency_channels: tuple[int, int] | None = None,
    joint_consistency_weight: float = 0.0,
    sample_clip_std: float = 3.5,
    persistence_dropout_prob: float = 0.0,
    persistence_slice: slice | None = None,
) -> float:
    """Run one epoch of noise-prediction training/validation. Mirrors the
    original scripts' `run_epoch(loader, train=True)`, with model/optimizer/
    loss_fn/schedule/device passed explicitly instead of closed over.

    `cond_dropout_prob` (classifier-free guidance training): during training
    only, each sample's condition vector z has the slice `cond_dropout_slice`
    (default None = the entire vector) independently zeroed with this
    probability, so the model also learns to denoise with that part
    unconditioned. Never applied during validation -- val_loss should keep
    measuring the checkpoint's real conditional performance.

    `joint_consistency_weight`/`joint_consistency_channels`: when weight>0,
    adds `_joint_radius_consistency_loss` to the training loss only (not
    validation -- val_loss stays a pure noise-prediction MSE, comparable
    across every run regardless of this setting).

    `persistence_dropout_prob`/`persistence_slice`: independent masking of
    the wind-persistence feature (see transforms.wind_persistence_slice),
    drawn separately from `cond_dropout_prob`'s mask -- this lets the model
    learn to work both with and without a real previous-day value, matching
    the API's "optional parameter" UX (generate_wind_profile samples a
    realistic default when none is given, which is drawn from the real data
    distribution, not this zeroed/no-info training value -- see api.py).
    """
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0

    for x, z in loader:
        x = x.to(device)  # (B, channels, seq_len)
        z = z.to(device)
        b = x.shape[0]

        if train and (cond_dropout_prob > 0 or (persistence_dropout_prob > 0 and persistence_slice is not None)):
            z = z.clone()

        if train and cond_dropout_prob > 0:
            drop_mask = torch.rand(b, device=device) < cond_dropout_prob
            target_slice = cond_dropout_slice if cond_dropout_slice is not None else slice(None)
            z[drop_mask, target_slice] = 0.0

        if train and persistence_dropout_prob > 0 and persistence_slice is not None:
            persistence_drop_mask = torch.rand(b, device=device) < persistence_dropout_prob
            z[persistence_drop_mask, persistence_slice] = 0.0

        t = torch.randint(0, schedule.T, (b,), device=device).long()
        x_t, noise = q_sample(schedule, x, t)

        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            pred_noise = model(x_t, t, z)
            loss = loss_fn(pred_noise, noise)
            if joint_consistency_weight > 0 and joint_consistency_channels is not None:
                loss = loss + joint_consistency_weight * _joint_radius_consistency_loss(
                    x, x_t, pred_noise, t, schedule, joint_consistency_channels, sample_clip_std
                )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred_noise = model(x_t, t, z)
                loss = loss_fn(pred_noise, noise)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def build_model(config: WindDiffusionConfig, n_channels: int, cond_dim: int, seq_len: int) -> TransformerDenoiser:
    return TransformerDenoiser(
        n_channels=n_channels,
        cond_dim=cond_dim,
        hidden_dim=config.hidden_dim,
        num_conv_blocks=config.num_conv_blocks,
        num_transformer_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        time_emb_dim=config.time_emb_dim,
        cond_emb_dim=config.cond_emb_dim,
        seq_len=seq_len,
        norm_first=config.norm_first,
    )


def train(config: WindDiffusionConfig, data_path: str | None = None) -> dict:
    """Run the full training loop and save checkpoints/artifacts to
    `config.output_dir`. Returns a summary dict with paths and best val loss.
    """
    set_seed(config.seed)

    device = resolve_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    prepared: PreparedData = prepare_datasets(config, data_path=data_path)
    train_dataset = prepared.train_dataset
    val_dataset = prepared.val_dataset

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    n_channels = train_dataset.windows.shape[1]
    seq_len = train_dataset.windows.shape[2]
    cond_dim = prepared.norm_stats.cond_dim

    # Save normalization stats up front (needed for inference regardless of
    # whether training completes / early-stops).
    norm_stats_path = os.path.join(config.output_dir, "norm_stats.json")
    with open(norm_stats_path, "w") as f:
        json.dump(prepared.norm_stats.to_dict(), f, indent=2)

    model = build_model(config, n_channels, cond_dim, seq_len).to(device)

    schedule = DiffusionSchedule.build(
        T=config.diffusion_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, config.num_epochs, config.warmup_epochs)
    loss_fn = nn.MSELoss()

    cond_dropout_slice = None
    if config.cond_dropout_prob > 0 and config.cond_dropout_scope == "wind_regime":
        cond_dropout_slice = wind_regime_slice(config.condition_on, wind_regime_encoding=config.wind_regime_encoding)

    persistence_slice = None
    if config.condition_on.get("wind_persistence") and config.wind_persistence_dropout_prob > 0:
        persistence_slice = wind_persistence_slice(config.condition_on, wind_regime_encoding=config.wind_regime_encoding)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(config.output_dir, "model_checkpoint.pt")

    for epoch in range(1, config.num_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, schedule, optimizer, loss_fn, device, train=True,
            cond_dropout_prob=config.cond_dropout_prob,
            cond_dropout_slice=cond_dropout_slice,
            joint_consistency_channels=config.joint_consistency_channels,
            joint_consistency_weight=config.joint_consistency_weight,
            sample_clip_std=config.sample_clip_std,
            persistence_dropout_prob=config.wind_persistence_dropout_prob,
            persistence_slice=persistence_slice,
        )
        val_loss = run_epoch(model, val_loader, schedule, None, loss_fn, device, train=False)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if config.save_checkpoints:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config.output_dir, "training_history.csv"), index=False)

    if config.make_plots:
        from enerdatasynth.resources.wind.evaluation.evaluate import plot_loss_curve

        plot_loss_curve(history_df, os.path.join(config.output_dir, "loss_curve.png"))

    # Load best checkpoint before returning (mirrors original scripts'
    # "load best checkpoint before sampling" step).
    if config.save_checkpoints and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    return {
        "output_dir": config.output_dir,
        "checkpoint_path": ckpt_path if config.save_checkpoints else None,
        "norm_stats_path": norm_stats_path,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a wind_diffusion conditional DDPM model.")
    parser.add_argument("--preset", choices=["1ch", "2ch", "3ch"], default="2ch")
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-conv-blocks", type=int, default=None)
    parser.add_argument("--num-transformer-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--cond-dropout-prob", type=float, default=None,
        help="Probability of zeroing part of a training sample's condition vector (classifier-free guidance training). Default 0.0 (disabled).",
    )
    parser.add_argument(
        "--cond-dropout-scope", choices=["full", "wind_regime"], default=None,
        help='Which part of z cond_dropout_prob zeroes: "full" (entire vector) or "wind_regime" (only the 3-dim regime one-hot slice). Default "full".',
    )
    parser.add_argument(
        "--wind-regime-encoding", choices=["onehot", "ordinal"], default=None,
        help='How wind_regime_bucket is encoded: "onehot" (3 dims, default) or "ordinal" (1 dim, low=1/mid=2/high=3 -- tests whether giving "mid" a genuine midpoint geometry fixes its persistently weak regime-steering accuracy).',
    )
    parser.add_argument(
        "--joint-consistency-channels", type=int, nargs=2, default=None, metavar=("C0", "C1"),
        help="Two channel indices whose predicted-x0 joint radius sqrt(c0^2+c1^2) is penalized against the real radius (e.g. 0 1 for 2ch's U/V). Default None (disabled).",
    )
    parser.add_argument(
        "--joint-consistency-weight", type=float, default=None,
        help="Weight for the joint-consistency auxiliary loss. Default 0.0 (disabled).",
    )
    parser.add_argument(
        "--wind-persistence", action="store_true",
        help="Enable conditioning on yesterday's actual daily-mean windspeed (opt-in, off by default; see config.py's wind_persistence_dropout_prob docstring).",
    )
    parser.add_argument(
        "--wind-persistence-dropout-prob", type=float, default=None,
        help="Probability of zeroing the wind-persistence feature during training, independent of --cond-dropout-prob. Default 0.3.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help='"auto" (default), "cpu", or "cuda"')
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.num_conv_blocks is not None:
        overrides["num_conv_blocks"] = args.num_conv_blocks
    if args.num_transformer_layers is not None:
        overrides["num_transformer_layers"] = args.num_transformer_layers
    if args.num_heads is not None:
        overrides["num_heads"] = args.num_heads
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.cond_dropout_prob is not None:
        overrides["cond_dropout_prob"] = args.cond_dropout_prob
    if args.cond_dropout_scope is not None:
        overrides["cond_dropout_scope"] = args.cond_dropout_scope
    if args.wind_regime_encoding is not None:
        overrides["wind_regime_encoding"] = args.wind_regime_encoding
    if args.joint_consistency_channels is not None:
        overrides["joint_consistency_channels"] = tuple(args.joint_consistency_channels)
    if args.joint_consistency_weight is not None:
        overrides["joint_consistency_weight"] = args.joint_consistency_weight
    if args.wind_persistence_dropout_prob is not None:
        overrides["wind_persistence_dropout_prob"] = args.wind_persistence_dropout_prob
    if args.norm_first is not None:
        overrides["norm_first"] = args.norm_first
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.data_path is not None:
        overrides["data_path"] = args.data_path
    if args.no_plots:
        overrides["make_plots"] = False

    preset: PresetName = args.preset
    config = get_config(preset, **overrides)
    if args.wind_persistence:
        config.condition_on = {**config.condition_on, "wind_persistence": True}

    summary = train(config, data_path=args.data_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
