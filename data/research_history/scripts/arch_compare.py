"""
Architecture comparison for conditional load diffusion  (RUN ON YOUR MAC)
=========================================================================
Controlled experiment: ONLY the denoiser backbone changes. Everything else --
data, Z5 conditioning, T=1000 linear beta schedule, epochs, lr, batch, seed,
and the corrected ancestral sampler -- is held identical.

Architectures compared:
  mlp          Current baseline: 4 residual blocks, width 256  (~688K params)
  cnn          1D conv over the 24-hour axis (dilated, residual)
  attention    MLP trunk + one self-attention block over the 24 hours
  transformer  Transformer encoder over the 24 hours (2 layers, 4 heads)

All four treat the 24-hour profile as a sequence except `mlp`, which sees it
as a flat vector -- that is exactly the hypothesis being tested: does giving
the model an explicit temporal axis help?

Usage (from project root):
    /opt/homebrew/bin/python3.11 arch_compare.py --epochs 300

Outputs -> reports/week6/arch/
    metrics.csv, summary_report.md, convergence.png, profiles.png

Requires: numpy, matplotlib, torch
"""
import argparse, math, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"

Xtr = np.load(OUTDIR/"X_train.npy").astype(np.float32)
Xva = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Ztr = np.load(OUTDIR/"Z5_train.npy").astype(np.float32)
Zva = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
Z_DIM = Ztr.shape[1]; H = 24

# ════════════════════════════════════════════════════════════════════════
# Shared pieces
# ════════════════════════════════════════════════════════════════════════
class SinusoidalEmbedding(nn.Module):
    def __init__(s, dim): super().__init__(); s.dim = dim
    def forward(s, t):
        h = s.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

class ResBlock(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)); s.act = nn.SiLU()
    def forward(s, x): return s.act(x + s.net(x))

# ── 1. MLP (current baseline, verbatim) ─────────────────────────────────
class MLPDenoiser(nn.Module):
    def __init__(s, x_dim=H, z_dim=Z_DIM, hidden=256, n_blocks=4, t_dim=64):
        super().__init__()
        s.t_embed = SinusoidalEmbedding(t_dim)
        s.z_proj = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        s.input_proj = nn.Linear(x_dim + t_dim + hidden, hidden)
        s.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        s.out = nn.Linear(hidden, x_dim)
    def forward(s, x, t, z):
        h = torch.cat([x, s.t_embed(t), s.z_proj(z)], -1)
        h = s.input_proj(h)
        for b in s.blocks: h = b(h)
        return s.out(h)

# ── 2. 1D CNN over the 24-hour axis ─────────────────────────────────────
class ConvBlock(nn.Module):
    """Residual dilated conv; FiLM-style conditioning injected per block."""
    def __init__(s, ch, dil, cond_dim):
        super().__init__()
        s.conv1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        s.conv2 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        s.cond  = nn.Linear(cond_dim, ch)          # add conditioning as a bias
        s.norm  = nn.GroupNorm(8, ch); s.act = nn.SiLU()
    def forward(s, h, c):
        r = h
        h = s.act(s.norm(s.conv1(h)))
        h = h + s.cond(c)[:, :, None]              # broadcast over the 24 hours
        h = s.conv2(h)
        return s.act(h + r)

class CNNDenoiser(nn.Module):
    def __init__(s, x_dim=H, z_dim=Z_DIM, ch=64, n_blocks=4, t_dim=64):
        super().__init__()
        s.t_embed = SinusoidalEmbedding(t_dim)
        s.cond = nn.Sequential(nn.Linear(z_dim + t_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        s.inp = nn.Conv1d(1, ch, 3, padding=1)
        s.blocks = nn.ModuleList([ConvBlock(ch, d, 128) for d in [1, 2, 4, 8][:n_blocks]])
        s.out = nn.Conv1d(ch, 1, 3, padding=1)
    def forward(s, x, t, z):
        c = s.cond(torch.cat([z, s.t_embed(t)], -1))
        h = s.inp(x[:, None, :])                   # (B,1,24) -> (B,ch,24)
        for b in s.blocks: h = b(h, c)
        return s.out(h)[:, 0, :]                   # (B,24)

# ── 3. MLP trunk + one self-attention block over hours ──────────────────
class AttnDenoiser(nn.Module):
    def __init__(s, x_dim=H, z_dim=Z_DIM, d=64, n_heads=4, t_dim=64):
        super().__init__()
        s.t_embed = SinusoidalEmbedding(t_dim)
        s.cond = nn.Sequential(nn.Linear(z_dim + t_dim, d), nn.SiLU(), nn.Linear(d, d))
        s.val = nn.Linear(1, d)                    # each hour -> a token
        s.pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        s.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        s.norm1 = nn.LayerNorm(d); s.norm2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))
        s.out = nn.Linear(d, 1)
    def forward(s, x, t, z):
        c = s.cond(torch.cat([z, s.t_embed(t)], -1))[:, None, :]   # (B,1,d)
        h = s.val(x[:, :, None]) + s.pos + c                        # (B,24,d)
        a, _ = s.attn(s.norm1(h), s.norm1(h), s.norm1(h))
        h = h + a
        h = h + s.ff(s.norm2(h))
        return s.out(h)[:, :, 0]                                    # (B,24)

# ── 4. Transformer encoder over hours ───────────────────────────────────
class TransformerDenoiser(nn.Module):
    def __init__(s, x_dim=H, z_dim=Z_DIM, d=64, n_heads=4, n_layers=2, t_dim=64):
        super().__init__()
        s.t_embed = SinusoidalEmbedding(t_dim)
        s.cond = nn.Sequential(nn.Linear(z_dim + t_dim, d), nn.SiLU(), nn.Linear(d, d))
        s.val = nn.Linear(1, d)
        s.pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4,
                                           batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        s.out = nn.Linear(d, 1)
    def forward(s, x, t, z):
        c = s.cond(torch.cat([z, s.t_embed(t)], -1))[:, None, :]
        h = s.val(x[:, :, None]) + s.pos + c
        h = s.enc(h)
        return s.out(h)[:, :, 0]

ARCHS = {"mlp": MLPDenoiser, "cnn": CNNDenoiser,
         "attention": AttnDenoiser, "transformer": TransformerDenoiser}

# ════════════════════════════════════════════════════════════════════════
# DDPM (linear schedule) + CORRECT ancestral sampler
# ════════════════════════════════════════════════════════════════════════
class DDPM:
    def __init__(s, T=1000, b0=1e-4, b1=0.02):
        s.T = T
        s.betas = torch.linspace(b0, b1, T)
        s.alphas = 1.0 - s.betas
        s.alpha_bar = torch.cumprod(s.alphas, 0)
    def training_loss(s, model, x0, z):
        B = x0.shape[0]
        t = torch.randint(0, s.T, (B,))
        noise = torch.randn_like(x0)
        ab = s.alpha_bar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return nn.functional.mse_loss(model(xt, t, z), noise)
    @torch.no_grad()
    def sample(s, model, z):
        B = z.shape[0]; x = torch.randn(B, H)
        for t in reversed(range(s.T)):
            tv = torch.full((B,), t, dtype=torch.long)
            eps = model(x, tv, z)
            a, ab, b = s.alphas[t], s.alpha_bar[t], s.betas[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (b.sqrt() * torch.randn_like(x) if t > 0 else 0)
        return x.numpy()

# ── metrics (same 7 as every other script) ──────────────────────────────
def _acf(M, ml=12):
    o = []
    for row in M:
        row = row - row.mean(); d = (row * row).sum()
        if d <= 0: continue
        o.append([(row[:len(row)-k] * row[k:]).sum() / d for k in range(ml + 1)])
    return np.array(o).mean(0) if o else np.zeros(ml + 1)
def metrics(r, g):
    return dict(
        Hourly = float(np.abs(r.mean(0)-g.mean(0)).mean() + np.abs((np.percentile(r,75,0)-np.percentile(r,25,0))-(np.percentile(g,75,0)-np.percentile(g,25,0))).mean()),
        FirstDiff = float(abs(np.diff(r,1).std() - np.diff(g,1).std())),
        ACF = float(np.abs(_acf(r)-_acf(g)).mean()),
        RMSE = float(np.sqrt(((r.mean(0)-g.mean(0))**2).mean())),
        Peak = float(abs(r.max(1).mean()-g.max(1).mean())),
        Energy = float(abs(r.sum(1).mean()-g.sum(1).mean())),
        Ramp = float(abs(np.abs(np.diff(r,1)).mean()-np.abs(np.diff(g,1)).mean())),
    )
MK = ["Hourly","FirstDiff","ACF","RMSE","Peak","Energy","Ramp"]

# ════════════════════════════════════════════════════════════════════════
def main(args):
    outdir = ROOT/"reports"/"week6"/"arch"; outdir.mkdir(parents=True, exist_ok=True)
    Xtr_t, Ztr_t = torch.tensor(Xtr), torch.tensor(Ztr)
    Xva_t, Zva_t = torch.tensor(Xva), torch.tensor(Zva)
    ddpm = DDPM()
    results, hist, gens = [], {}, {}

    for name in args.archs:
        print(f"\n=== {name} ===")
        torch.manual_seed(0); np.random.seed(0)      # identical init/data order
        model = ARCHS[name]()
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        dl = DataLoader(TensorDataset(Xtr_t, Ztr_t), batch_size=args.batch, shuffle=True, drop_last=True)
        print(f"  params: {n_par:,}")

        vh, best, best_state = [], 1e9, None
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, zb in dl:
                loss = ddpm.training_loss(model, xb, zb)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([ddpm.training_loss(model, Xva_t, Zva_t).item() for _ in range(4)]))
            vh.append(vl)
            if vl < best: best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 25 == 0 or ep == 1: print(f"  ep {ep:>3}  val {vl:.5f}  best {best:.5f}")
        model.load_state_dict(best_state); model.eval()
        torch.save(model.state_dict(), OUTDIR/f"arch_{name}_z5.pt")

        g = ddpm.sample(model, Zva_t)
        mm = metrics(Xva, g)
        oob = float(np.mean((g < -0.05) | (g > 1.05)) * 100)
        mins = (time.time() - t0) / 60
        gens[name] = g; hist[name] = vh
        results.append(dict(arch=name, params=n_par, best_val=round(best, 5),
                            minutes=round(mins, 1), out_of_range_pct=round(oob, 1),
                            **{k: round(mm[k], 4) for k in MK}))
        print(f"  best val {best:.5f} | RMSE {mm['RMSE']:.4f} | out-of-range {oob:.1f}% | {mins:.1f} min")

    # ── table ──
    print("\n" + "=" * 100)
    print(f"{'arch':13s}{'params':>10s}{'best_val':>10s}" + "".join(f"{k:>11s}" for k in MK))
    for r in results:
        print(f"{r['arch']:13s}{r['params']:>10,}{r['best_val']:>10.5f}" + "".join(f"{r[k]:>11.4f}" for k in MK))
    print("=" * 100)

    with open(outdir/"metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch","params","best_val","minutes","out_of_range_pct"] + MK)
        w.writeheader(); w.writerows(results)
    print(f"Wrote {outdir/'metrics.csv'}")

    # ── figures ──
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,
                         "grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
    COL = {"mlp":"#3266ad","cnn":"#1d9e75","attention":"#e09320","transformer":"#7B5EA7"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for n in args.archs:
        ax.plot(range(1, len(hist[n])+1), hist[n], color=COL[n], lw=2, label=n)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss (log)")
    ax.set_title("Architecture comparison — convergence", fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"convergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    hours = np.arange(H)
    ax.plot(hours, Xva.mean(0), color="black", lw=3, label="Real", zorder=10)
    for n in args.archs:
        ax.plot(hours, gens[n].mean(0), color=COL[n], lw=2, label=n, alpha=0.9)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalized load [0,1]"); ax.set_xticks(range(0, 24, 3))
    ax.set_title("Mean daily profile by architecture", fontweight="bold"); ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(outdir/"profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── report ──
    best_val = min(results, key=lambda r: r["best_val"])
    best_rmse = min(results, key=lambda r: r["RMSE"])
    lines = ["# Week 6 — Architecture Comparison\n",
             "Controlled experiment: only the denoiser backbone changes. Identical data, Z5 "
             "conditioning, T=1000 linear schedule, epochs, lr, batch, seed, and sampler.\n",
             "| Arch | Params | Best val | RMSE | Hourly | FirstDiff | ACF | Peak | Energy | Ramp | Min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['arch']} | {r['params']:,} | {r['best_val']} | {r['RMSE']} | {r['Hourly']} | "
                     f"{r['FirstDiff']} | {r['ACF']} | {r['Peak']} | {r['Energy']} | {r['Ramp']} | {r['minutes']} |")
    lines.append(f"\n*Lowest validation loss:* **{best_val['arch']}** ({best_val['best_val']}).")
    lines.append(f"*Best generated profile RMSE:* **{best_rmse['arch']}** ({best_rmse['RMSE']}).")
    lines.append("\nNote: validation loss is the noise-prediction MSE and is not directly comparable to "
                 "generation quality. Judge architectures on the generated metrics, not val loss alone.")
    (outdir/"summary_report.md").write_text("\n".join(lines))
    print(f"Wrote {outdir/'summary_report.md'}\nDONE. See {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--archs", nargs="+", default=["mlp","cnn","attention","transformer"],
                    choices=list(ARCHS))
    main(ap.parse_args())