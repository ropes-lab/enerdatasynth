"""
Train the Z5 baseline diffusion model to convergence  (RUN ON YOUR MAC)
=======================================================================
Reuses your week3.py model + DDPM verbatim. Adds what week3 didn't have:
  • validation-loss tracking each epoch (to see the plateau / catch overfitting)
  • best-val checkpoint saving
  • a train-vs-val loss curve

It does NOT touch baseline_model_z5.pt. The converged model is saved separately
as converged_model_z5.pt, so you can benchmark BOTH (under-trained vs converged).

Run from your project root:
    /opt/homebrew/bin/python3.11 train_converged.py

Requires: pip install numpy matplotlib torch
"""

import time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config (matches week3.py) ────────────────────────────────────────────
LR          = 3e-4
T_STEPS     = 1000
BATCH_SIZE  = 64
MAX_EPOCHS  = 300          # cap; early stopping usually ends sooner
PATIENCE    = 25           # stop if val hasn't improved for this many epochs
DEVICE      = "cpu"
SEED        = 0

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR_CANDIDATES = [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")]
OUTDIR = next((p for p in OUTDIR_CANDIDATES if p.exists()), None)
if OUTDIR is None:
    raise FileNotFoundError("Could not find week3_outputs/. Edit OUTDIR_CANDIDATES.")
print(f"Using outputs dir: {OUTDIR}")

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load arrays ──────────────────────────────────────────────────────────
X_train = np.load(OUTDIR/"X_train.npy").astype(np.float32)     # (2785, 24)
X_val   = np.load(OUTDIR/"X_val.npy").astype(np.float32)       # (600, 24)
Z_train = np.load(OUTDIR/"Z5_train.npy").astype(np.float32)    # (2785, 6)
Z_val   = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)      # (600, 6)
Z_DIM   = Z_train.shape[1]
print(f"X_train{X_train.shape}  X_val{X_val.shape}  Z_dim={Z_DIM}")

X_t  = torch.tensor(X_train); Z_t  = torch.tensor(Z_train)
Xv_t = torch.tensor(X_val);   Zv_t = torch.tensor(Z_val)

# ════════════════════════════════════════════════════════════════════════
# MODEL + DDPM — copied verbatim from your week3.py
# ════════════════════════════════════════════════════════════════════════
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim=24, z_dim=6, hidden=256, n_blocks=4, t_embed_dim=64):
        super().__init__()
        self.t_embed   = SinusoidalEmbedding(t_embed_dim)
        self.z_proj    = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        in_dim         = x_dim + t_embed_dim + hidden
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks    = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out       = nn.Linear(hidden, x_dim)
    def forward(self, x_t, t, z):
        t_emb  = self.t_embed(t)
        z_feat = self.z_proj(z)
        h = torch.cat([x_t, t_emb, z_feat], dim=-1)
        h = self.input_proj(h)
        for b in self.blocks: h = b(h)
        return self.out(h)

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T; self.device = device
        betas = torch.linspace(beta_start, beta_end, T)
        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)
    def training_loss(self, model, x0, z):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        eps_hat = model(x_t, t, z)
        return nn.functional.mse_loss(eps_hat, noise)

# ════════════════════════════════════════════════════════════════════════
# TRAIN with val tracking + best-checkpoint + early stopping
# ════════════════════════════════════════════════════════════════════════
model = ConditionalDenoiser(x_dim=24, z_dim=Z_DIM, hidden=256, n_blocks=4, t_embed_dim=64).to(DEVICE)
ddpm  = DDPM(T=T_STEPS, device=DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

dl = DataLoader(TensorDataset(X_t, Z_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

@torch.no_grad()
def val_loss(n_repeats=8):
    # average the diffusion MSE over several random-timestep draws for a stable estimate
    model.eval(); tot = 0.0
    for _ in range(n_repeats):
        tot += ddpm.training_loss(model, Xv_t, Zv_t).item()
    return tot / n_repeats

print(f"\nTraining to convergence (max {MAX_EPOCHS} epochs, patience {PATIENCE})")
print(f"  T={T_STEPS}, lr={LR}, batch={BATCH_SIZE}, params={sum(p.numel() for p in model.parameters()):,}")
print(f"  {'epoch':>6}  {'train':>10}  {'val':>10}  {'best_val':>10}")

train_hist, val_hist = [], []
best_val = float("inf"); best_epoch = 0; bad = 0
best_state = None
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for xb, zb in dl:
        loss = ddpm.training_loss(model, xb, zb)
        opt.zero_grad(); loss.backward(); opt.step()
        ep += loss.item()
    sched.step()
    tr = ep / len(dl); vl = val_loss()
    train_hist.append(tr); val_hist.append(vl)

    if vl < best_val - 1e-5:
        best_val = vl; best_epoch = epoch; bad = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  {epoch:>6}  {tr:>10.5f}  {vl:>10.5f}  {best_val:>10.5f}  {'*' if bad==0 else ''}")

    if bad >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
        break

elapsed = time.time() - t0
print(f"\n  Done. Best val {best_val:.5f} at epoch {best_epoch}. Total {elapsed/60:.1f} min")

# restore + save best
model.load_state_dict(best_state)
out_ckpt = OUTDIR / "converged_model_z5.pt"
torch.save(model.state_dict(), out_ckpt)
print(f"  Saved converged model → {out_ckpt}")
print(f"  (baseline_model_z5.pt left untouched)")

# ── loss curve ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, AMBER, RED = "#29ABE2", "#F2A900", "#C94040"
plt.rcParams.update({"axes.facecolor":"white","figure.facecolor":"white",
    "axes.grid":True,"grid.color":"#e8e8e8","axes.spines.top":False,"axes.spines.right":False})
ep = range(1, len(train_hist) + 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, train_hist, color=BLUE, lw=2, label="train")
ax.plot(ep, val_hist, color=AMBER, lw=2, label="val")
ax.axvline(best_epoch, color="#888", ls="--", lw=1.2, label=f"best epoch ({best_epoch})")
ax.scatter([best_epoch], [best_val], color=RED, zorder=5, s=40)
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log scale)")
ax.set_title("Z5 baseline — training to convergence", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
curve_path = ROOT / "reports" / "converged_loss_curve.png"
curve_path.parent.mkdir(exist_ok=True)
fig.savefig(curve_path, dpi=150, bbox_inches="tight")
print(f"  Saved loss curve → {curve_path}")

# save the history so we can quote exact numbers on the slide
np.savez(ROOT/"reports"/"converged_history.npz",
         train=np.array(train_hist), val=np.array(val_hist),
         best_val=best_val, best_epoch=best_epoch)
print(f"\n  KEY NUMBERS for the slide:")
print(f"    best val loss : {best_val:.5f}")
print(f"    best epoch    : {best_epoch}")
print(f"    final train   : {train_hist[-1]:.5f}")
print(f"    epochs run    : {len(train_hist)}")
print(f"    train time    : {elapsed/60:.1f} min")
print("\nDONE. Next: re-run week5_benchmark.py pointed at converged_model_z5.pt to get the new comparison numbers.")