"""
Calibrate the transformer baseline  (RUN ON YOUR MAC)
=====================================================
The transformer reproduces daily SHAPE better than the MLP but sits slightly
low at every hour (~-0.6%). Summing 24 hours multiplies that offset ~24x, which
is why daily-energy error regressed while every shape metric improved.

This script fixes it the simple, defensible way:
  1. Measure the per-hour bias on the TRAINING set  (never on validation --
     that would be fitting to the test data)
  2. Save it as a calibration file next to the checkpoint
  3. Apply it to validation generations and re-score all 7 metrics
  4. Report before/after so you can show the fix worked and cost nothing

Two correction modes are compared:
  scalar   one number added to every hour   (simplest; matches the diagnosis)
  perhour  a 24-value vector, one per hour  (tighter, slightly more to justify)

Run from project root:
    /opt/homebrew/bin/python3.11 calibrate.py

Outputs -> reports/week7/
    calibration.json, calibration_report.md, calibration_before_after.png

Requires: numpy, matplotlib, torch
"""
import math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"
REPORTS = ROOT/"reports"/"week7"; REPORTS.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

Xtr = np.load(OUTDIR/"X_train.npy").astype(np.float32)
Xva = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Ztr = np.load(OUTDIR/"Z5_train.npy").astype(np.float32)
Zva = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
H = 24; Z_DIM = Zva.shape[1]
CKPT = "transformer_baseline_z5.pt"

# ── model + sampler (same as week7) ─────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(s, dim): super().__init__(); s.dim = dim
    def forward(s, t):
        h = s.dim//2
        f = torch.exp(-math.log(10000)*torch.arange(h, device=t.device)/h)
        a = t[:, None].float()*f[None]
        return torch.cat([a.sin(), a.cos()], -1)

class TransformerDenoiser(nn.Module):
    def __init__(s, x_dim=H, z_dim=Z_DIM, d=64, n_heads=4, n_layers=2, t_dim=64):
        super().__init__()
        s.t_embed = SinusoidalEmbedding(t_dim)
        s.cond = nn.Sequential(nn.Linear(z_dim+t_dim, d), nn.SiLU(), nn.Linear(d, d))
        s.val = nn.Linear(1, d)
        s.pos = nn.Parameter(torch.randn(1, H, d)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4,
                                           batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        s.out = nn.Linear(d, 1)
    def forward(s, x, t, z):
        c = s.cond(torch.cat([z, s.t_embed(t)], -1))[:, None, :]
        h = s.val(x[:, :, None]) + s.pos + c
        return s.out(s.enc(h))[:, :, 0]

class DDPM:
    def __init__(s, T=1000, b0=1e-4, b1=0.02):
        s.T = T; s.betas = torch.linspace(b0, b1, T)
        s.alphas = 1.0-s.betas; s.alpha_bar = torch.cumprod(s.alphas, 0)
    @torch.no_grad()
    def sample(s, model, z):
        B = z.shape[0]; x = torch.randn(B, H)
        for t in reversed(range(s.T)):
            tv = torch.full((B,), t, dtype=torch.long)
            eps = model(x, tv, z)
            a, ab, b = s.alphas[t], s.alpha_bar[t], s.betas[t]
            mean = (x - b/(1-ab).sqrt()*eps)/a.sqrt()
            x = mean + (b.sqrt()*torch.randn_like(x) if t > 0 else 0)
        return x.numpy()

# ── metrics ──
def _acf(M, ml=12):
    o=[]
    for row in M:
        row=row-row.mean(); d=(row*row).sum()
        if d<=0: continue
        o.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(ml+1)])
    return np.array(o).mean(0) if o else np.zeros(ml+1)
def metrics(r,g):
    return dict(
        Hourly=float(np.abs(r.mean(0)-g.mean(0)).mean()+np.abs((np.percentile(r,75,0)-np.percentile(r,25,0))-(np.percentile(g,75,0)-np.percentile(g,25,0))).mean()),
        FirstDiff=float(abs(np.diff(r,1).std()-np.diff(g,1).std())),
        ACF=float(np.abs(_acf(r)-_acf(g)).mean()),
        RMSE=float(np.sqrt(((r.mean(0)-g.mean(0))**2).mean())),
        Peak=float(abs(r.max(1).mean()-g.max(1).mean())),
        Energy=float(abs(r.sum(1).mean()-g.sum(1).mean())),
        Ramp=float(abs(np.abs(np.diff(r,1)).mean()-np.abs(np.diff(g,1)).mean())))
MK = ["Hourly","FirstDiff","ACF","RMSE","Peak","Energy","Ramp"]

model = TransformerDenoiser()
model.load_state_dict(torch.load(OUTDIR/CKPT, map_location="cpu")); model.eval()
ddpm = DDPM()

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — measure the bias on TRAINING data (not validation)
# ════════════════════════════════════════════════════════════════════════
print("Measuring bias on the TRAINING set (so validation stays untouched)...")
n_fit = min(1200, len(Xtr))              # subset keeps runtime sane
idx = np.random.choice(len(Xtr), n_fit, replace=False)
g_tr = ddpm.sample(model, torch.tensor(Ztr[idx]))
bias_perhour = (Xtr[idx].mean(0) - g_tr.mean(0)).astype(float)   # add this to fix
bias_scalar  = float(Xtr[idx].mean() - g_tr.mean())

print(f"  scalar offset to add : {bias_scalar:+.5f}")
print(f"  per-hour offsets     : min {bias_perhour.min():+.4f}  max {bias_perhour.max():+.4f}  std {bias_perhour.std():.4f}")

json.dump({"mode_recommended":"scalar",
           "scalar": bias_scalar,
           "per_hour": bias_perhour.tolist(),
           "fitted_on":"training set", "n_days_used": int(n_fit),
           "checkpoint": CKPT},
          open(OUTDIR/"calibration.json","w"), indent=2)
print(f"  saved -> {OUTDIR/'calibration.json'}")

# ════════════════════════════════════════════════════════════════════════
# STEP 2 — apply to VALIDATION and re-score
# ════════════════════════════════════════════════════════════════════════
print("\nGenerating validation set and scoring before/after...")
g_va = ddpm.sample(model, torch.tensor(Zva))
variants = {
    "uncalibrated": g_va,
    "scalar":       g_va + bias_scalar,
    "per-hour":     g_va + bias_perhour[None, :],
}
rows = {k: metrics(Xva, v) for k, v in variants.items()}

print("\n" + "="*86)
print(f"{'variant':16s}" + "".join(f"{k:>10s}" for k in MK))
for k, m in rows.items():
    print(f"{k:16s}" + "".join(f"{m[c]:>10.4f}" for c in MK))
print("="*86)

base = rows["uncalibrated"]; sc = rows["scalar"]
print(f"\nEnergy error : {base['Energy']:.4f}  ->  {sc['Energy']:.4f}   "
      f"({100*(1-sc['Energy']/base['Energy']) if base['Energy'] else 0:.1f}% reduction)")
print("Shape metrics should be essentially unchanged (a constant shift cannot alter")
print("first-difference, ramp or ACF; only level-sensitive metrics move).")
oob = np.mean((variants['scalar']<-0.05)|(variants['scalar']>1.05))*100
print(f"Out-of-range after calibration: {oob:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# STEP 3 — figure + report
# ════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,
                     "grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
hours = np.arange(H)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
ax[0].plot(hours, Xva.mean(0), color="black", lw=2.5, label="Real")
ax[0].plot(hours, variants["uncalibrated"].mean(0), color="#c94040", lw=2, ls="--", label="Before")
ax[0].plot(hours, variants["scalar"].mean(0), color="#3266ad", lw=2, label="After")
ax[0].set_title("Mean profile", fontweight="bold"); ax[0].set_xlabel("Hour")
ax[0].set_ylabel("Normalized load"); ax[0].set_xticks(range(0,24,3)); ax[0].legend(fontsize=9)
names = ["uncalibrated","scalar","per-hour"]
ax[1].bar(np.arange(3), [rows[n]["Energy"] for n in names], color=["#c94040","#3266ad","#1d9e75"])
ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(names, fontsize=9)
ax[1].set_title("Daily energy error", fontweight="bold"); ax[1].set_ylabel("Energy error")
fig.suptitle("Calibration: removing the systematic level offset", fontweight="bold")
fig.tight_layout(); fig.savefig(REPORTS/"calibration_before_after.png", dpi=150, bbox_inches="tight"); plt.close(fig)

lines = ["# Calibration — removing the level offset\n",
         f"Bias measured on {n_fit} training days (validation never used for fitting).\n",
         f"- Scalar offset applied: **{bias_scalar:+.5f}** per hour",
         f"- Per-hour offsets: min {bias_perhour.min():+.4f}, max {bias_perhour.max():+.4f}, std {bias_perhour.std():.4f}\n",
         "| Variant | " + " | ".join(MK) + " |", "|" + "---|"*(len(MK)+1)]
for k in ["uncalibrated","scalar","per-hour"]:
    lines.append(f"| {k} | " + " | ".join(f"{rows[k][c]:.4f}" for c in MK) + " |")
lines.append(f"\nEnergy error {base['Energy']:.4f} -> {sc['Energy']:.4f} with a single scalar correction.")
lines.append("\nA constant shift cannot change first-difference, ramp or autocorrelation, so the "
             "shape metrics are unaffected. The correction is stored in `calibration.json` next to "
             "the checkpoint and should be applied at generation time.")
(REPORTS/"calibration_report.md").write_text("\n".join(lines))
print(f"\nWrote calibration_report.md + calibration_before_after.png")
print("DONE.")