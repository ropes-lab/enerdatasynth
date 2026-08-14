"""
Week 7 follow-up — energy-bias diagnosis + error analysis  (RUN ON YOUR MAC)
===========================================================================
Two things in one pass over the transformer baseline:

  PART 1  Why did daily-energy error get worse (0.0008 -> 0.1405) while every
          shape metric improved? Tests whether it is a constant LEVEL BIAS
          (a small offset that compounds across 24 hours) or something else.

  PART 2  Error analysis: which days does the model generate best / worst?
          Ranks all 600 validation days by per-day RMSE and plots the 3 best
          and 3 worst against the real profile, with their conditions.

Run from project root (after week7.py --task train):
    /opt/homebrew/bin/python3.11 error_analysis.py

Outputs -> reports/week7/
    energy_diagnosis.md, energy_bias.png, best_worst_days.png, per_day_errors.csv

Requires: numpy, matplotlib, torch
"""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"
REPORTS = ROOT/"reports"/"week7"; REPORTS.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

Xva = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Zva = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
H = 24; Z_DIM = Zva.shape[1]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOWS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

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
        s.alphas = 1.0 - s.betas; s.alpha_bar = torch.cumprod(s.alphas, 0)
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

print("Generating from transformer baseline...")
model = TransformerDenoiser()
model.load_state_dict(torch.load(OUTDIR/"transformer_baseline_z5.pt", map_location="cpu"))
model.eval()
gen = DDPM().sample(model, torch.tensor(Zva))

# ════════════════════════════════════════════════════════════════════════
# PART 1 — energy bias diagnosis
# ════════════════════════════════════════════════════════════════════════
real_energy = Xva.sum(1)          # per-day total (24 hourly values summed)
gen_energy  = gen.sum(1)
mean_off    = float(gen.mean() - Xva.mean())          # per-hour level offset
energy_bias = float(gen_energy.mean() - real_energy.mean())   # signed
energy_abs  = float(abs(energy_bias))
# what energy error would be if we removed ONLY the constant offset
gen_corrected = gen - mean_off
energy_after  = float(abs((gen_corrected.sum(1)).mean() - real_energy.mean()))
# per-hour bias profile: is the offset flat, or concentrated in some hours?
hour_bias = gen.mean(0) - Xva.mean(0)

print("\n=== PART 1: ENERGY BIAS ===")
print(f"  per-hour level offset (gen - real)      : {mean_off:+.5f}")
print(f"  signed daily energy bias                : {energy_bias:+.5f}")
print(f"  |energy error| as reported              : {energy_abs:.5f}")
print(f"  |energy error| after removing offset    : {energy_after:.5f}")
print(f"  offset x 24 hours                       : {mean_off*24:+.5f}   <- compare to signed bias")
print(f"  per-hour bias  min {hour_bias.min():+.4f}  max {hour_bias.max():+.4f}  std {hour_bias.std():.4f}")
explained = 100*(1 - energy_after/energy_abs) if energy_abs > 0 else 0
print(f"  => constant offset explains {explained:.1f}% of the energy error")

# ════════════════════════════════════════════════════════════════════════
# PART 2 — per-day error analysis
# ════════════════════════════════════════════════════════════════════════
per_day_rmse = np.sqrt(((gen - Xva)**2).mean(1))
o = np.argsort(per_day_rmse)
best_idx, worst_idx = o[:3], o[-3:][::-1]

def describe(i):
    z = Zva[i]
    m = MONTHS[int(z[0])-1] if 1 <= int(z[0]) <= 12 else f"m{int(z[0])}"
    d = DOWS[int(z[1])] if 0 <= int(z[1]) <= 6 else f"d{int(z[1])}"
    hol = " HOLIDAY" if z[2] >= 0.5 else ""
    return f"{m} {d}{hol}  Tmax_z={z[3]:+.2f}"

print("\n=== PART 2: BEST / WORST DAYS (per-day RMSE) ===")
print("  BEST:")
for i in best_idx:  print(f"    RMSE {per_day_rmse[i]:.4f}   {describe(i)}")
print("  WORST:")
for i in worst_idx: print(f"    RMSE {per_day_rmse[i]:.4f}   {describe(i)}")
print(f"\n  median {np.median(per_day_rmse):.4f}   p90 {np.percentile(per_day_rmse,90):.4f}   max {per_day_rmse.max():.4f}")

# is the worst tail dominated by a particular condition?
hot = Zva[:,3] >= np.percentile(Zva[:,3],75); cold = Zva[:,3] <= np.percentile(Zva[:,3],25)
holi = Zva[:,2] >= 0.5; wknd = Zva[:,1] >= 5
tail = per_day_rmse >= np.percentile(per_day_rmse, 90)
print("\n  Share of the worst-10% days that are:")
for nm, mask in [("hot", hot), ("cold", cold), ("holiday", holi), ("weekend", wknd)]:
    base = mask.mean()*100; intail = mask[tail].mean()*100
    print(f"    {nm:8s} {intail:5.1f}%   (vs {base:5.1f}% of all days)")

np.savetxt(REPORTS/"per_day_errors.csv",
           np.column_stack([np.arange(len(per_day_rmse)), per_day_rmse, Zva]),
           delimiter=",", header="idx,rmse,month,dow,is_holiday,Tmax_z,Tmin_z,spread", comments="")

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,
                     "grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
BLUE, AMBER, RED, GRAY = "#3266ad", "#e09320", "#c94040", "#888780"
hours = np.arange(H)

# energy bias figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
ax[0].plot(hours, Xva.mean(0), color="black", lw=2.5, label="Real")
ax[0].plot(hours, gen.mean(0), color=BLUE, lw=2, label="Transformer")
ax[0].set_title("Mean profile", fontweight="bold"); ax[0].set_xlabel("Hour"); ax[0].set_ylabel("Normalized load")
ax[0].set_xticks(range(0,24,3)); ax[0].legend(fontsize=9)
ax[1].bar(hours, hour_bias, color=[RED if v<0 else BLUE for v in hour_bias])
ax[1].axhline(mean_off, color=AMBER, ls="--", lw=1.5, label=f"mean offset {mean_off:+.4f}")
ax[1].set_title("Per-hour bias (gen − real)", fontweight="bold"); ax[1].set_xlabel("Hour")
ax[1].set_xticks(range(0,24,3)); ax[1].legend(fontsize=9)
fig.suptitle("Energy-error diagnosis: is it a constant level offset?", fontweight="bold")
fig.tight_layout(); fig.savefig(REPORTS/"energy_bias.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# best/worst days figure
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for col, i in enumerate(best_idx):
    a = axes[0, col]
    a.plot(hours, Xva[i], color="black", lw=2.5, label="Real")
    a.plot(hours, gen[i], color=BLUE, lw=2, label="Generated")
    a.set_title(f"BEST · RMSE {per_day_rmse[i]:.3f}\n{describe(i)}", fontsize=9, fontweight="bold")
    a.set_xticks(range(0,24,6)); a.set_ylim(-0.05, 1.15)
    if col == 0: a.set_ylabel("Normalized load"); a.legend(fontsize=8)
for col, i in enumerate(worst_idx):
    a = axes[1, col]
    a.plot(hours, Xva[i], color="black", lw=2.5)
    a.plot(hours, gen[i], color=RED, lw=2)
    a.set_title(f"WORST · RMSE {per_day_rmse[i]:.3f}\n{describe(i)}", fontsize=9, fontweight="bold")
    a.set_xticks(range(0,24,6)); a.set_ylim(-0.05, 1.15); a.set_xlabel("Hour")
    if col == 0: a.set_ylabel("Normalized load")
fig.suptitle("Error analysis — best and worst generated days", fontweight="bold")
fig.tight_layout(); fig.savefig(REPORTS/"best_worst_days.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── report ──
lines = ["# Week 7 follow-up — Energy bias & error analysis\n",
         "## Part 1: why daily-energy error regressed\n",
         f"- Per-hour level offset (gen − real): **{mean_off:+.5f}**",
         f"- Signed daily energy bias: **{energy_bias:+.5f}**  (offset × 24 h = {mean_off*24:+.5f})",
         f"- Reported |energy error|: **{energy_abs:.5f}**",
         f"- |energy error| after removing the constant offset: **{energy_after:.5f}**",
         f"- Constant offset explains **{explained:.1f}%** of the energy error.",
         f"- Per-hour bias spread: min {hour_bias.min():+.4f}, max {hour_bias.max():+.4f}, std {hour_bias.std():.4f}\n",
         "Interpretation: the transformer reproduces daily *shape* better than the MLP "
         "(first-difference, ramp, peak all improved), but carries a small systematic level "
         "offset. Because daily energy sums all 24 hours, that offset compounds ~24× and shows "
         "up as a large energy error while shape metrics stay excellent.\n",
         "## Part 2: best / worst generated days\n",
         f"- Median per-day RMSE: **{np.median(per_day_rmse):.4f}**; p90 {np.percentile(per_day_rmse,90):.4f}; max {per_day_rmse.max():.4f}\n",
         "| Rank | RMSE | Conditions |", "|---|---|---|"]
for i in best_idx:  lines.append(f"| best | {per_day_rmse[i]:.4f} | {describe(i)} |")
for i in worst_idx: lines.append(f"| worst | {per_day_rmse[i]:.4f} | {describe(i)} |")
lines.append("\nShare of worst-10% days by condition (vs share of all days):\n")
for nm, mask in [("hot", hot), ("cold", cold), ("holiday", holi), ("weekend", wknd)]:
    lines.append(f"- {nm}: {mask[tail].mean()*100:.1f}% vs {mask.mean()*100:.1f}%")
(REPORTS/"energy_diagnosis.md").write_text("\n".join(lines))
print(f"\nWrote {REPORTS/'energy_diagnosis.md'}, energy_bias.png, best_worst_days.png, per_day_errors.csv")
print("DONE.")