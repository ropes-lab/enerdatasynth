"""
Week 5 — PJM daily-load generation benchmark  (RUN ON YOUR MAC)
================================================================
Wired to YOUR real Week-3 pipeline:
  • loads arrays + checkpoint from   week3_outputs/
  • rebuilds your ConditionalDenoiser + DDIM sampler (copied from week3.py)
  • compares AR / ARMA / ARIMA / SARIMA vs your trained diffusion model
  • writes the comparison table + 5 figures

Run from your project root:
    /opt/homebrew/bin/python3.11 week5_benchmark.py

Requires: pip install numpy matplotlib statsmodels torch scikit-learn
No editing needed if your files are in the default week3_outputs/ location.
"""

import warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

# ── Locate week3_outputs (sits next to your raw data) ────────────────────
ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR_CANDIDATES = [
    ROOT / "week3_outputs",
    ROOT / "pjm_diffusion" / "week3_outputs",
    Path("week3_outputs"),
]
OUTDIR = next((p for p in OUTDIR_CANDIDATES if p.exists()), None)
if OUTDIR is None:
    raise FileNotFoundError(
        "Could not find week3_outputs/. Edit OUTDIR_CANDIDATES to point at the "
        "folder containing X_val.npy and baseline_model_z5.pt."
    )
print(f"Using outputs dir: {OUTDIR}")

REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
FIG = REPORTS / "figures"; FIG.mkdir(exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

# ── Load saved arrays (Z5 = 6-dim, the PI baseline your model trained on) ──
Xtr = np.load(OUTDIR / "X_train.npy").astype(float)        # (2785, 24)
Xva = np.load(OUTDIR / "X_val.npy").astype(float)          # (600, 24)
Z5va = np.load(OUTDIR / "Z5_val.npy").astype(np.float32)   # (600, 6)
real = Xva.copy(); N_VAL = len(Xva)
print(f"Loaded: X_train{Xtr.shape}  X_val{Xva.shape}  Z5_val{Z5va.shape}")

# ════════════════════════════════════════════════════════════════════════
# MODEL + SAMPLER — copied verbatim from your week3.py (no retraining)
# ════════════════════════════════════════════════════════════════════════
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim=24, z_dim=6, hidden=256, n_blocks=4, t_embed_dim=64):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(t_embed_dim)
        self.z_proj = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        in_dim = x_dim + t_embed_dim + hidden
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, x_dim)
    def forward(self, x_t, t, z):
        t_emb = self.t_embed(t); z_feat = self.z_proj(z)
        h = torch.cat([x_t, t_emb, z_feat], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks: h = block(h)
        return self.out(h)

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T; self.device = device
        betas = torch.linspace(beta_start, beta_end, T)
        self.betas = betas.to(device)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(device)
    @torch.no_grad()
    def ddim_sample(self, model, z, n_steps=100):
        B = z.shape[0]
        x = torch.randn(B, 24, device=self.device)
        step_idx = torch.linspace(self.T - 1, 0, n_steps).long()
        for i, t_curr in enumerate(step_idx):
            t_vec = torch.full((B,), t_curr.item(), dtype=torch.long, device=self.device)
            eps_pred = model(x, t_vec, z)
            ab_t = self.alpha_bar[t_curr].clamp(min=1e-8)
            x0_hat = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-6)
            x0_hat = x0_hat.clamp(-3, 3)
            if i + 1 < n_steps:
                t_prev = step_idx[i + 1]
                ab_p = self.alpha_bar[t_prev].clamp(min=1e-8)
                x = ab_p.sqrt() * x0_hat + (1 - ab_p).sqrt() * eps_pred
            else:
                x = x0_hat
        return x

def generate_diffusion(Z_val):
    model = ConditionalDenoiser(x_dim=24, z_dim=Z_val.shape[1], hidden=256,
                                n_blocks=4, t_embed_dim=64)
    ckpt = OUTDIR / "baseline_model_z5.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    ddpm = DDPM(T=1000, device="cpu")
    with torch.no_grad():
        g = ddpm.ddim_sample(model, torch.tensor(Z_val, dtype=torch.float32), n_steps=100)
    return g.numpy()

# ════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES
# ════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.arima.model import ARIMA

def fit_and_sample(order, seasonal_order, n_days, steps=24, label=""):
    t0 = time.time()
    mean_profile = Xtr.mean(axis=0)
    model = ARIMA(mean_profile, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(method_kwargs={"maxiter": 200})
    sims = np.zeros((n_days, steps))
    for i in range(n_days):
        s = res.simulate(nsimulations=steps, anchor="start", repetitions=1).reshape(-1)[:steps]
        sims[i] = s
    sims = sims - sims.mean(axis=0, keepdims=True) + mean_profile[None, :]
    rt = time.time() - t0
    print(f"  {label:8s} fit+sample {n_days} days in {rt:.1f}s")
    return np.clip(sims, -0.2, 1.2), rt

print("\nFitting classical baselines...")
specs = {"AR": ((2,0,0),(0,0,0,0)), "ARMA": ((2,0,2),(0,0,0,0)),
         "ARIMA": ((2,1,2),(0,0,0,0)), "SARIMA": ((1,0,1),(1,1,1,12))}
samples, runtimes = {}, {}
for name, (o, so) in specs.items():
    try: s, rt = fit_and_sample(o, so, N_VAL, label=name)
    except Exception as e:
        print(f"  {name} failed ({e}); falling back to AR(2)")
        s, rt = fit_and_sample((2,0,0),(0,0,0,0), N_VAL, label=name)
    samples[name], runtimes[name] = s, rt

print("\nGenerating from your trained diffusion model (baseline_model_z5.pt)...")
t0 = time.time()
diff = generate_diffusion(Z5va)
assert diff.shape == real.shape, f"diffusion {diff.shape} != real {real.shape}"
samples["Diffusion"] = diff; runtimes["Diffusion"] = time.time() - t0
print(f"  Diffusion generate in {runtimes['Diffusion']:.1f}s")

# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════
def hourly_dist_err(r, g):
    dm = np.abs(r.mean(0)-g.mean(0)).mean()
    di = np.abs((np.percentile(r,75,0)-np.percentile(r,25,0)) -
                (np.percentile(g,75,0)-np.percentile(g,25,0))).mean()
    return dm + di
def firstdiff_err(r,g): return abs(np.diff(r,axis=1).std()-np.diff(g,axis=1).std())
def acf_err(r,g,maxlag=12):
    def acf(M):
        out=[]
        for row in M:
            row=row-row.mean(); d=(row*row).sum()
            if d<=0: continue
            out.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(maxlag+1)])
        return np.array(out).mean(0)
    return np.abs(acf(r)-acf(g)).mean()
def peak_err(r,g): return abs(r.max(1).mean()-g.max(1).mean())
def energy_err(r,g): return abs(r.sum(1).mean()-g.sum(1).mean())
def rmse_profile(r,g): return np.sqrt(((r.mean(0)-g.mean(0))**2).mean())

metrics={}
for name,g in samples.items():
    metrics[name]={"Hourly_dist":hourly_dist_err(real,g),"First_diff":firstdiff_err(real,g),
        "ACF_err":acf_err(real,g),"RMSE":rmse_profile(real,g),"Peak_err":peak_err(real,g),
        "Energy_err":energy_err(real,g),"Runtime_s":runtimes[name]}

order=["AR","ARMA","ARIMA","SARIMA","Diffusion"]
cols=["Hourly_dist","First_diff","ACF_err","RMSE","Peak_err","Energy_err","Runtime_s"]
print("\n================= COMPARISON TABLE =================")
print(f"{'Model':10s}"+"".join(f"{c:>12s}" for c in cols))
for name in order:
    m=metrics[name]; print(f"{name:10s}"+"".join(f"{m[c]:>12.4f}" for c in cols))

import csv
with open(REPORTS/"week5_comparison_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Model"]+cols)
    for name in order: w.writerow([name]+[round(metrics[name][c],4) for c in cols])
print(f"\nWrote {REPORTS/'week5_comparison_table.csv'}")

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
COL={"AR":"#9aa7b8","ARMA":"#7B5EA7","ARIMA":"#1d9e75","SARIMA":"#e09320","Diffusion":"#3266ad"}
BLUE,AMBER="#3266ad","#e09320"
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6",
    "axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
hours=np.arange(24)

fig,ax=plt.subplots(figsize=(9,4.8))
ax.plot(hours,real.mean(0),color="black",lw=3,label="Real",zorder=10)
for m in order: ax.plot(hours,samples[m].mean(0),color=COL[m],lw=2,label=m,alpha=0.9)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalized load [0,1]"); ax.set_xticks(range(0,24,3))
ax.set_title("Mean daily profile — real vs all models",fontweight="bold"); ax.legend(ncol=2,fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"overlay_mean.png",dpi=150,bbox_inches="tight"); plt.close(fig)

def acf_set(M,maxlag=12):
    out=[]
    for r in M:
        r=r-r.mean(); d=(r*r).sum()
        if d<=0: continue
        out.append([(r[:len(r)-k]*r[k:]).sum()/d for k in range(maxlag+1)])
    return np.array(out).mean(0)
lags=np.arange(13)
fig,ax=plt.subplots(figsize=(9,4.8))
ax.plot(lags,acf_set(real),"o-",color="black",lw=2.5,label="Real",zorder=10)
for m in order: ax.plot(lags,acf_set(samples[m]),"o-",color=COL[m],lw=1.8,label=m,alpha=0.85,markersize=4)
ax.axhline(0,color="black",lw=0.6); ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation overlay",fontweight="bold"); ax.legend(ncol=2,fontsize=9); ax.set_xticks(lags)
fig.tight_layout(); fig.savefig(FIG/"acf_overlay.png",dpi=150,bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8))
def ldc(M): v=np.sort(np.clip(M,-0.05,1.05).ravel())[::-1]; return np.linspace(0,100,len(v)),v
x,y=ldc(real); ax.plot(x,y,color="black",lw=3,label="Real",zorder=10)
for m in order: x,y=ldc(samples[m]); ax.plot(x,y,color=COL[m],lw=2,label=m,alpha=0.9)
ax.set_ylim(-0.1,1.15); ax.set_xlabel("% of hours (sorted descending)"); ax.set_ylabel("Normalized load")
ax.set_title("Load duration curve",fontweight="bold"); ax.legend(ncol=2,fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"load_duration.png",dpi=150,bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8))
bins=np.linspace(-0.4,0.4,80)
ax.hist(np.diff(real,axis=1).ravel(),bins=bins,density=True,color="black",alpha=0.45,label="Real")
ax.hist(np.diff(samples["Diffusion"],axis=1).ravel(),bins=bins,density=True,color=BLUE,alpha=0.5,label="Diffusion")
ax.hist(np.diff(samples["SARIMA"],axis=1).ravel(),bins=bins,density=True,color=AMBER,alpha=0.45,label="SARIMA")
ax.set_ylim(0,18); ax.set_xlabel("Hour-over-hour Δ (normalized)"); ax.set_ylabel("Density")
ax.set_title("First-difference distribution",fontweight="bold"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"firstdiff_overlay.png",dpi=150,bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(11,4.4))
g=samples["Diffusion"]; pr=hours*2.0; pg=hours*2.0+0.8
b1=ax.boxplot([real[:,h] for h in hours],positions=pr,widths=0.7,patch_artist=True,
    medianprops={"color":"black"},flierprops={"marker":".","markersize":1,"alpha":0.2})
b2=ax.boxplot([g[:,h] for h in hours],positions=pg,widths=0.7,patch_artist=True,
    medianprops={"color":"black"},flierprops={"marker":".","markersize":1,"alpha":0.2})
for b in b1["boxes"]: b.set_facecolor(BLUE); b.set_alpha(0.6)
for b in b2["boxes"]: b.set_facecolor(AMBER); b.set_alpha(0.6)
ax.set_xticks(hours*2.0+0.4); ax.set_xticklabels([f"h{h:02d}" for h in hours],rotation=45,fontsize=8)
ax.set_ylabel("Normalized load [0,1]")
ax.set_title("Hourly distribution — Real (blue) vs Diffusion (orange)",fontweight="bold")
ax.legend(handles=[Patch(color=BLUE,alpha=0.6,label="Real"),Patch(color=AMBER,alpha=0.6,label="Diffusion")],fontsize=10)
fig.tight_layout(); fig.savefig(FIG/"box_diffusion.png",dpi=150,bbox_inches="tight"); plt.close(fig)

json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in metrics.items()},
          open(REPORTS/"week5_metrics.json","w"), indent=2)
print(f"Wrote 5 figures to {FIG}")
print("\nDONE. Numbers are in week5_comparison_table.csv — drop them into slide 4.")