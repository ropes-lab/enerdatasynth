"""
Correct DDPM sampler + verification  (RUN ON YOUR MAC)
======================================================
The benchmark's DDIM sampler was buggy (values exploded to ±3, NaN at high steps).
This is a correct ancestral DDPM sampler, written from scratch, that:
  • recomputes the noise prediction at EVERY timestep (the old one didn't)
  • follows the proper posterior mean + variance
  • runs the full T=1000 reverse chain (matches how the model was trained)

It prints the value range so you can confirm profiles now land in ~[0,1] like
real data, then writes a corrected comparison table.

Run from project root:
    /opt/homebrew/bin/python3.11 sampler_fixed.py

Requires: numpy, torch, matplotlib, statsmodels
"""
import math, time, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"
REPORTS = ROOT/"reports"; REPORTS.mkdir(exist_ok=True)
FIG = REPORTS/"figures"; FIG.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

Xtr  = np.load(OUTDIR/"X_train.npy").astype(float)
Xva  = np.load(OUTDIR/"X_val.npy").astype(float)
Z5va = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
real = Xva.copy(); N_VAL = len(Xva)

# ── model (verbatim from week3) ──
class SinusoidalEmbedding(nn.Module):
    def __init__(s,dim): super().__init__(); s.dim=dim
    def forward(s,t):
        h=s.dim//2; f=torch.exp(-math.log(10000)*torch.arange(h)/h)
        a=t[:,None].float()*f[None]; return torch.cat([a.sin(),a.cos()],-1)
class ResBlock(nn.Module):
    def __init__(s,d): super().__init__(); s.net=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d)); s.act=nn.SiLU()
    def forward(s,x): return s.act(x+s.net(x))
class ConditionalDenoiser(nn.Module):
    def __init__(s,x_dim=24,z_dim=6,hidden=256,n_blocks=4,t_embed_dim=64):
        super().__init__(); s.t_embed=SinusoidalEmbedding(t_embed_dim)
        s.z_proj=nn.Sequential(nn.Linear(z_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        s.input_proj=nn.Linear(x_dim+t_embed_dim+hidden,hidden)
        s.blocks=nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)]); s.out=nn.Linear(hidden,x_dim)
    def forward(s,x,t,z):
        h=torch.cat([x,s.t_embed(t),s.z_proj(z)],-1); h=s.input_proj(h)
        for b in s.blocks: h=b(h)
        return s.out(h)

# ════════════════════════════════════════════════════════════════════════
# CORRECT ancestral DDPM sampler
#   Standard DDPM reverse step (Ho et al. 2020):
#     x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1-abar_t) * eps_theta )
#               + sqrt(beta_t) * noise        (noise=0 at t=0)
#   eps_theta is RECOMPUTED at every t. This is the part the old loop got wrong.
# ════════════════════════════════════════════════════════════════════════
class DDPM:
    def __init__(s, T=1000, beta_start=1e-4, beta_end=0.02):
        s.T = T
        s.betas = torch.linspace(beta_start, beta_end, T)
        s.alphas = 1.0 - s.betas
        s.alpha_bar = torch.cumprod(s.alphas, dim=0)
    @torch.no_grad()
    def sample(s, model, z):
        B = z.shape[0]
        x = torch.randn(B, 24)
        for t in reversed(range(s.T)):
            tv = torch.full((B,), t, dtype=torch.long)
            eps = model(x, tv, z)                      # recompute every step
            alpha_t = s.alphas[t]
            abar_t  = s.alpha_bar[t]
            beta_t  = s.betas[t]
            mean = (x - beta_t / (1 - abar_t).sqrt() * eps) / alpha_t.sqrt()
            if t > 0:
                # posterior variance (sigma_t^2 = beta_t works; standard choice)
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.numpy()

def load(ckpt):
    m = ConditionalDenoiser(z_dim=Z5va.shape[1])
    m.load_state_dict(torch.load(OUTDIR/ckpt, map_location="cpu")); m.eval(); return m

ddpm = DDPM()
print("="*64)
print(f"REAL range: min={real.min():.3f} max={real.max():.3f} mean={real.mean():.3f}")
print("="*64)

samples = {}
for label, ckpt in [("Diffusion (baseline)","baseline_model_z5.pt"),
                    ("Diffusion (converged)","converged_model_z5.pt")]:
    if not (OUTDIR/ckpt).exists():
        print(f"[skip] {ckpt}"); continue
    t0=time.time(); g = ddpm.sample(load(ckpt), torch.tensor(Z5va))
    samples[label]=g
    oob = np.mean((g<-0.05)|(g>1.05))*100
    print(f"{label:24s} range[{g.min():6.2f},{g.max():6.2f}] "
          f"out-of-[0,1]={oob:4.1f}%  mean={g.mean():.3f}  ({time.time()-t0:.0f}s)")

print("\nIf range is now ~[0,1] and out-of-bounds is small (<5%), the sampler is FIXED.")
print("If still exploding, tell Claude — we'll check the model, not the sampler.\n")

# ── classical baselines (same as benchmark) ──
from statsmodels.tsa.arima.model import ARIMA
import warnings; warnings.filterwarnings("ignore")
def fit_sample(order, so, n, steps=24, label=""):
    mp = Xtr.mean(0)
    res = ARIMA(mp, order=order, seasonal_order=so,
                enforce_stationarity=False, enforce_invertibility=False).fit(method_kwargs={"maxiter":200})
    sims=np.zeros((n,steps))
    for i in range(n):
        sims[i]=res.simulate(nsimulations=steps,anchor="start",repetitions=1).reshape(-1)[:steps]
    return np.clip(sims-sims.mean(0,keepdims=True)+mp[None,:], -0.2, 1.2)
print("Fitting classical baselines...")
for name,(o,so) in {"AR":((2,0,0),(0,0,0,0)),"ARMA":((2,0,2),(0,0,0,0)),
                    "ARIMA":((2,1,2),(0,0,0,0)),"SARIMA":((1,0,1),(1,1,1,12))}.items():
    samples[name]=fit_sample(o,so,N_VAL,label=name)
    print(f"  {name} done")

# ── metrics ──
def hourly(r,g):
    return np.abs(r.mean(0)-g.mean(0)).mean()+np.abs((np.percentile(r,75,0)-np.percentile(r,25,0))-(np.percentile(g,75,0)-np.percentile(g,25,0))).mean()
def fdiff(r,g): return abs(np.diff(r,1).std()-np.diff(g,1).std())
def acf_e(r,g,ml=12):
    def acf(M):
        o=[]
        for row in M:
            row=row-row.mean(); d=(row*row).sum()
            if d<=0: continue
            o.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(ml+1)])
        return np.array(o).mean(0)
    return np.abs(acf(r)-acf(g)).mean()
def peak(r,g): return abs(r.max(1).mean()-g.max(1).mean())
def energy(r,g): return abs(r.sum(1).mean()-g.sum(1).mean())
def rmse(r,g): return np.sqrt(((r.mean(0)-g.mean(0))**2).mean())

order=["AR","ARMA","ARIMA","SARIMA","Diffusion (baseline)","Diffusion (converged)"]
order=[o for o in order if o in samples]
cols=["Hourly_dist","First_diff","ACF_err","RMSE","Peak_err","Energy_err"]
print("\n================= CORRECTED COMPARISON TABLE =================")
print(f"{'Model':24s}"+"".join(f"{c:>12s}" for c in cols))
rows=[]
for name in order:
    g=samples[name]
    vals=[hourly(real,g),fdiff(real,g),acf_e(real,g),rmse(real,g),peak(real,g),energy(real,g)]
    rows.append([name]+[round(v,4) for v in vals])
    print(f"{name:24s}"+"".join(f"{v:>12.4f}" for v in vals))

with open(REPORTS/"week5_comparison_corrected.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Model"]+cols); w.writerows(rows)
print(f"\nWrote {REPORTS/'week5_comparison_corrected.csv'}")

# ── refreshed figures (mean overlay + acf overlay, both diffusion rows) ──
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL={"AR":"#9aa7b8","ARMA":"#7B5EA7","ARIMA":"#1d9e75","SARIMA":"#e09320",
     "Diffusion (baseline)":"#9ec5e8","Diffusion (converged)":"#3266ad"}
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
hours=np.arange(24)
fig,ax=plt.subplots(figsize=(9,4.8))
ax.plot(hours,real.mean(0),color="black",lw=3,label="Real",zorder=10)
for m in order: ax.plot(hours,samples[m].mean(0),color=COL.get(m,"#aaa"),lw=2,label=m,alpha=0.9)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalized load [0,1]"); ax.set_xticks(range(0,24,3))
ax.set_title("Mean daily profile — corrected sampler",fontweight="bold"); ax.legend(ncol=2,fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"overlay_mean_fixed.png",dpi=150,bbox_inches="tight"); plt.close(fig)

def acf_set(M,ml=12):
    o=[]
    for r in M:
        r=r-r.mean(); d=(r*r).sum()
        if d<=0: continue
        o.append([(r[:len(r)-k]*r[k:]).sum()/d for k in range(ml+1)])
    return np.array(o).mean(0)
lags=np.arange(13)
fig,ax=plt.subplots(figsize=(9,4.8))
ax.plot(lags,acf_set(real),"o-",color="black",lw=2.5,label="Real",zorder=10)
for m in order: ax.plot(lags,acf_set(samples[m]),"o-",color=COL.get(m,"#aaa"),lw=1.8,label=m,alpha=0.85,markersize=4)
ax.axhline(0,color="black",lw=0.6); ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation — corrected sampler",fontweight="bold"); ax.legend(ncol=2,fontsize=8); ax.set_xticks(lags)
fig.tight_layout(); fig.savefig(FIG/"acf_overlay_fixed.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"Wrote overlay_mean_fixed.png + acf_overlay_fixed.png to {FIG}")
print("\nDONE. Paste the corrected table to Claude.")