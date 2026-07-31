"""
Diagnose the converged diffusion model  (RUN ON YOUR MAC)
=========================================================
Answers one question: is the converged model's worse first-diff/RMSE a SAMPLER
artifact (overshooting the [0,1] range) or a REAL property of the model?

It checks:
  1. Value range of generated profiles vs real (overshoot test)
  2. Whether more DDIM steps (100 vs 250 vs 1000) change the result
  3. Per-metric comparison at each step count

Run from project root:
    /opt/homebrew/bin/python3.11 diagnose_converged.py

Requires: numpy, torch, matplotlib
"""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"
torch.manual_seed(0); np.random.seed(0)

X_val = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Z_val = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
real = X_val.copy()

# ── model + DDPM (verbatim from week3) ──
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
class DDPM:
    def __init__(s,T=1000,beta_start=1e-4,beta_end=0.02,device="cpu"):
        s.T=T; s.device=device
        betas=torch.linspace(beta_start,beta_end,T)
        s.betas=betas; s.alpha_bar=torch.cumprod(1.0-betas,0)
    @torch.no_grad()
    def ddim_sample(s, model, z, n_steps=100, clamp_x0=True):
        B=z.shape[0]; x=torch.randn(B,24)
        idx=torch.linspace(s.T-1,0,n_steps).long()
        for i,tc in enumerate(idx):
            tv=torch.full((B,),tc.item(),dtype=torch.long)
            eps=model(x,tv,z)
            ab=s.alpha_bar[tc].clamp(min=1e-8)
            x0=(x-(1-ab).sqrt()*eps)/ab.sqrt().clamp(min=1e-6)
            if clamp_x0: x0=x0.clamp(-3,3)
            if i+1<n_steps:
                ap=s.alpha_bar[idx[i+1]].clamp(min=1e-8)
                x=ap.sqrt()*x0+(1-ap).sqrt()*eps
            else:
                x=x0
        return x.numpy()

def load(ckpt):
    m=ConditionalDenoiser(z_dim=Z_val.shape[1])
    m.load_state_dict(torch.load(OUTDIR/ckpt,map_location="cpu")); m.eval(); return m

def metrics(r,g):
    fd=abs(np.diff(r,1).std()-np.diff(g,1).std())
    rmse=np.sqrt(((r.mean(0)-g.mean(0))**2).mean())
    def acf(M,ml=12):
        o=[]
        for row in M:
            row=row-row.mean(); d=(row*row).sum()
            if d<=0: continue
            o.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(ml+1)])
        return np.array(o).mean(0)
    af=np.abs(acf(r)-acf(g)).mean()
    return fd,rmse,af

print("="*64)
print(f"REAL data range: min={real.min():.3f}  max={real.max():.3f}  "
      f"mean={real.mean():.3f}  std={real.std():.3f}")
print("="*64)

ddpm=DDPM()
for ckpt in ["baseline_model_z5.pt","converged_model_z5.pt"]:
    if not (OUTDIR/ckpt).exists():
        print(f"\n[skip] {ckpt} not found"); continue
    m=load(ckpt)
    print(f"\n### {ckpt}")
    for steps in [100, 250, 1000]:
        g=ddpm.ddim_sample(m,torch.tensor(Z_val),n_steps=steps)
        fd,rmse,af=metrics(real,g)
        frac_oob=np.mean((g<-0.05)|(g>1.05))
        print(f"  steps={steps:>4}  range[{g.min():>6.2f},{g.max():>6.2f}]  "
              f"out-of-[0,1]={frac_oob*100:>5.1f}%  |  firstdiff={fd:.3f}  rmse={rmse:.3f}  acf={af:.3f}")

print("\n" + "="*64)
print("HOW TO READ THIS:")
print("• If converged range is way past [0,1] (e.g. -3 to 4) and out-of-bounds %")
print("  is high → SAMPLER ARTIFACT. The model is fine; sampling overshoots.")
print("• If more steps (250/1000) sharply reduce firstdiff/rmse → SAMPLER issue,")
print("  fixable by sampling differently. Rerun benchmark with more steps.")
print("• If range stays ~[0,1] and metrics are stable across step counts →")
print("  REAL result. Converged model genuinely trades shape for correlation.")
print("="*64)