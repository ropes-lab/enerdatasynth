"""
Week 6 evaluation pipeline  (RUN ON YOUR MAC)
=============================================
One script, two tasks, selected by --task:

  conditional : slice the validation set by temperature / calendar / season and
                evaluate BOTH the baseline and converged diffusion models on each
                slice. Answers "where does the model do well / struggle?"

  noise       : train 3 diffusion models that differ ONLY in the beta schedule
                (linear / cosine / quadratic), everything else identical, and
                compare them. Answers "does the noise schedule matter?"

Uses the CORRECTED ancestral DDPM sampler (the earlier DDIM one was buggy).

Usage (from project root):
    /opt/homebrew/bin/python3.11 evaluate_model.py --task conditional
    /opt/homebrew/bin/python3.11 evaluate_model.py --task noise --epochs 300

Outputs go to  reports/week6/<task>/  :
    metrics.csv, summary_report.md, and per-slice / per-schedule PNGs.

Requires: numpy, matplotlib, torch
"""
import argparse, math, time, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found — edit OUTDIR."
torch.manual_seed(0); np.random.seed(0)

# ── data ─────────────────────────────────────────────────────────────────
Xtr  = np.load(OUTDIR/"X_train.npy").astype(np.float32)
Xva  = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Ztr  = np.load(OUTDIR/"Z5_train.npy").astype(np.float32)
Zva  = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
# Z5 layout: [month(1-12), dow(0-6), is_holiday(0/1), Tmax_z, Tmin_z, spread_norm]
COL_MONTH, COL_DOW, COL_HOL, COL_TMAX, COL_TMIN, COL_SPREAD = range(6)

# ════════════════════════════════════════════════════════════════════════
# MODEL + CORRECT SAMPLER (verbatim arch from week3; fixed ancestral sampler)
# ════════════════════════════════════════════════════════════════════════
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

def make_betas(schedule, T=1000, beta_start=1e-4, beta_end=0.02):
    if schedule=="linear":
        return torch.linspace(beta_start, beta_end, T)
    if schedule=="quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, T)**2
    if schedule=="cosine":
        s=0.008; steps=torch.arange(T+1)/T
        ab=torch.cos(((steps+s)/(1+s))*math.pi/2)**2; ab=ab/ab[0]
        betas=1-(ab[1:]/ab[:-1]); return betas.clamp(1e-4,0.999)
    raise ValueError(schedule)

class DDPM:
    def __init__(s, schedule="linear", T=1000):
        s.T=T; s.betas=make_betas(schedule,T)
        s.alphas=1.0-s.betas; s.alpha_bar=torch.cumprod(s.alphas,0)
    def training_loss(s, model, x0, z):
        B=x0.shape[0]; t=torch.randint(0,s.T,(B,))
        noise=torch.randn_like(x0); ab=s.alpha_bar[t][:,None]
        xt=ab.sqrt()*x0+(1-ab).sqrt()*noise
        return nn.functional.mse_loss(model(xt,t,z), noise)
    @torch.no_grad()
    def sample(s, model, z):
        B=z.shape[0]; x=torch.randn(B,24)
        for t in reversed(range(s.T)):
            tv=torch.full((B,),t,dtype=torch.long)
            eps=model(x,tv,z)
            a=s.alphas[t]; ab=s.alpha_bar[t]; b=s.betas[t]
            mean=(x-b/(1-ab).sqrt()*eps)/a.sqrt()
            x = mean + (b.sqrt()*torch.randn_like(x) if t>0 else 0)
        return x.numpy()

def load_model(ckpt):
    m=ConditionalDenoiser(z_dim=Zva.shape[1]); m.load_state_dict(torch.load(OUTDIR/ckpt,map_location="cpu")); m.eval(); return m

# ════════════════════════════════════════════════════════════════════════
# METRICS (same 7 as the benchmark)
# ════════════════════════════════════════════════════════════════════════
def _acf(M,ml=12):
    o=[]
    for row in M:
        row=row-row.mean(); d=(row*row).sum()
        if d<=0: continue
        o.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(ml+1)])
    return np.array(o).mean(0) if o else np.zeros(ml+1)
def metrics(r,g):
    hourly=np.abs(r.mean(0)-g.mean(0)).mean()+np.abs((np.percentile(r,75,0)-np.percentile(r,25,0))-(np.percentile(g,75,0)-np.percentile(g,25,0))).mean()
    fd=abs(np.diff(r,1).std()-np.diff(g,1).std())
    af=np.abs(_acf(r)-_acf(g)).mean()
    rmse=np.sqrt(((r.mean(0)-g.mean(0))**2).mean())
    peak=abs(r.max(1).mean()-g.max(1).mean())
    energy=abs(r.sum(1).mean()-g.sum(1).mean())
    ramp=abs(np.abs(np.diff(r,1)).mean()-np.abs(np.diff(g,1)).mean())
    return dict(Hourly=hourly,FirstDiff=fd,ACF=af,RMSE=rmse,Peak=peak,Energy=energy,Ramp=ramp)
METRIC_KEYS=["Hourly","FirstDiff","ACF","RMSE","Peak","Energy","Ramp"]

# ════════════════════════════════════════════════════════════════════════
# SLICE DEFINITIONS  (indices into the validation set)
# ════════════════════════════════════════════════════════════════════════
def make_slices():
    tmax=Zva[:,COL_TMAX]; month=Zva[:,COL_MONTH]; dow=Zva[:,COL_DOW]; hol=Zva[:,COL_HOL]
    q25,q75=np.percentile(tmax,[25,75])
    slices={}
    # temperature terciles (by z-scored Tmax)
    slices["temp"]={
        "Cold (bottom 25%)": tmax<=q25,
        "Mild (middle 50%)": (tmax>q25)&(tmax<q75),
        "Hot (top 25%)":     tmax>=q75,
    }
    # calendar
    slices["calendar"]={
        "Weekday": (dow<=4)&(hol<0.5),
        "Weekend": dow>=5,
        "Holiday": hol>=0.5,
    }
    # season by month
    def mon(ms): return np.isin(month.astype(int), ms)
    slices["season"]={
        "Winter (DJF)": mon([12,1,2]),
        "Spring (MAM)": mon([3,4,5]),
        "Summer (JJA)": mon([6,7,8]),
        "Fall (SON)":   mon([9,10,11]),
    }
    return slices

# ════════════════════════════════════════════════════════════════════════
# TASK: conditional subset evaluation
# ════════════════════════════════════════════════════════════════════════
def task_conditional(args):
    outdir=ROOT/"reports"/"week6"/"conditional"; (outdir).mkdir(parents=True,exist_ok=True)
    ddpm=DDPM("linear")
    models={}
    for label,ckpt in [("baseline","baseline_model_z5.pt"),("converged","converged_model_z5.pt")]:
        if (OUTDIR/ckpt).exists(): models[label]=load_model(ckpt)
        else: print(f"[skip] {ckpt} missing")
    assert models, "no checkpoints found"

    # generate once per model on the FULL val set, then slice the arrays (fast + consistent)
    gen={}
    for label,m in models.items():
        print(f"Generating full val set from {label} ...")
        gen[label]=ddpm.sample(m, torch.tensor(Zva))

    slices=make_slices()
    rows=[]
    for group,groupslices in slices.items():
        for sname,mask in groupslices.items():
            n=int(mask.sum())
            if n<10:  # too few days to be meaningful
                print(f"  [skip] {group}/{sname}: only {n} days"); continue
            r=Xva[mask]
            for label in models:
                mm=metrics(r, gen[label][mask])
                rows.append(dict(group=group, slice=sname, n=n, model=label, **{k:round(mm[k],4) for k in METRIC_KEYS}))

    # write metrics.csv
    csv_path=outdir/"metrics.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["group","slice","n","model"]+METRIC_KEYS); w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")

    # figures: for each metric, a grouped bar chart across slices (converged model), + baseline vs converged deltas
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
    def rows_for(group,model): return [r for r in rows if r["group"]==group and r["model"]==model]
    BLUE,LTBLUE="#3266ad","#9ec5e8"
    for group in ["temp","calendar","season"]:
        conv=rows_for(group,"converged"); base=rows_for(group,"baseline")
        if not conv: continue
        names=[r["slice"] for r in conv]
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for ax,key in zip(axes,["RMSE","FirstDiff","Peak"]):
            x=np.arange(len(names)); w=0.38
            if base: ax.bar(x-w/2,[r[key] for r in base],w,color=LTBLUE,label="baseline")
            ax.bar(x+w/2,[r[key] for r in conv],w,color=BLUE,label="converged")
            ax.set_xticks(x); ax.set_xticklabels(names,rotation=20,ha="right",fontsize=8)
            ax.set_title(key,fontweight="bold",fontsize=11); 
            if key=="RMSE": ax.legend(fontsize=8)
        fig.suptitle(f"Performance by {group} slice (lower = better)",fontweight="bold")
        fig.tight_layout(); fig.savefig(outdir/f"by_{group}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        print(f"  wrote by_{group}.png")

    # summary_report.md
    write_conditional_report(outdir, rows, slices)
    print(f"\nDONE. See {outdir}")

def write_conditional_report(outdir, rows, slices):
    lines=["# Week 6 — Conditional Subset Evaluation\n",
           "Diffusion models evaluated on validation slices (temperature, calendar, season).",
           "Lower is better on every metric. All values in normalized [0,1] load units.\n"]
    for group in ["temp","calendar","season"]:
        lines.append(f"\n## {group.capitalize()}\n")
        conv=[r for r in rows if r["group"]==group and r["model"]=="converged"]
        if not conv: continue
        lines.append("| Slice | n | RMSE | FirstDiff | ACF | Peak | Energy | Ramp |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in conv:
            f=lambda k: f"{float(r[k]):.4f}"
            lines.append(f"| {r['slice']} | {r['n']} | {f('RMSE')} | {f('FirstDiff')} | {f('ACF')} | {f('Peak')} | {f('Energy')} | {f('Ramp')} |")
        # highlight hardest slice by RMSE
        worst=max(conv,key=lambda r:r["RMSE"]); best=min(conv,key=lambda r:r["RMSE"])
        lines.append(f"\n*Hardest {group} slice by RMSE:* **{worst['slice']}** ({float(worst['RMSE']):.4f}).  "
                     f"*Easiest:* **{best['slice']}** ({float(best['RMSE']):.4f}).")
    (outdir/"summary_report.md").write_text("\n".join(lines))
    print(f"  wrote summary_report.md")

# ════════════════════════════════════════════════════════════════════════
# TASK: noise-schedule experiment (train 3, compare)
# ════════════════════════════════════════════════════════════════════════
def task_noise(args):
    outdir=ROOT/"reports"/"week6"/"noise"; outdir.mkdir(parents=True,exist_ok=True)
    Xtr_t=torch.tensor(Xtr); Ztr_t=torch.tensor(Ztr); Zva_t=torch.tensor(Zva)
    schedules=["linear","cosine","quadratic"]
    hist={}; results=[]
    for sch in schedules:
        print(f"\n=== Training schedule: {sch} ===")
        torch.manual_seed(0)  # identical init/data order per schedule (only schedule differs)
        model=ConditionalDenoiser(z_dim=Zva.shape[1])
        ddpm=DDPM(sch)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
        dl=DataLoader(TensorDataset(Xtr_t,Ztr_t),batch_size=args.batch,shuffle=True,drop_last=True)
        val_hist=[]; best=1e9; best_state=None
        t0=time.time()
        for ep in range(1,args.epochs+1):
            model.train()
            for xb,zb in dl:
                loss=ddpm.training_loss(model,xb,zb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vl=np.mean([ddpm.training_loss(model,torch.tensor(Xva),Zva_t).item() for _ in range(4)])
            val_hist.append(vl)
            if vl<best: best=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}
            if ep%25==0 or ep==1: print(f"  ep {ep:>3}  val {vl:.5f}  best {best:.5f}")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), OUTDIR/f"noise_{sch}_z5.pt")
        hist[sch]=val_hist
        # evaluate generated quality with the correct sampler
        g=ddpm.sample(model, Zva_t)
        mm=metrics(Xva, g)
        results.append(dict(schedule=sch, best_val=round(best,5), epochs=args.epochs,
                            minutes=round((time.time()-t0)/60,1), **{k:round(mm[k],4) for k in METRIC_KEYS}))
        print(f"  {sch}: best val {best:.5f}, gen metrics {mm}")

    # metrics.csv
    with open(outdir/"metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["schedule","best_val","epochs","minutes"]+METRIC_KEYS); w.writeheader(); w.writerows(results)
    print(f"\nWrote {outdir/'metrics.csv'}")

    # convergence-curve figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
    COL={"linear":"#3266ad","cosine":"#e09320","quadratic":"#1d9e75"}
    fig,ax=plt.subplots(figsize=(9,4.8))
    for sch in schedules:
        ax.plot(range(1,len(hist[sch])+1),hist[sch],color=COL[sch],lw=2,label=sch)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss (log)")
    ax.set_title("Noise schedule — convergence",fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"convergence.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    # summary report
    lines=["# Week 6 — Noise Schedule Experiment\n",
           "Three diffusion models, identical except the β schedule. Same epochs, lr, architecture, conditioning.\n",
           "| Schedule | Best val | RMSE | FirstDiff | ACF | Peak | Energy | Ramp | Min |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['schedule']} | {r['best_val']} | {r['RMSE']} | {r['FirstDiff']} | {r['ACF']} | {r['Peak']} | {r['Energy']} | {r['Ramp']} | {r['minutes']} |")
    best_val=min(results,key=lambda r:r["best_val"]); best_rmse=min(results,key=lambda r:r["RMSE"])
    lines.append(f"\n*Lowest val loss:* **{best_val['schedule']}** ({best_val['best_val']}).  "
                 f"*Best generated RMSE:* **{best_rmse['schedule']}** ({best_rmse['RMSE']}).")
    (outdir/"summary_report.md").write_text("\n".join(lines))
    print(f"DONE. See {outdir}")

# ════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["conditional","noise"], required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    args=ap.parse_args()
    print(f"Using outputs dir: {OUTDIR}")
    if args.task=="conditional": task_conditional(args)
    else: task_noise(args)