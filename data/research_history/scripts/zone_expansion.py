"""
Phase 2 — Zone Expansion  (RUN ON YOUR MAC)
===========================================
Replicates the PS pipeline across the other PJM zones. Same preprocessing, same
Z5 conditioning, same transformer architecture, same corrected sampler --
only the load data, scaler parameters and temperature column change.

Tasks:
  --task build    Build per-zone datasets from the raw CSVs into
                  zone_outputs/<ZONE>/ (X_train, X_val, Z5_train, Z5_val, scaler)
  --task train    Train + calibrate + evaluate a transformer per zone
  --task report   Cross-zone comparison table and figures

Typical run:
    python3 zone_expansion.py --task build
    python3 zone_expansion.py --task train --zones PS JC PE ME PN
    python3 zone_expansion.py --task report

Start with a few zones to check timing, then add more. PS is included so you
can confirm the rebuilt pipeline reproduces your existing result.

Requires: numpy, pandas, matplotlib, torch, holidays
"""
import argparse, math, json, time, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
ZONEDIR = ROOT/"zone_outputs"; ZONEDIR.mkdir(exist_ok=True)
REPORTS = ROOT/"reports"/"phase2"; REPORTS.mkdir(parents=True, exist_ok=True)
H = 24

# Zones that have BOTH metered load and a temperature column
ALL_ZONES = ["PS","JC","PE","ME","PN","AE","BC","DPL","PL","RECO",
             "AP","CE","DAY","DEOK","DUQ","DOM","EKPC"]

# ════════════════════════════════════════════════════════════════════════
# BUILD — per-zone datasets (mirrors week3.py preprocessing)
# ════════════════════════════════════════════════════════════════════════
def task_build(args):
    import pandas as pd
    try:
        import holidays as hol_lib
        us_hol = hol_lib.US()
    except ImportError:
        print("  (holidays package missing — install with: pip3 install holidays)")
        us_hol = None

    print("Loading raw load CSVs...")
    load_files = sorted(ROOT.glob("hrl_load_metered_*.csv"))
    if not load_files:
        load_files = sorted(Path(".").glob("hrl_load_metered_*.csv"))
    assert load_files, "no hrl_load_metered_*.csv found"
    dfs = []
    for f in load_files:
        d = pd.read_csv(f, usecols=["datetime_beginning_ept","zone","mw"])
        dfs.append(d)
    load = pd.concat(dfs, ignore_index=True)
    load["dt"] = pd.to_datetime(load["datetime_beginning_ept"], format="mixed", errors="coerce")
    load = load.dropna(subset=["dt"])
    print(f"  {len(load):,} rows across {load['zone'].nunique()} zones")

    print("Loading temperature table...")
    tpath = ROOT/"temperature data/pjm_hourly_temps_wide.csv"
    if not tpath.exists(): tpath = Path("temperature data/pjm_hourly_temps_wide.csv")
    temps = pd.read_csv(tpath)
    # datetime_ept already carries an EPT offset; strip it rather than convert
    # (avoids depending on the system timezone database)
    _dt = pd.to_datetime(temps["datetime_ept"], errors="coerce", utc=True)
    try:
        temps["dt"] = _dt.dt.tz_convert("US/Eastern").dt.tz_localize(None)
    except Exception:
        # fallback: parse the local wall time directly from the string
        temps["dt"] = pd.to_datetime(temps["datetime_ept"].astype(str).str.slice(0, 19), errors="coerce")
    temps = temps.dropna(subset=["dt"])
    temps["date"] = temps["dt"].dt.date

    zones = args.zones if args.zones else ALL_ZONES
    summary = []
    for zone in zones:
        if zone not in temps.columns:
            print(f"  [skip] {zone}: no temperature column"); continue
        zl = load[load["zone"] == zone].copy()
        if zl.empty:
            print(f"  [skip] {zone}: no load rows"); continue
        zl["date"] = zl["dt"].dt.date
        zl["hour"] = zl["dt"].dt.hour

        # pivot to one row per day x 24 hours; drop days without exactly 24 (DST)
        piv = zl.pivot_table(index="date", columns="hour", values="mw", aggfunc="mean")
        piv = piv.dropna(axis=0, how="any")
        piv = piv[[c for c in range(24) if c in piv.columns]]
        piv = piv[piv.notna().sum(axis=1) == 24]

        # daily temperature aggregates
        tz = temps[["date", zone]].dropna()
        tagg = tz.groupby("date")[zone].agg(["max","min"])
        df = piv.join(tagg, how="inner").dropna()

        # drop COVID period (Mar-Jun 2020), matching week3
        idx = pd.to_datetime(pd.Series(df.index.astype(str)))
        covid = (idx.dt.year == 2020) & (idx.dt.month.between(3, 6))
        df = df[~covid.values]
        if len(df) < 500:
            print(f"  [skip] {zone}: only {len(df)} clean days"); continue

        dates = pd.to_datetime(pd.Series(df.index.astype(str)))
        X_all = df[[c for c in range(24)]].to_numpy(dtype=np.float32)
        month = dates.dt.month.to_numpy(); dow = dates.dt.dayofweek.to_numpy()
        is_hol = np.array([1.0 if (us_hol and d.date() in us_hol) else 0.0 for d in dates], dtype=np.float32)
        tmax = df["max"].to_numpy(dtype=np.float32); tmin = df["min"].to_numpy(dtype=np.float32)
        spread_raw = X_all.max(1) - X_all.min(1)

        # chronological split: train <=2023, val 2024+
        yr = dates.dt.year.to_numpy()
        tr, va = yr <= 2023, yr >= 2024
        if tr.sum() < 300 or va.sum() < 50:
            print(f"  [skip] {zone}: split too small ({tr.sum()}/{va.sum()})"); continue

        # per-hour MinMax fit on TRAIN only
        lo = X_all[tr].min(0); hi = X_all[tr].max(0); rng = np.where(hi-lo == 0, 1, hi-lo)
        Xn = ((X_all - lo)/rng).astype(np.float32)
        # z-score temps on train; minmax spread on train
        tmu, tsd = tmax[tr].mean(), tmax[tr].std() or 1
        nmu, nsd = tmin[tr].mean(), tmin[tr].std() or 1
        smin, smax = spread_raw[tr].min(), spread_raw[tr].max()
        srng = (smax - smin) or 1
        Z = np.column_stack([month, dow, is_hol,
                             (tmax-tmu)/tsd, (tmin-nmu)/nsd,
                             (spread_raw-smin)/srng]).astype(np.float32)

        out = ZONEDIR/zone; out.mkdir(parents=True, exist_ok=True)
        np.save(out/"X_train.npy", Xn[tr]); np.save(out/"X_val.npy", Xn[va])
        np.save(out/"Z5_train.npy", Z[tr]); np.save(out/"Z5_val.npy", Z[va])
        json.dump({"zone":zone,"load_min":lo.tolist(),"load_max":hi.tolist(),
                   "tmax_mean":float(tmu),"tmax_std":float(tsd),
                   "tmin_mean":float(nmu),"tmin_std":float(nsd),
                   "spread_min":float(smin),"spread_max":float(smax),
                   "n_train":int(tr.sum()),"n_val":int(va.sum()),
                   "mean_mw":float(X_all[tr].mean()),
                   "cv_pct":float(100*X_all[tr].std()/X_all[tr].mean())},
                  open(out/"scaler.json","w"), indent=2)
        summary.append(dict(zone=zone, n_train=int(tr.sum()), n_val=int(va.sum()),
                            mean_mw=round(float(X_all[tr].mean()),1),
                            cv_pct=round(float(100*X_all[tr].std()/X_all[tr].mean()),1)))
        print(f"  {zone:6s} train {tr.sum():5d}  val {va.sum():4d}  mean {X_all[tr].mean():8.1f} MW  CV {100*X_all[tr].std()/X_all[tr].mean():5.1f}%")

    with open(REPORTS/"zone_datasets.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["zone","n_train","n_val","mean_mw","cv_pct"]); w.writeheader(); w.writerows(summary)
    print(f"\nBuilt {len(summary)} zone datasets -> {ZONEDIR}")
    print(f"Wrote {REPORTS/'zone_datasets.csv'}")
    print("\nNEXT:  --task train --zones " + " ".join(s["zone"] for s in summary[:5]))

# ════════════════════════════════════════════════════════════════════════
# MODEL (transformer, same as week7) + sampler
# ════════════════════════════════════════════════════════════════════════
def _model_bits():
    import torch, torch.nn as nn
    class SinusoidalEmbedding(nn.Module):
        def __init__(s, dim): super().__init__(); s.dim = dim
        def forward(s, t):
            h=s.dim//2
            f=torch.exp(-math.log(10000)*torch.arange(h, device=t.device)/h)
            a=t[:,None].float()*f[None]; return torch.cat([a.sin(),a.cos()],-1)
    class TransformerDenoiser(nn.Module):
        def __init__(s, z_dim=6, d=64, n_heads=4, n_layers=2, t_dim=64):
            super().__init__()
            s.t_embed=SinusoidalEmbedding(t_dim)
            s.cond=nn.Sequential(nn.Linear(z_dim+t_dim,d), nn.SiLU(), nn.Linear(d,d))
            s.val=nn.Linear(1,d); s.pos=nn.Parameter(torch.randn(1,H,d)*0.02)
            layer=nn.TransformerEncoderLayer(d_model=d,nhead=n_heads,dim_feedforward=d*4,
                                             batch_first=True,dropout=0.0,activation="gelu")
            s.enc=nn.TransformerEncoder(layer,num_layers=n_layers); s.out=nn.Linear(d,1)
        def forward(s,x,t,z):
            c=s.cond(torch.cat([z,s.t_embed(t)],-1))[:,None,:]
            return s.out(s.enc(s.val(x[:,:,None])+s.pos+c))[:,:,0]
    class DDPM:
        def __init__(s,T=1000,b0=1e-4,b1=0.02):
            s.T=T; s.betas=torch.linspace(b0,b1,T)
            s.alphas=1.0-s.betas; s.alpha_bar=torch.cumprod(s.alphas,0)
        def training_loss(s,model,x0,z):
            B=x0.shape[0]; t=torch.randint(0,s.T,(B,))
            noise=torch.randn_like(x0); ab=s.alpha_bar[t][:,None]
            xt=ab.sqrt()*x0+(1-ab).sqrt()*noise
            return nn.functional.mse_loss(model(xt,t,z),noise)
        @torch.no_grad()
        def sample(s,model,z):
            B=z.shape[0]; x=torch.randn(B,H)
            for t in reversed(range(s.T)):
                tv=torch.full((B,),t,dtype=torch.long); eps=model(x,tv,z)
                a,ab,b=s.alphas[t],s.alpha_bar[t],s.betas[t]
                mean=(x-b/(1-ab).sqrt()*eps)/a.sqrt()
                x=mean+(b.sqrt()*torch.randn_like(x) if t>0 else 0)
            return x.numpy()
    return TransformerDenoiser, DDPM

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
MK=["Hourly","FirstDiff","ACF","RMSE","Peak","Energy","Ramp"]

# ════════════════════════════════════════════════════════════════════════
# TRAIN — one transformer per zone, calibrated, evaluated
# ════════════════════════════════════════════════════════════════════════
def task_train(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    TransformerDenoiser, DDPM = _model_bits()
    zones = args.zones if args.zones else [p.name for p in sorted(ZONEDIR.iterdir()) if p.is_dir()]
    results=[]
    for zone in zones:
        zd = ZONEDIR/zone
        if not (zd/"X_train.npy").exists():
            print(f"[skip] {zone}: not built"); continue
        Xtr=np.load(zd/"X_train.npy"); Xva=np.load(zd/"X_val.npy")
        Ztr=np.load(zd/"Z5_train.npy"); Zva=np.load(zd/"Z5_val.npy")
        print(f"\n=== {zone} ===  train {Xtr.shape}  val {Xva.shape}")
        torch.manual_seed(0); np.random.seed(0)
        model=TransformerDenoiser(z_dim=Ztr.shape[1]); ddpm=DDPM()
        opt=torch.optim.AdamW(model.parameters(), lr=args.lr)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(Ztr)),
                      batch_size=args.batch, shuffle=True, drop_last=True)
        Xva_t=torch.tensor(Xva); Zva_t=torch.tensor(Zva)
        best=1e9; best_state=None; bad=0; t0=time.time()
        for ep in range(1,args.epochs+1):
            model.train()
            for xb,zb in dl:
                loss=ddpm.training_loss(model,xb,zb)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step(); model.eval()
            with torch.no_grad():
                vl=float(np.mean([ddpm.training_loss(model,Xva_t,Zva_t).item() for _ in range(4)]))
            if vl<best-1e-5: best,best_state,bad=vl,{k:v.clone() for k,v in model.state_dict().items()},0
            else: bad+=1
            if ep%50==0 or ep==1: print(f"  ep {ep:>3}  val {vl:.5f}  best {best:.5f}")
            if bad>=args.patience: print(f"  early stop @ {ep}"); break
        model.load_state_dict(best_state); model.eval()
        torch.save(model.state_dict(), zd/"transformer_z5.pt")

        # calibrate on TRAIN, then evaluate on VAL
        nfit=min(800,len(Xtr))
        sel=np.random.choice(len(Xtr),nfit,replace=False)
        g_tr=ddpm.sample(model, torch.tensor(Ztr[sel]))
        offset=float(Xtr[sel].mean()-g_tr.mean())
        g=ddpm.sample(model, Zva_t)+offset
        mm=metrics(Xva,g)
        oob=float(np.mean((g<-0.05)|(g>1.05))*100)
        json.dump({"offset":offset}, open(zd/"calibration.json","w"), indent=2)
        mins=(time.time()-t0)/60
        results.append(dict(zone=zone, n_train=len(Xtr), n_val=len(Xva),
                            best_val=round(best,5), offset=round(offset,5),
                            out_of_range_pct=round(oob,1), minutes=round(mins,1),
                            **{k:round(mm[k],4) for k in MK}))
        print(f"  best val {best:.5f} | RMSE {mm['RMSE']:.4f} | offset {offset:+.4f} | {mins:.1f} min")

    path=REPORTS/"zone_results.csv"
    prev=[]
    if path.exists():
        with open(path) as f: prev=[r for r in csv.DictReader(f) if r["zone"] not in {x["zone"] for x in results}]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["zone","n_train","n_val","best_val","offset","out_of_range_pct","minutes"]+MK)
        w.writeheader()
        for r in prev: w.writerow(r)
        w.writerows(results)
    print(f"\nWrote {path}")

# ════════════════════════════════════════════════════════════════════════
# REPORT — cross-zone comparison
# ════════════════════════════════════════════════════════════════════════
def task_report(args):
    path=REPORTS/"zone_results.csv"
    assert path.exists(), "run --task train first"
    with open(path) as f: rows=list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["RMSE"]))
    print(f"\n{'zone':7s}{'n_train':>9s}{'RMSE':>9s}{'Peak':>9s}{'Ramp':>9s}{'Energy':>9s}{'min':>7s}")
    for r in rows:
        print(f"{r['zone']:7s}{int(r['n_train']):>9d}{float(r['RMSE']):>9.4f}"
              f"{float(r['Peak']):>9.4f}{float(r['Ramp']):>9.4f}{float(r['Energy']):>9.4f}{float(r['minutes']):>7.1f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,
                         "grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
    z=[r["zone"] for r in rows]; v=[float(r["RMSE"]) for r in rows]
    fig,ax=plt.subplots(figsize=(max(7,len(z)*0.7),4.2))
    ax.bar(z,v,color=["#3266ad" if x!="PS" else "#1d9e75" for x in z])
    ax.set_ylabel("RMSE"); ax.set_title("Transformer performance by PJM zone (PS in green)",fontweight="bold")
    plt.xticks(rotation=45,ha="right",fontsize=9)
    fig.tight_layout(); fig.savefig(REPORTS/"zone_rmse.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    lines=["# Phase 2 — Zone Expansion Results\n",
           f"{len(rows)} PJM zones, identical pipeline: same preprocessing, Z5 conditioning, "
           "transformer architecture, corrected sampler and calibration step. Only the load data, "
           "scaler parameters and temperature column differ.\n",
           "| Zone | Train days | RMSE | Peak | Ramp | Energy | Min |","|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['zone']} | {r['n_train']} | {float(r['RMSE']):.4f} | {float(r['Peak']):.4f} | "
                     f"{float(r['Ramp']):.4f} | {float(r['Energy']):.4f} | {float(r['minutes']):.1f} |")
    best,worst=rows[0],rows[-1]
    lines.append(f"\nBest zone: **{best['zone']}** (RMSE {float(best['RMSE']):.4f}). "
                 f"Hardest: **{worst['zone']}** (RMSE {float(worst['RMSE']):.4f}).")
    (REPORTS/"summary_report.md").write_text("\n".join(lines))
    print(f"\nWrote {REPORTS/'summary_report.md'} and zone_rmse.png")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=["build","train","report"],required=True)
    ap.add_argument("--zones",nargs="*",default=None)
    ap.add_argument("--epochs",type=int,default=300)
    ap.add_argument("--patience",type=int,default=40)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--batch",type=int,default=64)
    a=ap.parse_args()
    {"build":task_build,"train":task_train,"report":task_report}[a.task](a)