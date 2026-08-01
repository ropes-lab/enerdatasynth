"""
Week 7 — Promote the transformer to the primary baseline  (RUN ON YOUR MAC)
===========================================================================
Week 6 showed a transformer denoiser beats the MLP on nearly every metric at
~6x fewer parameters. This week we make it official:

  --task train    Train the transformer to convergence and save it as the new
                  primary model (transformer_baseline_z5.pt). Val tracking +
                  best-checkpoint + early stopping, like train_converged.py.

  --task evaluate Re-run the FULL evaluation on the transformer baseline:
                    (a) head-to-head vs MLP (converged) + 4 classical models
                    (b) conditional slices (temperature / calendar / season)
                  so we can confirm the win holds up everywhere, not just on
                  the global average.

Everything uses the CORRECTED ancestral sampler.

Usage (from project root), in order:
    /opt/homebrew/bin/python3.11 week7.py --task train --epochs 400
    /opt/homebrew/bin/python3.11 week7.py --task evaluate

Outputs -> reports/week7/
Requires: numpy, matplotlib, torch, statsmodels
"""
import argparse, math, time, csv, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

ROOT = Path("/Users/sonakshisharma/Downloads/data_for_everyone_pjm")
OUTDIR = next((p for p in [ROOT/"week3_outputs", ROOT/"pjm_diffusion"/"week3_outputs", Path("week3_outputs")] if p.exists()), None)
assert OUTDIR, "week3_outputs not found"
REPORTS = ROOT/"reports"/"week7"; REPORTS.mkdir(parents=True, exist_ok=True)

Xtr = np.load(OUTDIR/"X_train.npy").astype(np.float32)
Xva = np.load(OUTDIR/"X_val.npy").astype(np.float32)
Ztr = np.load(OUTDIR/"Z5_train.npy").astype(np.float32)
Zva = np.load(OUTDIR/"Z5_val.npy").astype(np.float32)
Z_DIM = Ztr.shape[1]; H = 24
COL_MONTH, COL_DOW, COL_HOL, COL_TMAX = 0, 1, 2, 3

TRANSFORMER_CKPT = "transformer_baseline_z5.pt"

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

# ── Transformer denoiser (the Week-6 winner, verbatim) ──────────────────
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

# ── MLP denoiser (for the comparison row) ───────────────────────────────
class ResBlock(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d)); s.act = nn.SiLU()
    def forward(s, x): return s.act(x + s.net(x))
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

class DDPM:
    def __init__(s, T=1000, b0=1e-4, b1=0.02):
        s.T = T; s.betas = torch.linspace(b0, b1, T)
        s.alphas = 1.0 - s.betas; s.alpha_bar = torch.cumprod(s.alphas, 0)
    def training_loss(s, model, x0, z):
        B = x0.shape[0]; t = torch.randint(0, s.T, (B,))
        noise = torch.randn_like(x0); ab = s.alpha_bar[t][:, None]
        xt = ab.sqrt()*x0 + (1-ab).sqrt()*noise
        return nn.functional.mse_loss(model(xt, t, z), noise)
    @torch.no_grad()
    def sample(s, model, z):
        B = z.shape[0]; x = torch.randn(B, H)
        for t in reversed(range(s.T)):
            tv = torch.full((B,), t, dtype=torch.long)
            eps = model(x, tv, z)
            a, ab, b = s.alphas[t], s.alpha_bar[t], s.betas[t]
            mean = (x - b/(1-ab).sqrt()*eps) / a.sqrt()
            x = mean + (b.sqrt()*torch.randn_like(x) if t > 0 else 0)
        return x.numpy()

# ── metrics ──
def _acf(M, ml=12):
    o = []
    for row in M:
        row = row - row.mean(); d = (row*row).sum()
        if d <= 0: continue
        o.append([(row[:len(row)-k]*row[k:]).sum()/d for k in range(ml+1)])
    return np.array(o).mean(0) if o else np.zeros(ml+1)
def metrics(r, g):
    return dict(
        Hourly=float(np.abs(r.mean(0)-g.mean(0)).mean()+np.abs((np.percentile(r,75,0)-np.percentile(r,25,0))-(np.percentile(g,75,0)-np.percentile(g,25,0))).mean()),
        FirstDiff=float(abs(np.diff(r,1).std()-np.diff(g,1).std())),
        ACF=float(np.abs(_acf(r)-_acf(g)).mean()),
        RMSE=float(np.sqrt(((r.mean(0)-g.mean(0))**2).mean())),
        Peak=float(abs(r.max(1).mean()-g.max(1).mean())),
        Energy=float(abs(r.sum(1).mean()-g.sum(1).mean())),
        Ramp=float(abs(np.abs(np.diff(r,1)).mean()-np.abs(np.diff(g,1)).mean())))
MK = ["Hourly","FirstDiff","ACF","RMSE","Peak","Energy","Ramp"]

# ════════════════════════════════════════════════════════════════════════
# TASK: train the transformer to convergence -> new baseline
# ════════════════════════════════════════════════════════════════════════
def task_train(args):
    Xtr_t, Ztr_t = torch.tensor(Xtr), torch.tensor(Ztr)
    Xva_t, Zva_t = torch.tensor(Xva), torch.tensor(Zva)
    torch.manual_seed(0); np.random.seed(0)
    model = TransformerDenoiser(); ddpm = DDPM()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dl = DataLoader(TensorDataset(Xtr_t, Ztr_t), batch_size=args.batch, shuffle=True, drop_last=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Training TRANSFORMER baseline to convergence — {n_par:,} params, max {args.epochs} epochs, patience {args.patience}")

    tr_hist, val_hist = [], []
    best, best_ep, bad, best_state = 1e9, 0, 0, None
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train(); acc = 0.0
        for xb, zb in dl:
            loss = ddpm.training_loss(model, xb, zb)
            opt.zero_grad(); loss.backward(); opt.step(); acc += loss.item()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([ddpm.training_loss(model, Xva_t, Zva_t).item() for _ in range(6)]))
        tr_hist.append(acc/len(dl)); val_hist.append(vl)
        if vl < best - 1e-5: best, best_ep, bad, best_state = vl, ep, 0, {k:v.clone() for k,v in model.state_dict().items()}
        else: bad += 1
        if ep % 20 == 0 or ep == 1: print(f"  ep {ep:>3}  train {acc/len(dl):.5f}  val {vl:.5f}  best {best:.5f}")
        if bad >= args.patience:
            print(f"  early stop at epoch {ep}"); break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUTDIR/TRANSFORMER_CKPT)
    mins = (time.time()-t0)/60
    print(f"\nBest val {best:.5f} at epoch {best_ep}. {mins:.1f} min.")
    print(f"Saved -> {OUTDIR/TRANSFORMER_CKPT}")

    # loss curve
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.grid":True,"grid.color":"#e8e8e8","axes.spines.top":False,"axes.spines.right":False})
    ep = range(1, len(tr_hist)+1)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(ep, tr_hist, color="#29ABE2", lw=2, label="train")
    ax.plot(ep, val_hist, color="#F2A900", lw=2, label="val")
    ax.axvline(best_ep, color="#888", ls="--", lw=1.2, label=f"best epoch ({best_ep})")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
    ax.set_title("Transformer baseline — training to convergence", fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS/"transformer_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {REPORTS/'transformer_loss.png'}")
    print("\nNEXT: run  week7.py --task evaluate")

# ════════════════════════════════════════════════════════════════════════
# TASK: full evaluation of the transformer baseline
# ════════════════════════════════════════════════════════════════════════
def load(cls, ckpt):
    m = cls(); m.load_state_dict(torch.load(OUTDIR/ckpt, map_location="cpu")); m.eval(); return m

def task_evaluate(args):
    assert (OUTDIR/TRANSFORMER_CKPT).exists(), "Train first: week7.py --task train"
    ddpm = DDPM()

    # ---- (a) head-to-head: transformer vs MLP(converged) vs classical ----
    print("Generating from transformer baseline...")
    tf = load(TransformerDenoiser, TRANSFORMER_CKPT)
    gen = {"Transformer (new baseline)": ddpm.sample(tf, torch.tensor(Zva))}
    if (OUTDIR/"converged_model_z5.pt").exists():
        print("Generating from MLP (converged, old baseline)...")
        gen["MLP (old baseline)"] = ddpm.sample(load(MLPDenoiser, "converged_model_z5.pt"), torch.tensor(Zva))

    from statsmodels.tsa.arima.model import ARIMA
    def fit_sample(order, so, n, steps=24):
        mp = Xtr.mean(0)
        res = ARIMA(mp, order=order, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False).fit(method_kwargs={"maxiter":200})
        sims = np.zeros((n, steps))
        for i in range(n): sims[i] = res.simulate(nsimulations=steps, anchor="start", repetitions=1).reshape(-1)[:steps]
        return np.clip(sims - sims.mean(0, keepdims=True) + mp[None,:], -0.2, 1.2)
    print("Fitting classical baselines...")
    for name,(o,so) in {"AR":((2,0,0),(0,0,0,0)),"ARMA":((2,0,2),(0,0,0,0)),
                        "ARIMA":((2,1,2),(0,0,0,0)),"SARIMA":((1,0,1),(1,1,1,12))}.items():
        gen[name] = fit_sample(o, so, len(Xva)); print(f"  {name}")

    order = ["AR","ARMA","ARIMA","SARIMA","MLP (old baseline)","Transformer (new baseline)"]
    order = [o for o in order if o in gen]
    rows = [dict(model=m, **{k:round(metrics(Xva, gen[m])[k],4) for k in MK}) for m in order]
    with open(REPORTS/"headtohead.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"]+MK); w.writeheader(); w.writerows(rows)
    print("\n=== HEAD-TO-HEAD ===")
    print(f"{'model':28s}"+"".join(f"{k:>10s}" for k in MK))
    for r in rows: print(f"{r['model']:28s}"+"".join(f"{r[k]:>10.4f}" for k in MK))

    # ---- (b) conditional slices for the transformer ----
    tmax=Zva[:,COL_TMAX]; month=Zva[:,COL_MONTH]; dow=Zva[:,COL_DOW]; hol=Zva[:,COL_HOL]
    q25,q75=np.percentile(tmax,[25,75])
    def mon(ms): return np.isin(month.astype(int), ms)
    slices={
        ("temp","Cold (bottom 25%)"): tmax<=q25, ("temp","Mild (mid 50%)"): (tmax>q25)&(tmax<q75), ("temp","Hot (top 25%)"): tmax>=q75,
        ("calendar","Weekday"): (dow<=4)&(hol<0.5), ("calendar","Weekend"): dow>=5, ("calendar","Holiday"): hol>=0.5,
        ("season","Winter"): mon([12,1,2]), ("season","Spring"): mon([3,4,5]), ("season","Summer"): mon([6,7,8]), ("season","Fall"): mon([9,10,11]),
    }
    g_tf = gen["Transformer (new baseline)"]
    srows=[]
    for (grp,sname),mask in slices.items():
        n=int(mask.sum())
        if n<10: continue
        mm=metrics(Xva[mask], g_tf[mask])
        srows.append(dict(group=grp, slice=sname, n=n, **{k:round(mm[k],4) for k in MK}))
    with open(REPORTS/"conditional_transformer.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["group","slice","n"]+MK); w.writeheader(); w.writerows(srows)
    print("\n=== TRANSFORMER BY SLICE (RMSE / Peak / Ramp) ===")
    for r in srows: print(f"  {r['group']:9s} {r['slice']:18s} n={r['n']:<4d} RMSE {r['RMSE']:.4f}  Peak {r['Peak']:.4f}  Ramp {r['Ramp']:.4f}")

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#f8f8f6","axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False})
    COL={"AR":"#9aa7b8","ARMA":"#7B5EA7","ARIMA":"#1d9e75","SARIMA":"#e09320","MLP (old baseline)":"#9ec5e8","Transformer (new baseline)":"#3266ad"}
    hours=np.arange(H)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(hours, Xva.mean(0), color="black", lw=3, label="Real", zorder=10)
    for m in order: ax.plot(hours, gen[m].mean(0), color=COL.get(m,"#aaa"), lw=2, label=m, alpha=0.9)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalized load [0,1]"); ax.set_xticks(range(0,24,3))
    ax.set_title("Mean profile — transformer baseline vs all", fontweight="bold"); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS/"profiles_headtohead.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # conditional bar (RMSE by slice group)
    for grp in ["temp","calendar","season"]:
        rr=[r for r in srows if r["group"]==grp]
        if not rr: continue
        fig,ax=plt.subplots(figsize=(7,3.4))
        ax.bar([r["slice"] for r in rr], [r["RMSE"] for r in rr], color="#3266ad")
        ax.set_title(f"Transformer RMSE by {grp} slice", fontweight="bold"); ax.set_ylabel("RMSE")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        fig.tight_layout(); fig.savefig(REPORTS/f"tf_by_{grp}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- report ----
    tf_row = next(r for r in rows if r["model"].startswith("Transformer"))
    mlp_row = next((r for r in rows if r["model"].startswith("MLP")), None)
    best_classical = {k: min(r[k] for r in rows if r["model"] in ["AR","ARMA","ARIMA","SARIMA"]) for k in MK}
    lines = ["# Week 7 — Transformer Promoted to Baseline\n",
             "The transformer denoiser (Week-6 architecture winner) is trained to convergence and",
             "adopted as the primary model. Full re-evaluation below.\n",
             "## Head-to-head (lower is better)\n",
             "| Model | " + " | ".join(MK) + " |", "|" + "---|"*(len(MK)+1)]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(f"{r[k]}" for k in MK) + " |")
    wins = sum(1 for k in MK if tf_row[k] <= best_classical[k])
    lines.append(f"\nTransformer beats the best classical baseline on **{wins} of {len(MK)}** metrics.")
    if mlp_row:
        better = sum(1 for k in MK if tf_row[k] <= mlp_row[k])
        lines.append(f"Transformer beats the old MLP baseline on **{better} of {len(MK)}** metrics, "
                     f"at ~6x fewer parameters (110K vs 688K).")
    lines.append("\n## Consistency across regimes (transformer)\n")
    lines.append("| Group | Slice | n | RMSE | Peak | Ramp |"); lines.append("|---|---|---|---|---|---|")
    for r in srows: lines.append(f"| {r['group']} | {r['slice']} | {r['n']} | {r['RMSE']} | {r['Peak']} | {r['Ramp']} |")
    worst = max(srows, key=lambda r: r["RMSE"])
    lines.append(f"\nHardest slice for the new baseline: **{worst['group']}/{worst['slice']}** (RMSE {worst['RMSE']}).")
    (REPORTS/"summary_report.md").write_text("\n".join(lines))
    print(f"\nWrote {REPORTS/'summary_report.md'} + figures. DONE.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["train","evaluate"], required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    print(f"Using outputs dir: {OUTDIR}")
    if args.task == "train": task_train(args)
    else: task_evaluate(args)