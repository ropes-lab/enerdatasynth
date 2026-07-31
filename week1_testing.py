"""
PJM Hourly Load – Exploratory Data Analysis
Covers every pattern observable in the 2016-2025 dataset.
Run from the project root (where the hrl_load_metered_*.csv files live).

Requirements:
    pip install pandas numpy matplotlib seaborn tzdata
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f6",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE  = "#3266ad"
AMBER = "#e09320"
RED   = "#c94040"
TEAL  = "#1d9e75"
GRAY  = "#888780"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
dfs = []
for yr in range(2016, 2026):
    df = pd.read_csv(f"hrl_load_metered_{yr}.csv")
    df["year"] = yr
    dfs.append(df)
all_load = pd.concat(dfs, ignore_index=True)

# Parse timestamps once
all_load["dt"] = pd.to_datetime(all_load["datetime_beginning_ept"])
all_load["hour"]       = all_load["dt"].dt.hour
all_load["dow"]        = all_load["dt"].dt.dayofweek   # 0=Mon
all_load["month"]      = all_load["dt"].dt.month
all_load["is_weekend"] = all_load["dow"].isin([5, 6])

rto = all_load[all_load["zone"] == "RTO"].copy().sort_values("dt").reset_index(drop=True)
print(f"  RTO rows: {len(rto):,}  |  all-zone rows: {len(all_load):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TEMPERATURE DATA  (for V-curve analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading temperature data …")
temp = pd.read_csv("out/pjm_hourly_temps_wide.csv", usecols=["datetime_ept", "RTO"])
temp["dt"] = pd.to_datetime(temp["datetime_ept"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)
temp_rto = temp[["dt", "RTO"]].rename(columns={"RTO": "temp_f"})

rto_with_temp = rto.merge(temp_rto, on="dt", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY STATISTICS  (printed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── RTO descriptive stats ──────────────────────────────")
print(rto["mw"].describe().round(0).to_string())

print("\n── Year-over-year mean / min / max (MW) ───────────────")
print(rto.groupby("year")["mw"].agg(["mean","min","max"]).round(0).to_string())

print("\n── Monthly average MW ─────────────────────────────────")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = rto.groupby("month")["mw"].mean().round(0)
for m, v in monthly.items():
    print(f"  {month_names[m-1]}: {v:>8,.0f} MW")

print("\n── Weekday vs Weekend average MW ──────────────────────")
print(rto.groupby("is_weekend")["mw"].mean().rename({False: "Weekday", True: "Weekend"}).round(0).to_string())

print("\n── Autocorrelation at key lags ────────────────────────")
s = rto.set_index("dt")["mw"].sort_index()
for lag, label in [(1, "1 h"), (24, "24 h (1 day)"), (168, "168 h (1 week)")]:
    print(f"  lag {label:<14}: {s.autocorr(lag):.4f}")

print("\n── Hourly ramp rates (MW/h) ───────────────────────────")
rto["ramp"] = rto["mw"].diff()
print(f"  Mean abs ramp : {rto['ramp'].abs().mean():>8,.0f}")
print(f"  Max ramp-up   : {rto['ramp'].max():>8,.0f}")
print(f"  Max ramp-down : {rto['ramp'].min():>8,.0f}")

print("\n── COVID impact (Apr 2019 vs Apr 2020) ────────────────")
apr19 = rto[(rto["year"]==2019) & (rto["month"]==4)]["mw"].mean()
apr20 = rto[(rto["year"]==2020) & (rto["month"]==4)]["mw"].mean()
print(f"  Apr 2019: {apr19:,.0f} MW  |  Apr 2020: {apr20:,.0f} MW  |  drop: {(apr19-apr20)/apr19*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTS  (3 figure files, each self-contained)
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure 1: Time-series overview ─────────────────────────────────────────
print("\nPlotting Figure 1 – time series overview …")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("PJM RTO Load – Time Series Overview", fontsize=14, fontweight="bold", y=0.99)

# a) Full 10-year monthly average
monthly_ts = rto.groupby(rto["dt"].dt.to_period("M"))["mw"].mean()
monthly_ts.index = monthly_ts.index.to_timestamp()
axes[0].fill_between(monthly_ts.index, monthly_ts.values / 1000, alpha=0.25, color=BLUE)
axes[0].plot(monthly_ts.index, monthly_ts.values / 1000, color=BLUE, lw=1.5)
axes[0].set_title("Monthly average load, 2016–2025", fontsize=11)
axes[0].set_ylabel("GW")
axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
                color=RED, alpha=0.1, label="COVID period")
axes[0].legend(fontsize=9)

# b) One representative winter week (Jan 2024) + one summer week (Jul 2024)
jan_week = rto[(rto["year"]==2024) & (rto["month"]==1)].iloc[:168]
jul_week = rto[(rto["year"]==2024) & (rto["month"]==7)].iloc[:168]
hours = np.arange(168)
axes[1].plot(hours, jan_week["mw"].values / 1000, color=BLUE,  lw=1.5, label="Jan 2024 (winter)")
axes[1].plot(hours, jul_week["mw"].values / 1000, color=AMBER, lw=1.5, label="Jul 2024 (summer)")
axes[1].set_title("168-hour profile: winter vs summer week", fontsize=11)
axes[1].set_ylabel("GW")
axes[1].set_xlabel("Hour of week")
axes[1].legend(fontsize=9)
for d in range(0, 168, 24):
    axes[1].axvline(d, color="gray", lw=0.4, ls="--")

# c) Year-over-year box plots
rto_box = rto[["year","mw"]].copy()
years_sorted = sorted(rto_box["year"].unique())
bp_data = [rto_box[rto_box["year"]==y]["mw"].values / 1000 for y in years_sorted]
bp = axes[2].boxplot(bp_data, labels=years_sorted, patch_artist=True,
                     medianprops={"color":"black","lw":1.5},
                     whiskerprops={"lw":0.8}, capprops={"lw":0.8},
                     flierprops={"marker":".", "markersize":2, "color":GRAY, "alpha":0.3})
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.4)
axes[2].set_title("Annual load distribution by year", fontsize=11)
axes[2].set_ylabel("GW")
axes[2].set_xlabel("Year")

fig.tight_layout()
fig.savefig("out_eda_fig1_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: out_eda_fig1_timeseries.png")

# ── Figure 2: Temporal patterns ─────────────────────────────────────────────
print("Plotting Figure 2 – temporal patterns …")
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("PJM RTO Load – Temporal Patterns", fontsize=14, fontweight="bold")

# a) Diurnal profile: weekday vs weekend
ax1 = fig.add_subplot(gs[0, 0])
wd_hr = rto[~rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
we_hr = rto[ rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
ax1.plot(wd_hr.index, wd_hr.values, color=BLUE,  lw=2, label="Weekday")
ax1.plot(we_hr.index, we_hr.values, color=AMBER, lw=2, label="Weekend")
ax1.fill_between(wd_hr.index, wd_hr.values, we_hr.values,
                 where=(wd_hr.values > we_hr.values), alpha=0.15, color=BLUE)
ax1.set_title("Diurnal profile (avg by hour)", fontsize=11)
ax1.set_xlabel("Hour of day (EPT)")
ax1.set_ylabel("GW")
ax1.set_xticks(range(0, 24, 3))
ax1.legend(fontsize=9)

# b) Monthly seasonal pattern
ax2 = fig.add_subplot(gs[0, 1])
monthly_vals = rto.groupby("month")["mw"].mean() / 1000
colors_month = [BLUE if m in [12,1,2] else AMBER if m in [6,7,8] else GRAY
                for m in monthly_vals.index]
bars = ax2.bar(month_names, monthly_vals.values, color=colors_month, alpha=0.75, edgecolor="white")
ax2.set_title("Seasonal pattern (monthly avg)", fontsize=11)
ax2.set_ylabel("GW")
ax2.set_xticklabels(month_names, rotation=45)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=BLUE, alpha=0.75, label="Winter"),
                    Patch(color=AMBER, alpha=0.75, label="Summer"),
                    Patch(color=GRAY, alpha=0.75, label="Shoulder")], fontsize=9)

# c) Day-of-week pattern
ax3 = fig.add_subplot(gs[1, 0])
dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_vals  = rto.groupby("dow")["mw"].mean() / 1000
dow_colors = [BLUE if d < 5 else AMBER for d in range(7)]
ax3.bar(dow_names, dow_vals.values, color=dow_colors, alpha=0.75, edgecolor="white")
ax3.set_title("Day-of-week pattern", fontsize=11)
ax3.set_ylabel("GW")

# d) Hourly heat-map: hour × month
ax4 = fig.add_subplot(gs[1, 1])
hm = rto.groupby(["month","hour"])["mw"].mean().unstack(level=1) / 1000
im = ax4.imshow(hm.values, aspect="auto", cmap="RdYlBu_r",
                extent=[-0.5, 23.5, 12.5, 0.5])
ax4.set_title("Load heat-map: month × hour", fontsize=11)
ax4.set_xlabel("Hour of day")
ax4.set_ylabel("Month")
ax4.set_yticks(range(1, 13))
ax4.set_yticklabels(month_names)
ax4.set_xticks(range(0, 24, 3))
fig.colorbar(im, ax=ax4, label="GW", shrink=0.85)

fig.savefig("out_eda_fig2_patterns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: out_eda_fig2_patterns.png")

# ── Figure 3: Load-temperature V-curve + zone correlations ──────────────────
print("Plotting Figure 3 – temperature & zone analysis …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PJM – Temperature Sensitivity & Zone Correlations", fontsize=14, fontweight="bold")

# a) V-curve (scatter + binned mean)
merged_valid = rto_with_temp.dropna(subset=["temp_f"])
sample = merged_valid.sample(min(15000, len(merged_valid)), random_state=42)
axes[0].scatter(sample["temp_f"], sample["mw"] / 1000,
                alpha=0.05, s=4, color=GRAY, rasterized=True)
# binned means
rto_with_temp["temp_bin_mid"] = (pd.cut(rto_with_temp["temp_f"],
                                         bins=range(0, 105, 5),
                                         labels=range(2, 102, 5)
                                        ).astype(float))
binned = rto_with_temp.groupby("temp_bin_mid", observed=True)["mw"].mean() / 1000
axes[0].plot(binned.index, binned.values, color=RED, lw=2.5, zorder=5, label="Bin mean")
axes[0].axvline(62, color=TEAL, lw=1.2, ls="--", label="Comfort zone ≈ 60–65°F")
axes[0].axvline(67, color=TEAL, lw=1.2, ls="--")
axes[0].set_title("Load vs temperature – V-curve", fontsize=11)
axes[0].set_xlabel("Temperature (°F) – RTO average")
axes[0].set_ylabel("Load (GW)")
axes[0].legend(fontsize=9)

# b) Zone correlation heat-map (2024)
load24 = all_load[all_load["year"] == 2024].copy()
pivot24 = load24.pivot_table(index="dt", columns="zone", values="mw")
zone_sel = ["RTO","DOM","AEP","CE","DPL","PN","PS","DUQ","AP","EKPC"]
zone_sel = [z for z in zone_sel if z in pivot24.columns]
corr_mat = pivot24[zone_sel].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
            vmin=0.6, vmax=1.0, mask=mask, square=True,
            linewidths=0.3, annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8})
axes[1].set_title("Zone load correlation matrix (2024)", fontsize=11)
axes[1].tick_params(axis="x", rotation=45)
axes[1].tick_params(axis="y", rotation=0)

fig.tight_layout()
fig.savefig("out_eda_fig3_temp_zones.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: out_eda_fig3_temp_zones.png")

# ── Figure 4: Autocorrelation + load duration curve + ramps ─────────────────
print("Plotting Figure 4 – ACF, load duration, ramps …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PJM RTO – Signal Structure", fontsize=14, fontweight="bold")

# a) ACF up to 200 lags
s_rto = rto.set_index("dt")["mw"].sort_index().dropna()
max_lag = 200
acf_vals = [s_rto.autocorr(lag) for lag in range(1, max_lag + 1)]
axes[0].bar(range(1, max_lag + 1), acf_vals, color=BLUE, alpha=0.6, width=1)
for lag, label in [(24, "24 h"), (48, "48 h"), (168, "1 wk")]:
    axes[0].axvline(lag, color=RED, lw=1, ls="--")
    axes[0].text(lag + 1, 0.55, label, color=RED, fontsize=8)
axes[0].set_title("Autocorrelation function (lags 1–200 h)", fontsize=11)
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("ACF")
axes[0].set_ylim(0, 1.05)

# b) Load duration curve by year (select years)
years_ldc = [2016, 2019, 2020, 2022, 2025]
cmap_ldc  = plt.cm.Blues(np.linspace(0.35, 0.95, len(years_ldc)))
for i, yr in enumerate(years_ldc):
    yr_data = rto[rto["year"] == yr]["mw"].sort_values(ascending=False).values / 1000
    pct = np.linspace(0, 100, len(yr_data))
    axes[1].plot(pct, yr_data, lw=1.5, color=cmap_ldc[i],
                 label=str(yr), alpha=0.9)
axes[1].set_title("Load duration curve by year", fontsize=11)
axes[1].set_xlabel("% of hours in year")
axes[1].set_ylabel("GW")
axes[1].legend(fontsize=8, title="Year")

# c) Hourly ramp distribution
ramps = rto["ramp"].dropna() / 1000
axes[2].hist(ramps, bins=120, color=BLUE, alpha=0.7, edgecolor="none")
axes[2].axvline(0, color="black", lw=0.8)
pct5, pct95 = np.percentile(ramps, 5), np.percentile(ramps, 95)
axes[2].axvline(pct5,  color=RED, lw=1, ls="--", label=f"P5 = {pct5:.1f} GW")
axes[2].axvline(pct95, color=RED, lw=1, ls="--", label=f"P95 = {pct95:.1f} GW")
axes[2].set_title("Hourly ramp-rate distribution", fontsize=11)
axes[2].set_xlabel("ΔMW hour-over-hour (GW)")
axes[2].set_ylabel("Count")
axes[2].legend(fontsize=9)

fig.tight_layout()
fig.savefig("out_eda_fig4_signal.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: out_eda_fig4_signal.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. QUICK METRICS SUMMARY  (will be useful for T1.2 week 2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Evaluation metrics summary (T1.2 prep) ─────────────")
print("Marginal distribution (GW):")
pcts = rto["mw"].quantile([0.01,0.05,0.25,0.5,0.75,0.95,0.99]) / 1000
for q, v in pcts.items():
    print(f"  P{int(q*100):>2}: {v:.2f}")

print("\nSeasonal averages (GW):")
for season, months in [("Summer (JJA)", [6,7,8]), ("Winter (DJF)", [12,1,2]),
                        ("Spring (MAM)", [3,4,5]), ("Fall   (SON)", [9,10,11])]:
    val = rto[rto["month"].isin(months)]["mw"].mean() / 1000
    print(f"  {season}: {val:.2f}")

print("\nAll 4 figures saved. EDA complete.")