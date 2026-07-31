"""
PJM RTO Load – EDA Plots
Produces 6 publication-ready PNG figures matching the interactive dashboard.

Requirements:
    pip install pandas numpy matplotlib seaborn

Usage:
    python eda_plots.py

Output PNGs saved to the current directory:
    fig1_patterns.png       – Day-of-week, season, and monthly bar charts
    fig2_hourly_profiles.png – Weekday/weekend and seasonal diurnal curves
    fig3_heatmap.png        – Month × hour load heatmap
    fig4_trend_covid.png    – 10-year trend, COVID comparison, year-over-year
    fig5_signal.png         – Autocorrelation function + ramp-rate histogram
    fig6_vcurve.png         – Load vs temperature V-curve
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f6",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

BLUE   = "#3266ad"
AMBER  = "#e09320"
RED    = "#c94040"
TEAL   = "#1d9e75"
GRAY   = "#888780"
PURPLE = "#7f77dd"

MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
SEASONS      = ["Spring","Summer","Fall","Winter"]
HOURS        = list(range(24))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
dfs = []
for yr in range(2016, 2026):
    df = pd.read_csv(f"hrl_load_metered_{yr}.csv")
    df["year"] = yr
    dfs.append(df)
all_load = pd.concat(dfs, ignore_index=True)
all_load["dt"]         = pd.to_datetime(all_load["datetime_beginning_ept"], format="mixed")
all_load["hour"]       = all_load["dt"].dt.hour
all_load["dow"]        = all_load["dt"].dt.dayofweek   # 0=Mon
all_load["month"]      = all_load["dt"].dt.month
all_load["is_weekend"] = all_load["dow"].isin([5, 6])
season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
all_load["season"] = all_load["month"].map(season_map)

rto = all_load[all_load["zone"] == "RTO"].copy().sort_values("dt").reset_index(drop=True)
print(f"  RTO rows: {len(rto):,}")

# Load temperature data for V-curve
print("Loading temperature data …")
temp = pd.read_csv("out/pjm_hourly_temps_wide.csv", usecols=["datetime_ept", "RTO"])
temp["dt"] = pd.to_datetime(temp["datetime_ept"], utc=True).dt.tz_localize(None)
rto_t = rto[["dt","mw"]].copy()
merged = rto_t.merge(temp[["dt","RTO"]], on="dt", how="left").dropna(subset=["RTO"])
merged["tbin"] = pd.cut(merged["RTO"], bins=range(0, 105, 5),
                        labels=range(2, 102, 5)).astype(float)
vcurve = merged.groupby("tbin", observed=True)["mw"].mean() / 1000

# ── Pre-compute aggregates ────────────────────────────────────────────────────
dow_avg    = rto.groupby("dow")["mw"].mean() / 1000
month_avg  = rto.groupby("month")["mw"].mean() / 1000
season_avg = rto.groupby("season")["mw"].mean() / 1000
wd_hr      = rto[~rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
we_hr      = rto[ rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
season_hr  = {s: rto[rto["season"]==s].groupby("hour")["mw"].mean() / 1000
              for s in SEASONS}
hm         = rto.groupby(["month","hour"])["mw"].mean().unstack(level=1) / 1000
yoy        = rto.groupby("year")["mw"].agg(["mean","max","min"]) / 1000
monthly_ts = rto.groupby(rto["dt"].dt.to_period("M"))["mw"].mean() / 1000
monthly_ts.index = monthly_ts.index.to_timestamp()
covid_2019 = rto[rto["year"]==2019].groupby("month")["mw"].mean() / 1000
covid_2020 = rto[rto["year"]==2020].groupby("month")["mw"].mean() / 1000
rto["ramp"] = rto["mw"].diff() / 1000
s_rto = rto.set_index("dt")["mw"].sort_index() / 1000
acf_vals = [s_rto.autocorr(lag) for lag in range(1, 169)]


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Patterns: DoW, season, monthly
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig1_patterns.png …")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("PJM RTO Load – Key Patterns (2016–2025)", fontsize=13, fontweight="bold", y=1.01)

# 1a – Day of week
dow_colors = [BLUE if d < 5 else AMBER for d in range(7)]
axes[0].bar(DOW_NAMES, dow_avg.values, color=dow_colors, alpha=0.85, edgecolor="white", width=0.65)
axes[0].set_title("Average load by day of week")
axes[0].set_ylabel("GW")
axes[0].set_ylim(80, 96)
for i, (name, val) in enumerate(zip(DOW_NAMES, dow_avg.values)):
    axes[0].text(i, val + 0.15, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
axes[0].legend(handles=[mpatches.Patch(color=BLUE, alpha=0.85, label="Weekday"),
                         mpatches.Patch(color=AMBER, alpha=0.85, label="Weekend")], fontsize=8)

# 1b – Season
s_colors = [TEAL, RED, AMBER, BLUE]
s_vals   = [float(season_avg[s]) for s in SEASONS]
axes[1].bar(SEASONS, s_vals, color=s_colors, alpha=0.85, edgecolor="white", width=0.6)
axes[1].set_title("Average load by season")
axes[1].set_ylabel("GW")
axes[1].set_ylim(70, 110)
for i, (s, v) in enumerate(zip(SEASONS, s_vals)):
    axes[1].text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

# 1c – Month
m_colors = [BLUE if m in [12, 1, 2] else RED if m in [6, 7, 8] else GRAY
            for m in range(1, 13)]
axes[2].bar(MONTHS_SHORT, month_avg.values, color=m_colors, alpha=0.85, edgecolor="white", width=0.7)
axes[2].set_title("Average load by month")
axes[2].set_ylabel("GW")
axes[2].set_ylim(68, 115)
for i, (m, v) in enumerate(zip(MONTHS_SHORT, month_avg.values)):
    axes[2].text(i, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
axes[2].legend(handles=[mpatches.Patch(color=BLUE, alpha=0.85, label="Winter"),
                         mpatches.Patch(color=RED, alpha=0.85, label="Summer"),
                         mpatches.Patch(color=GRAY, alpha=0.85, label="Shoulder")], fontsize=8)

fig.tight_layout()
fig.savefig("fig1_patterns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig1_patterns.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Hourly profiles
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig2_hourly_profiles.png …")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("PJM RTO Load – Hourly Profiles", fontsize=13, fontweight="bold", y=1.01)

# 2a – Weekday vs weekend
axes[0].plot(HOURS, wd_hr.values, color=BLUE,  lw=2.2, label="Weekday")
axes[0].plot(HOURS, we_hr.values, color=AMBER, lw=2.2, label="Weekend", linestyle="--")
axes[0].fill_between(HOURS, wd_hr.values, we_hr.values,
                     where=(wd_hr.values > we_hr.values), alpha=0.12, color=BLUE)
axes[0].axvspan(0, 5, color=GRAY, alpha=0.07, label="Night (0–5h)")
axes[0].set_title("Diurnal profile: weekday vs weekend")
axes[0].set_xlabel("Hour of day (EPT)")
axes[0].set_ylabel("GW")
axes[0].set_xticks(range(0, 24, 3))
axes[0].set_ylim(70, 112)
axes[0].legend(fontsize=9)
axes[0].annotate(f"WD peak\n{wd_hr.max():.1f} GW",
                 xy=(wd_hr.idxmax(), wd_hr.max()),
                 xytext=(wd_hr.idxmax()-3, wd_hr.max()+1.5),
                 fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

# 2b – Seasonal hourly profiles
season_styles = {
    "Summer": (RED,   "-",   2.2),
    "Winter": (BLUE,  "--",  2.2),
    "Fall":   (AMBER, "-.",  1.8),
    "Spring": (TEAL,  ":",   1.8),
}
for s, (col, ls, lw) in season_styles.items():
    axes[1].plot(HOURS, season_hr[s].values, color=col, lw=lw, linestyle=ls, label=s)
axes[1].set_title("Hourly profile by season")
axes[1].set_xlabel("Hour of day (EPT)")
axes[1].set_ylabel("GW")
axes[1].set_xticks(range(0, 24, 3))
axes[1].set_ylim(65, 130)
axes[1].legend(fontsize=9)
# Annotate summer peak
pk_hr = int(season_hr["Summer"].idxmax())
pk_v  = season_hr["Summer"].max()
axes[1].annotate(f"Summer peak\n{pk_v:.0f} GW @ {pk_hr}:00",
                 xy=(pk_hr, pk_v), xytext=(pk_hr-5, pk_v-7),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

fig.tight_layout()
fig.savefig("fig2_hourly_profiles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig2_hourly_profiles.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Heatmap (month × hour)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig3_heatmap.png …")
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("PJM RTO Load – Month × Hour Heatmap (GW)", fontsize=13, fontweight="bold")

cmap = LinearSegmentedColormap.from_list(
    "rybw", ["#2166ac", "#ffffbf", "#d73027"], N=256)
im = ax.imshow(hm.values, aspect="auto", cmap=cmap,
               extent=[-0.5, 23.5, 12.5, 0.5])
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], fontsize=8)
ax.set_yticks(range(1, 13))
ax.set_yticklabels(MONTHS_SHORT, fontsize=9)
ax.set_xlabel("Hour of day (EPT)")
ax.set_ylabel("Month")

# Annotate peak cell
peak_m, peak_h = np.unravel_index(hm.values.argmax(), hm.values.shape)
peak_v = hm.values[peak_m, peak_h]
ax.text(peak_h, peak_m + 1, f"★\n{peak_v:.0f}", ha="center", va="center",
        fontsize=8, color="white", fontweight="bold")

# Annotate each cell with value
for i in range(12):
    for j in range(24):
        v = hm.values[i, j]
        col = "white" if (v > 105 or v < 73) else "#333"
        ax.text(j, i + 1, f"{v:.0f}", ha="center", va="center",
                fontsize=6.5, color=col)

cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("GW", fontsize=9)

fig.tight_layout()
fig.savefig("fig3_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig3_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Trend, COVID, year-over-year
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig4_trend_covid.png …")
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("PJM RTO Load – Temporal Trends", fontsize=13, fontweight="bold")

# 4a – 10-year monthly trend (full width top)
ax0 = fig.add_subplot(gs[0, :])
ax0.fill_between(monthly_ts.index, monthly_ts.values, alpha=0.15, color=BLUE)
ax0.plot(monthly_ts.index, monthly_ts.values, color=BLUE, lw=1.4)
ax0.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
            color=RED, alpha=0.12, label="COVID period (Mar–Jul 2020)")
ax0.set_title("10-year monthly average load, 2016–2025")
ax0.set_ylabel("GW")
ax0.set_ylim(65, 120)
ax0.legend(fontsize=9)
# Mark 2025 average
ax0.annotate(f"2025 avg: {yoy.loc[2025,'mean']:.1f} GW (decade high)",
             xy=(pd.Timestamp("2025-06-01"), yoy.loc[2025,"mean"]),
             xytext=(pd.Timestamp("2023-01-01"), 107),
             fontsize=8, color=TEAL,
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

# 4b – COVID: 2019 vs 2020 monthly
ax1 = fig.add_subplot(gs[1, 0])
x = np.arange(12)
w = 0.38
ax1.bar(x - w/2, covid_2019.values, width=w, color=BLUE,  alpha=0.8, label="2019", edgecolor="white")
ax1.bar(x + w/2, covid_2020.values, width=w, color=RED,   alpha=0.8, label="2020", edgecolor="white")
ax1.set_title("COVID impact: 2019 vs 2020")
ax1.set_ylabel("GW")
ax1.set_ylim(65, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(MONTHS_SHORT)
ax1.legend(fontsize=9)
# Annotate worst drop
worst_m = int((covid_2019.values - covid_2020.values).argmax())
drop = covid_2019.values[worst_m] - covid_2020.values[worst_m]
ax1.annotate(f"–{drop:.1f} GW\n({drop/covid_2019.values[worst_m]*100:.1f}%)",
             xy=(worst_m + w/2, covid_2020.values[worst_m]),
             xytext=(worst_m + 1.2, covid_2020.values[worst_m] + 3),
             fontsize=8, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

# 4c – Year-over-year peak / mean / min
ax2 = fig.add_subplot(gs[1, 1])
years = list(yoy.index)
ax2.plot(years, yoy["max"].values,  color=RED,  lw=2, marker="o", ms=4, label="Peak")
ax2.plot(years, yoy["mean"].values, color=BLUE, lw=2, marker="s", ms=4, label="Mean", linestyle="--")
ax2.plot(years, yoy["min"].values,  color=TEAL, lw=2, marker="^", ms=4, label="Minimum", linestyle=":")
ax2.set_title("Year-over-year: peak, mean, minimum")
ax2.set_ylabel("GW")
ax2.set_xticks(years)
ax2.set_xticklabels([str(y) for y in years], rotation=30)
ax2.legend(fontsize=9)
# COVID dip arrow
ax2.annotate("COVID\ndip", xy=(2020, yoy.loc[2020,"mean"]),
             xytext=(2021, yoy.loc[2020,"mean"] - 5),
             fontsize=8, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

fig.savefig("fig4_trend_covid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig4_trend_covid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Signal structure: ACF + ramp-rate
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig5_signal.png …")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("PJM RTO Load – Signal Structure", fontsize=13, fontweight="bold", y=1.01)

# 5a – ACF
key_lags = {24: "24 h\n(daily)", 48: "48 h", 168: "168 h\n(weekly)"}
bar_colors = [RED if lag in key_lags else BLUE + "99" for lag in range(1, 169)]
axes[0].bar(range(1, 169), acf_vals, color=bar_colors, width=1, alpha=0.75)
for lag, label in key_lags.items():
    axes[0].axvline(lag, color=RED, lw=1, linestyle="--", alpha=0.6)
    axes[0].text(lag + 1, 0.55, label, color=RED, fontsize=8, va="bottom")
axes[0].set_title("Autocorrelation function (lags 1–168 h)")
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("ACF")
axes[0].set_ylim(0, 1.05)
axes[0].set_xlim(0, 170)

# 5b – Ramp-rate histogram
ramps = rto["ramp"].dropna()
axes[1].hist(ramps, bins=100, color=BLUE, alpha=0.7, edgecolor="none", range=(-12, 12))
axes[1].axvline(0, color="black", lw=0.8)
p5, p95 = np.percentile(ramps, 5), np.percentile(ramps, 95)
axes[1].axvline(p5,  color=RED, lw=1.2, linestyle="--", label=f"P5  = {p5:.1f} GW/h")
axes[1].axvline(p95, color=RED, lw=1.2, linestyle="--", label=f"P95 = {p95:.1f} GW/h")
axes[1].set_title("Hourly ramp-rate distribution")
axes[1].set_xlabel("ΔLoad (GW/h)")
axes[1].set_ylabel("Count")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig("fig5_signal.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig5_signal.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Load vs temperature V-curve
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting fig6_vcurve.png …")
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("PJM RTO Load – Temperature Sensitivity (V-curve)", fontsize=13, fontweight="bold")

# Scatter sample (thin out to avoid overplotting)
sample = merged.sample(min(20000, len(merged)), random_state=42)
ax.scatter(sample["RTO"], sample["mw"] / 1000,
           alpha=0.04, s=3, color=GRAY, rasterized=True)

# Binned mean line
ax.plot(vcurve.index, vcurve.values, color=RED, lw=2.5, zorder=5, label="Bin mean (5°F bins)")
ax.scatter(vcurve.index, vcurve.values, color=RED, s=30, zorder=6)

# Comfort zone band
ax.axvspan(60, 65, color=TEAL, alpha=0.12, label="Comfort zone (60–65°F)")
ax.axvline(62, color=TEAL, lw=1.2, linestyle="--")

# Annotate minimum
min_bin = vcurve.idxmin()
min_v   = vcurve.min()
ax.annotate(f"Min load\n{min_v:.1f} GW @ {min_bin:.0f}°F",
            xy=(min_bin, min_v), xytext=(min_bin + 5, min_v + 5),
            fontsize=9, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))

# Annotate hot extreme
ax.annotate(f"{vcurve.iloc[-1]:.0f} GW @ {vcurve.index[-1]:.0f}°F",
            xy=(vcurve.index[-1], vcurve.iloc[-1]),
            xytext=(vcurve.index[-1] - 12, vcurve.iloc[-1] - 8),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

ax.set_xlabel("Temperature (°F) — RTO population-weighted average")
ax.set_ylabel("Load (GW)")
ax.set_xlim(0, 100)
ax.set_ylim(60, 155)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("fig6_vcurve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved: fig6_vcurve.png")

print("\nAll 6 figures saved.")