"""
research_history/scripts/eda_exploration.py
==============================================
FROZEN HISTORICAL RECORD — merged from the original week1_testing.py
(despite its name, an EDA script, not a pytest suite) and eda_plots.py
per the Phase 1 repository audit: both are independent explorations of
the same raw PJM RTO load data, forked from a common ancestor, each
producing a different but overlapping set of figures. Neither is part
of the maintained package (see src/powerdiffusion/, and
examples/exploratory_data_analysis.py for a small demonstration against
the processed dataset).

Each part below is wrapped in its own function and kept internally
verbatim from the original script — including one real discrepancy
between them that's worth flagging rather than silently harmonizing:
  - `run_signal_structure_eda` (week1_testing.py) tz-converts the
    temperature timestamps to US/Eastern before joining with load data.
  - `run_pattern_eda` (eda_plots.py) tz-localizes them directly from
    UTC without the US/Eastern conversion step.
  - The two scripts also visualize the COVID exclusion window
    differently from what `preprocess.py` actually excludes — see the
    Phase 1 audit's "Cross-file duplication summary", item 8.
These were not reconciled when merging, since the correction belongs
to a live pipeline decision, not to a frozen historical record; do not
copy this loading code into new work without checking which behavior
you actually want. Run from the project root (the original CSVs
`hrl_load_metered_*.csv` and `out/pjm_hourly_temps_wide.csv` are no
longer at those paths — see data/raw/ instead — so this needs path
updates to actually run).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

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


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — week1_testing.py
# "PJM Hourly Load – Exploratory Data Analysis": descriptive stats plus
# 4 figures (time series overview, temporal patterns, temperature/zone
# analysis, signal structure) — the earlier, broader-scope EDA pass.
# ═══════════════════════════════════════════════════════════════════════

def run_signal_structure_eda():
    BLUE  = "#3266ad"
    AMBER = "#e09320"
    RED   = "#c94040"
    TEAL  = "#1d9e75"
    GRAY  = "#888780"

    print("Loading data …")
    dfs = []
    for yr in range(2016, 2026):
        df = pd.read_csv(f"hrl_load_metered_{yr}.csv")
        df["year"] = yr
        dfs.append(df)
    all_load = pd.concat(dfs, ignore_index=True)

    all_load["dt"] = pd.to_datetime(all_load["datetime_beginning_ept"])
    all_load["hour"]       = all_load["dt"].dt.hour
    all_load["dow"]        = all_load["dt"].dt.dayofweek
    all_load["month"]      = all_load["dt"].dt.month
    all_load["is_weekend"] = all_load["dow"].isin([5, 6])

    rto = all_load[all_load["zone"] == "RTO"].copy().sort_values("dt").reset_index(drop=True)
    print(f"  RTO rows: {len(rto):,}  |  all-zone rows: {len(all_load):,}")

    print("Loading temperature data …")
    temp = pd.read_csv("out/pjm_hourly_temps_wide.csv", usecols=["datetime_ept", "RTO"])
    temp["dt"] = pd.to_datetime(temp["datetime_ept"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)
    temp_rto = temp[["dt", "RTO"]].rename(columns={"RTO": "temp_f"})

    rto_with_temp = rto.merge(temp_rto, on="dt", how="left")

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

    # ── Figure 1: Time-series overview ─────────────────────────────────
    print("\nPlotting Figure 1 – time series overview …")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("PJM RTO Load – Time Series Overview", fontsize=14, fontweight="bold", y=0.99)

    monthly_ts = rto.groupby(rto["dt"].dt.to_period("M"))["mw"].mean()
    monthly_ts.index = monthly_ts.index.to_timestamp()
    axes[0].fill_between(monthly_ts.index, monthly_ts.values / 1000, alpha=0.25, color=BLUE)
    axes[0].plot(monthly_ts.index, monthly_ts.values / 1000, color=BLUE, lw=1.5)
    axes[0].set_title("Monthly average load, 2016–2025", fontsize=11)
    axes[0].set_ylabel("GW")
    axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
                    color=RED, alpha=0.1, label="COVID period")
    axes[0].legend(fontsize=9)

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

    # ── Figure 2: Temporal patterns ─────────────────────────────────────
    print("Plotting Figure 2 – temporal patterns …")
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("PJM RTO Load – Temporal Patterns", fontsize=14, fontweight="bold")

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

    ax2 = fig.add_subplot(gs[0, 1])
    monthly_vals = rto.groupby("month")["mw"].mean() / 1000
    colors_month = [BLUE if m in [12,1,2] else AMBER if m in [6,7,8] else GRAY
                    for m in monthly_vals.index]
    ax2.bar(month_names, monthly_vals.values, color=colors_month, alpha=0.75, edgecolor="white")
    ax2.set_title("Seasonal pattern (monthly avg)", fontsize=11)
    ax2.set_ylabel("GW")
    ax2.set_xticklabels(month_names, rotation=45)
    ax2.legend(handles=[Patch(color=BLUE, alpha=0.75, label="Winter"),
                        Patch(color=AMBER, alpha=0.75, label="Summer"),
                        Patch(color=GRAY, alpha=0.75, label="Shoulder")], fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_vals  = rto.groupby("dow")["mw"].mean() / 1000
    dow_colors = [BLUE if d < 5 else AMBER for d in range(7)]
    ax3.bar(dow_names, dow_vals.values, color=dow_colors, alpha=0.75, edgecolor="white")
    ax3.set_title("Day-of-week pattern", fontsize=11)
    ax3.set_ylabel("GW")

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

    # ── Figure 3: Load-temperature V-curve + zone correlations ──────────
    print("Plotting Figure 3 – temperature & zone analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PJM – Temperature Sensitivity & Zone Correlations", fontsize=14, fontweight="bold")

    merged_valid = rto_with_temp.dropna(subset=["temp_f"])
    sample = merged_valid.sample(min(15000, len(merged_valid)), random_state=42)
    axes[0].scatter(sample["temp_f"], sample["mw"] / 1000,
                    alpha=0.05, s=4, color=GRAY, rasterized=True)
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

    # ── Figure 4: Autocorrelation + load duration curve + ramps ─────────
    print("Plotting Figure 4 – ACF, load duration, ramps …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PJM RTO – Signal Structure", fontsize=14, fontweight="bold")

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


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — eda_plots.py
# "PJM RTO Load – EDA Plots": 6 publication-ready figures matching an
# interactive dashboard (patterns, hourly profiles, heatmap, trend/COVID,
# signal structure, V-curve) — the later, presentation-oriented EDA pass.
# ═══════════════════════════════════════════════════════════════════════

def run_pattern_eda():
    plt.rcParams.update({
        "font.family": "sans-serif",
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

    print("Loading data …")
    dfs = []
    for yr in range(2016, 2026):
        df = pd.read_csv(f"hrl_load_metered_{yr}.csv")
        df["year"] = yr
        dfs.append(df)
    all_load = pd.concat(dfs, ignore_index=True)
    all_load["dt"]         = pd.to_datetime(all_load["datetime_beginning_ept"], format="mixed")
    all_load["hour"]       = all_load["dt"].dt.hour
    all_load["dow"]        = all_load["dt"].dt.dayofweek
    all_load["month"]      = all_load["dt"].dt.month
    all_load["is_weekend"] = all_load["dow"].isin([5, 6])
    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    all_load["season"] = all_load["month"].map(season_map)

    rto = all_load[all_load["zone"] == "RTO"].copy().sort_values("dt").reset_index(drop=True)
    print(f"  RTO rows: {len(rto):,}")

    print("Loading temperature data …")
    temp = pd.read_csv("out/pjm_hourly_temps_wide.csv", usecols=["datetime_ept", "RTO"])
    temp["dt"] = pd.to_datetime(temp["datetime_ept"], utc=True).dt.tz_localize(None)
    rto_t = rto[["dt","mw"]].copy()
    merged = rto_t.merge(temp[["dt","RTO"]], on="dt", how="left").dropna(subset=["RTO"])
    merged["tbin"] = pd.cut(merged["RTO"], bins=range(0, 105, 5),
                            labels=range(2, 102, 5)).astype(float)
    vcurve = merged.groupby("tbin", observed=True)["mw"].mean() / 1000

    dow_avg    = rto.groupby("dow")["mw"].mean() / 1000
    month_avg  = rto.groupby("month")["mw"].mean() / 1000
    season_avg = rto.groupby("season")["mw"].mean() / 1000
    wd_hr      = rto[~rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
    we_hr      = rto[ rto["is_weekend"]].groupby("hour")["mw"].mean() / 1000
    season_hr  = {s: rto[rto["season"]==s].groupby("hour")["mw"].mean() / 1000
                  for s in SEASONS}
    hm         = rto.groupby(["month","hour"])["mw"].mean().unstack(level=1) / 1000
    yoy        = rto.groupby("year")["mw"].agg(["mean","max","min"]) / 1000
    monthly_ts = rto.groupby(rto["dt"].dt.to_period("M"))["mw"].mean()
    monthly_ts.index = monthly_ts.index.to_timestamp()
    covid_2019 = rto[rto["year"]==2019].groupby("month")["mw"].mean() / 1000
    covid_2020 = rto[rto["year"]==2020].groupby("month")["mw"].mean() / 1000
    rto["ramp"] = rto["mw"].diff() / 1000
    s_rto = rto.set_index("dt")["mw"].sort_index() / 1000
    acf_vals = [s_rto.autocorr(lag) for lag in range(1, 169)]

    # ── Fig 1 – Patterns: DoW, season, monthly ───────────────────────────
    print("Plotting fig1_patterns.png …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("PJM RTO Load – Key Patterns (2016–2025)", fontsize=13, fontweight="bold", y=1.01)

    dow_colors = [BLUE if d < 5 else AMBER for d in range(7)]
    axes[0].bar(DOW_NAMES, dow_avg.values, color=dow_colors, alpha=0.85, edgecolor="white", width=0.65)
    axes[0].set_title("Average load by day of week")
    axes[0].set_ylabel("GW")
    axes[0].set_ylim(80, 96)
    for i, (name, val) in enumerate(zip(DOW_NAMES, dow_avg.values)):
        axes[0].text(i, val + 0.15, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(handles=[mpatches.Patch(color=BLUE, alpha=0.85, label="Weekday"),
                             mpatches.Patch(color=AMBER, alpha=0.85, label="Weekend")], fontsize=8)

    s_colors = [TEAL, RED, AMBER, BLUE]
    s_vals   = [float(season_avg[s]) for s in SEASONS]
    axes[1].bar(SEASONS, s_vals, color=s_colors, alpha=0.85, edgecolor="white", width=0.6)
    axes[1].set_title("Average load by season")
    axes[1].set_ylabel("GW")
    axes[1].set_ylim(70, 110)
    for i, (s, v) in enumerate(zip(SEASONS, s_vals)):
        axes[1].text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

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

    # ── Fig 2 – Hourly profiles ───────────────────────────────────────────
    print("Plotting fig2_hourly_profiles.png …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("PJM RTO Load – Hourly Profiles", fontsize=13, fontweight="bold", y=1.01)

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

    # ── Fig 3 – Heatmap (month × hour) ────────────────────────────────────
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

    peak_m, peak_h = np.unravel_index(hm.values.argmax(), hm.values.shape)
    peak_v = hm.values[peak_m, peak_h]
    ax.text(peak_h, peak_m + 1, f"★\n{peak_v:.0f}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

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

    # ── Fig 4 – Trend, COVID, year-over-year ─────────────────────────────
    print("Plotting fig4_trend_covid.png …")
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("PJM RTO Load – Temporal Trends", fontsize=13, fontweight="bold")

    ax0 = fig.add_subplot(gs[0, :])
    ax0.fill_between(monthly_ts.index, monthly_ts.values, alpha=0.15, color=BLUE)
    ax0.plot(monthly_ts.index, monthly_ts.values, color=BLUE, lw=1.4)
    ax0.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
                color=RED, alpha=0.12, label="COVID period (Mar–Jul 2020)")
    ax0.set_title("10-year monthly average load, 2016–2025")
    ax0.set_ylabel("GW")
    ax0.set_ylim(65, 120)
    ax0.legend(fontsize=9)
    ax0.annotate(f"2025 avg: {yoy.loc[2025,'mean']:.1f} GW (decade high)",
                 xy=(pd.Timestamp("2025-06-01"), yoy.loc[2025,"mean"]),
                 xytext=(pd.Timestamp("2023-01-01"), 107),
                 fontsize=8, color=TEAL,
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

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
    worst_m = int((covid_2019.values - covid_2020.values).argmax())
    drop = covid_2019.values[worst_m] - covid_2020.values[worst_m]
    ax1.annotate(f"–{drop:.1f} GW\n({drop/covid_2019.values[worst_m]*100:.1f}%)",
                 xy=(worst_m + w/2, covid_2020.values[worst_m]),
                 xytext=(worst_m + 1.2, covid_2020.values[worst_m] + 3),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

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
    ax2.annotate("COVID\ndip", xy=(2020, yoy.loc[2020,"mean"]),
                 xytext=(2021, yoy.loc[2020,"mean"] - 5),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    fig.savefig("fig4_trend_covid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: fig4_trend_covid.png")

    # ── Fig 5 – Signal structure: ACF + ramp-rate ─────────────────────────
    print("Plotting fig5_signal.png …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("PJM RTO Load – Signal Structure", fontsize=13, fontweight="bold", y=1.01)

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

    # ── Fig 6 – Load vs temperature V-curve ───────────────────────────────
    print("Plotting fig6_vcurve.png …")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("PJM RTO Load – Temperature Sensitivity (V-curve)", fontsize=13, fontweight="bold")

    sample = merged.sample(min(20000, len(merged)), random_state=42)
    ax.scatter(sample["RTO"], sample["mw"] / 1000,
               alpha=0.04, s=3, color=GRAY, rasterized=True)

    ax.plot(vcurve.index, vcurve.values, color=RED, lw=2.5, zorder=5, label="Bin mean (5°F bins)")
    ax.scatter(vcurve.index, vcurve.values, color=RED, s=30, zorder=6)

    ax.axvspan(60, 65, color=TEAL, alpha=0.12, label="Comfort zone (60–65°F)")
    ax.axvline(62, color=TEAL, lw=1.2, linestyle="--")

    min_bin = vcurve.idxmin()
    min_v   = vcurve.min()
    ax.annotate(f"Min load\n{min_v:.1f} GW @ {min_bin:.0f}°F",
                xy=(min_bin, min_v), xytext=(min_bin + 5, min_v + 5),
                fontsize=9, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))

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


if __name__ == "__main__":
    run_signal_structure_eda()
    run_pattern_eda()
