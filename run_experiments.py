"""Runs every experiment reported in the paper and writes results/*.csv and figures/*.pdf.
All data is SYNTHETIC (see generator_settings.json). Usage: python run_experiments.py"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import waitsim as w

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
S = w.load_settings()
SEEDS = list(range(1, 11))          # 10 independent synthetic datasets
MODELS = ["Linear Regression", "Ridge (alpha=1)", "Random Forest"]
BASES = ["B0: Global mean", "B1: Meal-window mean", "B2: Day x 30-min bin mean"]
NULL_WORLD = {"surge_amplitude_min": 0.0, "queue_probs_surge": S["queue_probs_normal"]}

def data(seed, **ov):
    df, log = w.clean(w.generate(S, seed, **ov))
    return df, log

# ---- cleaning summary + legacy files
logs = pd.DataFrame([data(s)[1] for s in SEEDS])
logs.to_csv("results/cleaning_log.csv", index=False)
real, real_log = w.clean(pd.read_csv("wait_log.csv"))
cp2, cp2_log = w.clean(pd.read_csv("DEMO_wait_log_SYNTHETIC.csv"))
json.dump({"real_form_tests": real_log, "cp2_synthetic": cp2_log,
           "generated_mean": logs.mean().round(1).to_dict()}, open("results/cleaning_summary.json", "w"), indent=1)

# ---- E1: main comparison, grouped vs shuffled CV
e1 = []
for s in SEEDS:
    df, _ = data(s)
    for grouped in (True, False):
        r = w.evaluate(df, grouped=grouped).reset_index()
        r["seed"], r["cv"], r["n"] = s, "grouped" if grouped else "shuffled", len(df)
        e1.append(r)
e1 = pd.concat(e1); e1.to_csv("results/e1_main.csv", index=False)
e1s = e1.groupby(["cv", "method"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(2)
e1s.to_csv("results/e1_summary.csv"); print(e1s)

# ---- E2: dismissal-feature ablation in a surge world and a no-surge (null) world
e2 = []
for world, ov in [("surge", {}), ("no_surge", NULL_WORLD)]:
    for s in SEEDS:
        df, _ = data(s, **ov)
        for use in (True, False):
            r = w.evaluate(df, use_dismissal=use, include_baselines=use).reset_index()
            r["seed"], r["world"], r["dismissal_feature"] = s, world, use
            e2.append(r)
e2 = pd.concat(e2); e2.to_csv("results/e2_ablation.csv", index=False)
e2s = (e2[e2.method.isin(MODELS + BASES)]
       .groupby(["world", "dismissal_feature", "method"])["MAE"].agg(["mean", "std"]).round(2))
e2s.to_csv("results/e2_summary.csv"); print(e2s)

# ---- E3: wrong class schedule. True dismissals shifted; pipeline keeps assumed times.
e3 = []
for shift in [0, 5, 10, 15, 20, 30]:
    for s in SEEDS:
        df, _ = data(s, schedule_shift_min=shift)
        on = w.evaluate(df, use_dismissal=True, include_baselines=False)["MAE"]
        off = w.evaluate(df, use_dismissal=False, include_baselines=False)["MAE"]
        for m in MODELS:
            e3.append({"shift": shift, "seed": s, "method": m,
                       "MAE_with": on[m], "MAE_without": off[m], "gain": off[m] - on[m]})
e3 = pd.DataFrame(e3); e3.to_csv("results/e3_schedule_shift.csv", index=False)
e3s = e3.groupby(["shift", "method"])[["MAE_with", "MAE_without", "gain"]].agg(["mean", "std"]).round(2)
e3s.to_csv("results/e3_summary.csv"); print(e3s)

# ---- E4: how many observations are needed? (subsample a large synthetic pool)
e4 = []
rng = np.random.default_rng(2027)
for s in range(1, 21):
    pool, _ = data(100 + s, n_weekdays=40)
    for n in [30, 50, 75, 100, 150, 200, 300]:
        sub = pool.sample(n=n, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        if sub["service_date"].nunique() < 5:
            continue
        r = w.evaluate(sub)["MAE"]
        bb, bm = r[BASES].min(), r[MODELS].min()
        e4.append({"n": n, "seed": s, "best_baseline": bb, "best_model": bm,
                   "reduction": 1 - bm / bb, "target_met": (bm < 3.0) and (bm <= 0.8 * bb)})
e4 = pd.DataFrame(e4); e4.to_csv("results/e4_sample_size.csv", index=False)
e4s = e4.groupby("n").agg(best_baseline=("best_baseline", "mean"), best_model=("best_model", "mean"),
                          model_sd=("best_model", "std"), reduction=("reduction", "mean"),
                          target_met_rate=("target_met", "mean"), runs=("seed", "count")).round(3)
e4s.to_csv("results/e4_summary.csv"); print(e4s)

# ---------------------------------------------------------------- figures
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False})

# Figure 1: MAE by method, grouped CV
g = e1[e1.cv == "grouped"].groupby("method")["MAE"].agg(["mean", "std"]).loc[BASES + MODELS]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
lbl = ["B0 Global mean", "B1 Meal-window", "B2 Day x 30-min bin", "Linear", "Ridge", "Random Forest"]
ax.barh(range(6), g["mean"], xerr=g["std"], color=[GRAY] * 3 + [BLUE] * 3, height=0.62,
        error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 2})
ax.set_yticks(range(6), lbl)
ax.axvline(3.0, ls="--", lw=0.8, color=ORANGE)
ax.text(3.0, -0.75, "3.0-min target", color=INK, fontsize=7, ha="center", va="bottom")
ax.set_ylim(5.5, -1.0)
ax.set_xlabel("MAE (minutes)")
fig.tight_layout(); fig.savefig("figures/fig_main.pdf"); plt.close(fig)

# Figure 2: gain from dismissal feature vs schedule error (Ridge, RF)
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for m, c, mk in [("Ridge (alpha=1)", BLUE, "o"), ("Random Forest", ORANGE, "s")]:
    d = e3[e3.method == m].groupby("shift")["gain"].agg(["mean", "std"])
    ax.errorbar(d.index, d["mean"], yerr=d["std"], color=c, marker=mk, ms=4, lw=1.5, capsize=2,
                label=m.replace(" (alpha=1)", ""))
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Error in assumed class end times (minutes)")
ax.set_ylabel("MAE reduction (min)")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig_shift.pdf"); plt.close(fig)

# Figure 3: sample size
d = e4.groupby("n")[["best_baseline", "best_model"]].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ax.errorbar(d.index, d[("best_baseline", "mean")], yerr=d[("best_baseline", "std")], color=GRAY,
            marker="s", ms=4, lw=1.5, capsize=2, label="Best baseline")
ax.errorbar(d.index, d[("best_model", "mean")], yerr=d[("best_model", "std")], color=BLUE,
            marker="o", ms=4, lw=1.5, capsize=2, label="Best model")
ax.set_xlabel("Number of observations"); ax.set_ylabel("MAE (minutes)")
ax.set_ylim(bottom=0); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig_samples.pdf"); plt.close(fig)
print("done")
