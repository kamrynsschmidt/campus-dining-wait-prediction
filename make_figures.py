"""Makes the four paper figures as 300-dpi PNGs (PNG avoids ACMSE's font rule for embedded PDFs).

Run AFTER the experiments, from the package folder (or just run: python run_all.py):
    python run_experiments.py && python run_e5.py && python run_e6.py
    python make_figures.py
In Google Colab, first run:  !apt-get -qq install fonts-linuxlibertine
(optional: gives the figures the same Libertine font as the paper; falls back to a serif font otherwise)

Outputs: figures/fig_main.png, figures/fig_shift.png, figures/fig_samples.png, figures/fig_scatter.png (the figures used in the paper)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import GroupKFold
import waitsim as w

# ---- style
font_manager._load_fontmanager(try_read_cache=False)
fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O"] if "Linux Libertine O" in fonts else ["DejaVu Serif"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "legend.frameon": False, "savefig.dpi": 300,
})
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
BASES = ["B0: Global mean", "B1: Meal-window mean", "B2: Day x 30-min bin mean"]
MODELS = ["Linear Regression", "Ridge (alpha=1)", "Random Forest"]
EB = dict(ms=4, lw=1.5, capsize=2)
SIZE = (3.3, 1.9)   # fits one ACM column

# ---- Figure: error by method (grouped CV, mean +/- SD over 10 synthetic datasets)
e1 = pd.read_csv("results/e1_main.csv")
g = e1[e1.cv == "grouped"].groupby("method")["MAE"].agg(["mean", "std"]).loc[BASES + MODELS]
fig, ax = plt.subplots(figsize=SIZE)
ax.barh(range(6), g["mean"], xerr=g["std"], color=[GRAY] * 3 + [BLUE] * 3, height=0.62,
        error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 2})
ax.set_yticks(range(6), ["B0 Global mean", "B1 Meal-window", "B2 Day x 30-min bin",
                         "Linear", "Ridge", "Random Forest"])
ax.axvline(3.0, ls="--", lw=0.8, color=ORANGE)
ax.text(3.0, -0.75, "3.0-min target", color=INK, fontsize=7, ha="center", va="bottom")
ax.set_ylim(5.5, -1.0)
ax.set_xlabel("MAE (minutes)")
fig.tight_layout(); fig.savefig("figures/fig_main.png"); plt.close(fig)

# ---- Figure: gain from dismissal feature vs. schedule error (Random Forest)
e5 = pd.read_csv("results/e5_noqueue_shift.csv")
fig, ax = plt.subplots(figsize=SIZE)
for setting, c, mk, lab in [("no_queue", BLUE, "o", "Without queue report"),
                            ("with_queue", ORANGE, "s", "With queue report")]:
    d = (e5[(e5.method == "Random Forest") & (e5.setting == setting)]
         .groupby("shift")["gain"].agg(["mean", "std"]))
    ax.errorbar(d.index, d["mean"], yerr=d["std"], color=c, marker=mk, label=lab, **EB)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Error in assumed class end times (minutes)")
ax.set_ylabel("MAE reduction (min)")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig_shift.png"); plt.close(fig)

# ---- Figure: error vs. number of observations
e4 = pd.read_csv("results/e4_sample_size.csv")
d = e4.groupby("n")[["best_baseline", "best_model"]].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=SIZE)
ax.errorbar(d.index, d[("best_baseline", "mean")], yerr=d[("best_baseline", "std")],
            color=GRAY, marker="s", label="Best baseline", **EB)
ax.errorbar(d.index, d[("best_model", "mean")], yerr=d[("best_model", "std")],
            color=BLUE, marker="o", label="Best model", **EB)
ax.set_xlabel("Number of observations"); ax.set_ylabel("MAE (minutes)")
ax.set_ylim(bottom=0); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig_samples.png"); plt.close(fig)

# ---- Figure: predicted vs. generated waits, Random Forest, synthetic dataset seed 1
df, _ = w.clean(w.generate(w.load_settings(), 1))
models, feats = w.make_models(True)
rf = models["Random Forest"]
pred = pd.Series(index=df.index, dtype=float)
for tr, te in GroupKFold(5).split(df, groups=df["service_date"]):
    rf.fit(df.iloc[tr][feats], df.iloc[tr]["wait"])
    pred.iloc[te] = rf.predict(df.iloc[te][feats])
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ax.scatter(df["wait"], pred, s=10, alpha=0.6, color=BLUE, edgecolor="white", linewidth=0.3)
lim = [0, max(df["wait"].max(), pred.max()) * 1.05]
ax.plot(lim, lim, ls="--", lw=0.8, color=MUTED)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Generated wait (minutes)"); ax.set_ylabel("Predicted wait (minutes)")
fig.tight_layout(); fig.savefig("figures/fig_scatter.png"); plt.close(fig)

print("Saved figures/fig_main.png, fig_shift.png, fig_samples.png, fig_scatter.png")
