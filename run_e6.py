"""E6: error by queue category (weak-case analysis) and predicted-vs-actual figure. SYNTHETIC."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import waitsim as w
S = w.load_settings(); rows = []; scatter = None
for s in range(1, 11):
    df, _ = w.clean(w.generate(S, s))
    models, feats = w.make_models(True)
    rf = models["Random Forest"]; p = pd.Series(index=df.index, dtype=float); b1 = p.copy()
    for tr, te in GroupKFold(5).split(df, groups=df["service_date"]):
        rf.fit(df.iloc[tr][feats], df.iloc[tr]["wait"]); p.iloc[te] = rf.predict(df.iloc[te][feats])
        b1.iloc[te] = w.baselines(df.iloc[tr], df.iloc[te])["B1: Meal-window mean"].values
    d = df.assign(pred=p, b1=b1)
    for q, g in d.groupby("queue_ordinal"):
        rows.append(dict(seed=s, queue="ABCD"[q], n=len(g), rf_mae=(g.wait - g.pred).abs().mean(),
                         b1_mae=(g.wait - g.b1).abs().mean(), mean_wait=g.wait.mean(),
                         rf_bias=(g.pred - g.wait).mean()))
    if s == 1: scatter = d
e6 = pd.DataFrame(rows); e6.to_csv("results/e6_by_queue.csv", index=False)
print(e6.groupby("queue")[["n", "mean_wait", "rf_mae", "b1_mae", "rf_bias"]].mean().round(2))
top = scatter.assign(err=(scatter.wait - scatter.pred).abs()).nlargest(5, "err")
print(top[["line_join_time", "visual_queue_category", "minutes_post_class_dismissal", "wait", "pred"]])
# share of error in surge vs not
sc = scatter.assign(err=(scatter.wait - scatter.pred).abs(),
                    surge=scatter.minutes_post_class_dismissal.between(0, 20))
print(sc.groupby("surge").err.agg(["mean", "count"]).round(2))
BLUE, MUTED = "#2a78d6", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ax.scatter(scatter.wait, scatter.pred, s=10, alpha=0.6, color=BLUE, edgecolor="white", linewidth=0.3)
lim = [0, max(scatter.wait.max(), scatter.pred.max()) * 1.05]
ax.plot(lim, lim, ls="--", lw=0.8, color=MUTED); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Generated wait (minutes)"); ax.set_ylabel("Predicted wait (minutes)")
fig.tight_layout(); fig.savefig("figures/fig_scatter.pdf"); plt.close(fig)
