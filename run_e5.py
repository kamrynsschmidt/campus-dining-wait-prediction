"""E5: forecasting WITHOUT a queue report (time + dismissal features only), across
errors in the assumed class schedule. SYNTHETIC data. Also redraws Figure 2."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import waitsim as w
S = w.load_settings(); SEEDS = range(1, 11)

def mae(df, num, model):
    cat = ["day_of_week", "meal_period"]; feats = num + cat
    pre = ColumnTransformer([("num", StandardScaler(), num) if num else ("num", "drop", []),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    p = pd.Series(index=df.index, dtype=float)
    for tr, te in GroupKFold(5).split(df, groups=df["service_date"]):
        m = Pipeline([("pre", pre), ("m", model())]).fit(df.iloc[tr][feats], df.iloc[tr]["wait"])
        p.iloc[te] = m.predict(df.iloc[te][feats])
    return mean_absolute_error(df["wait"], p)

RF = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=365, n_jobs=-1)
RG = lambda: Ridge(alpha=1.0)
rows = []
for shift in [0, 5, 10, 15, 20, 30]:
    for s in SEEDS:
        df, _ = w.clean(w.generate(S, s, schedule_shift_min=shift))
        for mname, M in [("Random Forest", RF), ("Ridge (alpha=1)", RG)]:
            for setting, base in [("with_queue", ["queue_ordinal"]), ("no_queue", [])]:
                on = mae(df, base + ["minutes_post_class_dismissal"], M)
                off = mae(df, base, M)
                rows.append(dict(shift=shift, seed=s, method=mname, setting=setting,
                                 MAE_with=on, MAE_without=off, gain=off - on))
e5 = pd.DataFrame(rows); e5.to_csv("results/e5_noqueue_shift.csv", index=False)
e5s = e5.groupby(["setting", "method", "shift"])[["MAE_with", "MAE_without", "gain"]].agg(["mean", "std"]).round(2)
e5s.to_csv("results/e5_summary.csv"); print(e5s)

BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for setting, c, mk, lab in [("no_queue", BLUE, "o", "Without queue report"),
                            ("with_queue", ORANGE, "s", "With queue report")]:
    d = e5[(e5.method == "Random Forest") & (e5.setting == setting)].groupby("shift")["gain"].agg(["mean", "std"])
    ax.errorbar(d.index, d["mean"], yerr=d["std"], color=c, marker=mk, ms=4, lw=1.5, capsize=2, label=lab)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Error in assumed class end times (minutes)")
ax.set_ylabel("MAE reduction (min)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig_shift.pdf"); plt.close(fig)
