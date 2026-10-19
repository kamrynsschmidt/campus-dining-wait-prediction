"""Synthetic generator, cleaning pipeline, baselines and models for the
Williams Dining Hall wait-time project.

Everything produced by `generate()` is SYNTHETIC. The generator encodes the
author's assumptions (generator_settings.json); models that fit it well are
recovering those assumptions, not measuring the real dining hall.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------- configuration
# Assumed (NOT verified) class end times and meal hours, unchanged from CP1/CP2.
CLASS_END_TIMES = {
    "MWF": ["08:50", "09:50", "10:50", "11:50", "12:50", "13:50", "14:50", "15:50", "16:50"],
    "TR":  ["09:15", "10:45", "12:15", "13:45", "15:15", "16:45", "18:15"],
}
DAY_PATTERN = {0: "MWF", 1: "TR", 2: "MWF", 3: "TR", 4: "MWF"}
MEAL_PERIODS = [("Breakfast", "07:00", "10:30"), ("Lunch", "10:30", "14:00"),
                ("Late Lunch", "14:00", "16:30"), ("Dinner", "16:30", "20:00")]
QUEUE_LABELS = ["A: Inside turnstiles (0-5 people)",
                "B: Back to the double doors (6-15 people)",
                "C: Past double doors into lobby (16-30 people)",
                "D: Outside the entrance (30+ people)"]
QUEUE_ORDER = {q: i for i, q in enumerate(QUEUE_LABELS)}
MAX_WAIT = 90
WAIT_TOLERANCE_MIN = 0.5   # supplied wait must agree with timestamps within this


def load_settings(path="generator_settings.json"):
    with open(path) as f:
        s = json.load(f)
    s.pop("_comment", None)
    return s


def _to_min(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def meal_period(ts):
    t = ts.hour * 60 + ts.minute
    for name, start, end in MEAL_PERIODS:
        if _to_min(start) <= t < _to_min(end):
            return name
    return None


def minutes_post_dismissal(ts, shift=0.0):
    """Signed minutes from the nearest class end time (+ = after).
    `shift` moves the schedule; the pipeline always uses shift=0 (assumed times)."""
    pattern = DAY_PATTERN.get(ts.dayofweek)
    if pattern is None:
        return np.nan
    t = ts.hour * 60 + ts.minute + ts.second / 60
    ends = [_to_min(e) + shift for e in CLASS_END_TIMES[pattern]]
    return min((t - e for e in ends), key=abs)


# -------------------------------------------------------------------- generator
def generate(settings, seed, **overrides):
    """Return a raw log (same columns as the live Google Sheet export).
    The TRUE dismissal schedule is the assumed one moved by schedule_shift_min."""
    s = {**settings, **overrides}
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(s["start_date"], periods=s["n_weekdays"])
    meals = list(s["meal_weights"])
    mw = np.array([s["meal_weights"][m] for m in meals])
    bounds = {n: (_to_min(a), _to_min(b)) for n, a, b in MEAL_PERIODS}
    rows = []
    for day in days:
        day_effect = rng.normal(0, s["day_effect_sd_min"])
        for _ in range(rng.poisson(s["obs_per_day_mean"])):
            meal = meals[rng.choice(len(meals), p=mw)]
            lo, hi = bounds[meal]
            join = day + pd.Timedelta(minutes=int(rng.integers(lo, hi)),
                                      seconds=int(rng.integers(0, 60)))
            true_mpd = minutes_post_dismissal(join, s["schedule_shift_min"])
            in_surge = 0 <= true_mpd <= s["surge_window_min"]
            surge = (s["surge_amplitude_min"] * np.exp(-true_mpd / s["surge_decay_tau_min"])
                     if in_surge else 0.0)
            probs = s["queue_probs_surge"] if in_surge else s["queue_probs_normal"]
            q = rng.choice(4, p=probs)
            wait = (s["queue_base_minutes"][q] + surge + s["meal_offset_min"][meal]
                    + day_effect + rng.normal(0, s["noise_sd_min"]))
            wait = round(max(0.3, wait), 1)
            tap = join + pd.Timedelta(seconds=round(wait * 60))
            rows.append({"source": "synthetic", "line_join_time": join, "card_tap_time": tap,
                         "visual_queue_category": QUEUE_LABELS[q],
                         "actual_wait_minutes": wait, "server_received_time": tap, "flag": ""})
    df = pd.DataFrame(rows).sort_values("line_join_time").reset_index(drop=True)
    df = _inject_malformed(df, s["n_malformed_rows"], rng)
    df.insert(0, "submission_id", [f"syn{seed:03d}_{i:04d}" for i in range(len(df))])
    for c in ["line_join_time", "card_tap_time", "server_received_time"]:
        df[c] = df[c].astype(str)
    return df


def _inject_malformed(df, n, rng):
    """Append deliberately bad rows so the cleaning step has something to catch.
    Includes the case flagged in CP2 feedback: reversed timestamps with a
    positive supplied wait."""
    kinds = ["missing_ts", "bad_category", "reversed_ts", "wait_mismatch",
             "duplicate", "outside_hours"]
    bad = []
    for i in range(n):
        r = df.iloc[rng.integers(len(df))].copy()
        k = kinds[i % len(kinds)]
        if k == "missing_ts":
            r["card_tap_time"] = pd.NaT
        elif k == "bad_category":
            r["visual_queue_category"] = "E: Not sure"
        elif k == "reversed_ts":
            r["line_join_time"], r["card_tap_time"] = r["card_tap_time"], r["line_join_time"]
        elif k == "wait_mismatch":
            r["actual_wait_minutes"] = round(r["actual_wait_minutes"] + rng.uniform(3, 10), 1)
        elif k == "outside_hours":
            r["line_join_time"] = r["line_join_time"].normalize() + pd.Timedelta(hours=22)
            r["card_tap_time"] = r["line_join_time"] + pd.Timedelta(minutes=r["actual_wait_minutes"])
        r["flag"] = f"injected:{k}"
        bad.append(r)
    return pd.concat([df, pd.DataFrame(bad)], ignore_index=True)


# --------------------------------------------------------------------- cleaning
def clean(raw, verbose=False):
    """Fixed cleaning step. The wait used as the target is RECOMPUTED from the two
    timestamps; rows whose timestamps are reversed, or whose supplied wait
    disagrees with the timestamps by more than WAIT_TOLERANCE_MIN, are rejected."""
    df = raw.copy()
    log = {"raw": len(df)}
    for c in ["line_join_time", "card_tap_time"]:
        df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    df["supplied_wait"] = pd.to_numeric(df["actual_wait_minutes"], errors="coerce")

    def drop(mask, why):
        nonlocal df
        log[why] = int((~mask).sum())
        df = df[mask]

    drop(df[["line_join_time", "card_tap_time", "visual_queue_category"]].notna().all(axis=1),
         "missing_field")
    drop(df["visual_queue_category"].isin(QUEUE_ORDER), "unknown_category")
    df["computed_wait"] = (df["card_tap_time"] - df["line_join_time"]).dt.total_seconds() / 60
    drop(df["computed_wait"] >= 0, "reversed_timestamps")
    drop(df["computed_wait"] <= MAX_WAIT, "over_max_wait")
    drop(df["supplied_wait"].isna() |
         ((df["supplied_wait"] - df["computed_wait"]).abs() <= WAIT_TOLERANCE_MIN),
         "wait_mismatch")
    drop(~df.duplicated(subset=["line_join_time", "card_tap_time", "visual_queue_category"]),
         "duplicate")
    df["meal_period"] = df["line_join_time"].apply(meal_period)
    drop(df["meal_period"].notna(), "outside_meal_hours")

    df["wait"] = df["computed_wait"].round(2)          # target = recomputed wait
    df["day_of_week"] = df["line_join_time"].dt.day_name()
    df["time_bin"] = df["line_join_time"].dt.floor("30min").dt.strftime("%H:%M")
    df["minutes_post_class_dismissal"] = df["line_join_time"].apply(minutes_post_dismissal)
    df["queue_ordinal"] = df["visual_queue_category"].map(QUEUE_ORDER)
    df["service_date"] = df["line_join_time"].dt.strftime("%Y-%m-%d")
    log["clean"] = len(df)
    if verbose:
        print(log)
    return df.sort_values("line_join_time").reset_index(drop=True), log


# --------------------------------------------------------- baselines and models
def baselines(train, test):
    g = train["wait"].mean()
    meal = train.groupby("meal_period")["wait"].mean()
    binned = train.groupby(["day_of_week", "time_bin"])["wait"].mean()
    b1 = test["meal_period"].map(meal).fillna(g)
    keys = list(zip(test["day_of_week"], test["time_bin"]))
    b2 = pd.Series([binned.get(k, np.nan) for k in keys], index=test.index).fillna(b1)
    return {"B0: Global mean": pd.Series(g, index=test.index),
            "B1: Meal-window mean": b1,
            "B2: Day x 30-min bin mean": b2}


def make_models(use_dismissal=True, seed=365):
    num = ["queue_ordinal"] + (["minutes_post_class_dismissal"] if use_dismissal else [])
    cat = ["day_of_week", "meal_period"]

    def pipe(model, scale=True):
        pre = ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        return Pipeline([("pre", pre), ("model", model)])

    return {"Linear Regression": pipe(LinearRegression()),
            "Ridge (alpha=1)": pipe(Ridge(alpha=1.0)),
            "Random Forest": pipe(RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1), scale=False)}, num + cat


def evaluate(df, grouped=True, use_dismissal=True, n_splits=5, seed=365, include_baselines=True):
    """Every method is fit and scored on identical folds. Baselines are fit
    inside each training fold. With grouped=True, all rows from one service
    date stay in the same fold (GroupKFold)."""
    y = df["wait"]
    if grouped:
        splits = list(GroupKFold(n_splits=n_splits).split(df, y, groups=df["service_date"]))
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df))
    models, feats = make_models(use_dismissal, seed)
    preds = {}
    for tr, te in splits:
        train, test = df.iloc[tr], df.iloc[te]
        if include_baselines:
            for k, p in baselines(train, test).items():
                preds.setdefault(k, []).append(p)
        for k, m in models.items():
            m.fit(train[feats], train["wait"])
            preds.setdefault(k, []).append(pd.Series(m.predict(test[feats]), index=test.index))
    out = []
    for k, parts in preds.items():
        p = pd.concat(parts).loc[y.index]
        out.append({"method": k, "MAE": mean_absolute_error(y, p),
                    "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p)})
    return pd.DataFrame(out).set_index("method")
