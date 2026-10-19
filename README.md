# Crowdsourced Wait-Time Prediction for Campus Dining: Reproducibility Package

Kamryn Schmidt · COMP 365 Individual Project · Fall 2026

This package rebuilds every result, table and figure in the paper *"When the Crowd Does Not Report: A Simulation Study of Queue-Report Features for Campus Dining Wait-Time Prediction"*, along with the intake web form used for data collection.

> **Important: the evaluated data is SYNTHETIC.** The deployed form collected only two rows, both of them the author's own tests. Every model result comes from a seeded generator whose settings (`generator_settings.json`) are the author's assumptions, not measurements of a real dining hall.

---

## 1. Contents

| File / folder | What it is |
|---|---|
| `waitsim.py` | Core library: synthetic generator, cleaning pipeline, features, 3 baselines, 3 models, grouped evaluation |
| `generator_settings.json` | Every generator setting (see Section 5) |
| `run_experiments.py` | Experiments E1–E4: main comparison (grouped vs. shuffled CV), dismissal ablation, schedule shift, sample size |
| `run_e5.py` | E5: dismissal feature with and without the queue report, across schedule errors |
| `run_e6.py` | E6: error broken down by queue category |
| `make_figures.py` | Draws the four paper figures as PNGs from `results/` |
| `run_all.py` | Runs all of the above in order |
| `Wait_Time_Demo.ipynb` | Google Colab notebook: step-by-step demo plus full rebuild |
| `requirements.txt` | Exact package versions used |
| `wait_log.csv` | Real export from the live form: 2 rows, both author tests (Sept 20, 2026); excluded from evaluation |
| `DEMO_wait_log_SYNTHETIC.csv` | The Checkpoint 2 synthetic file, kept for continuity; not used for the paper's results |
| `intake_form/` | Google Apps Script source for the anonymous two-tap intake form |
| `results/` | Reference outputs from the author's run (CSV/JSON), overwritten when you rerun |
| `figures/` | Reference figures used in the paper (PNG), overwritten when you rerun |

---

## 2. Quick start

### Option A: Google Colab (no install needed)
1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → choose `Wait_Time_Demo.ipynb`.
2. Run the cells top to bottom. When asked, upload `Schmidt_IP_Final_Reproducibility.zip` (this whole package, zipped).
3. Cells 2–5 show the pipeline on one dataset (about 30 seconds). Cell 6 runs the full rebuild (about 10–20 minutes).

### Option B: Your own computer
Requires Python 3.11 or newer.
```bash
cd Schmidt_IP_Final_Reproducibility
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```
Outputs are written to `results/` (tables) and `figures/` (PNGs).

To run one step at a time instead:
```bash
python run_experiments.py   # E1-E4, the longest step
python run_e5.py            # E5, a few minutes
python run_e6.py            # E6, under 1 min
python make_figures.py      # paper figures, under 1 min
```

---

## 3. Dependencies and hardware

| Item | Version / requirement |
|---|---|
| Python | 3.13.16 (any 3.11+ should work) |
| NumPy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Hardware | Any CPU; no GPU needed. Tested on 2 CPU cores and 7 GB RAM; peak memory use is well under 1 GB |
| Runtime | About 10–20 minutes for the full rebuild, depending on the machine |

All randomness is seeded, so with the versions above the numbers match the paper exactly. With other library versions the Random Forest results may differ in the second decimal place.

---

## 4. Data

- **Real data (`wait_log.csv`).** Two rows exported from the live Google Sheet. Both were submitted by the author on Sept 20, 2026, to test the form; neither is an observed dining-hall wait. They are run through the cleaning pipeline only as an end-to-end check. The file contains no names, IDs or emails.
- **Synthetic data.** Not stored; it is regenerated on demand by `waitsim.generate(settings, seed)`. The seeds used are:
  - Main experiments (E1–E3, E5, E6): seeds 1–10, 20 weekdays each
  - Sample-size experiment (E4): pools from seeds 101–120, 40 weekdays each, subsampled with seed 2027
- **Collecting new real data.** Deploy the intake form (Section 7). Export the sheet as CSV with **File → Download → CSV** and pass it to `waitsim.clean()`. Any new collection from other people requires an IRB determination first.

---

## 5. Model and experiment settings

**Generator** (`generator_settings.json`): 20 weekdays per dataset; Poisson mean of 12 observations per day; meal-period weights 0.12 / 0.45 / 0.13 / 0.30 (breakfast / lunch / late lunch / dinner); base wait by queue category A–D of 2 / 5 / 9 / 14 min; queue-category probabilities 0.40 / 0.35 / 0.18 / 0.07 normally and 0.10 / 0.30 / 0.35 / 0.25 during a surge; post-dismissal surge of 4.0 min amplitude over a 20-min window with an 8-min decay; day-effect SD 1.0 min; noise SD 1.3 min; 12 deliberately malformed rows per dataset.

**Cleaning** (`waitsim.clean`): the target wait is recomputed from the two timestamps. A row is rejected if it has a missing field, an unknown category, reversed timestamps, a recomputed wait over 90 min, a supplied wait differing from the recomputed wait by more than 0.5 min, or a duplicate entry, or if it falls outside posted meal hours.

**Features:** queue category as an ordinal (0–3), signed minutes from the nearest *assumed* class end time, day of week (one-hot), and meal period (one-hot). Numeric features are standardized inside each training fold.

**Baselines:** B0 global mean; B1 meal-period mean; B2 weekday × 30-min-bin mean, falling back to B1.

**Models (scikit-learn, trained from scratch, no tuning):**
- `LinearRegression()`
- `Ridge(alpha=1.0)`
- `RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=365)`

**Evaluation:** `GroupKFold(n_splits=5)` grouped by service date. Baselines, scalers and encoders are fit on the training folds only. Shuffled `KFold(5, shuffle=True, random_state=365)` is reported for comparison. Metrics are MAE (primary), RMSE and R².

---

## 6. Expected results and where they appear in the paper

| Output file | Paper | Key number to check |
|---|---|---|
| `results/e1_summary.csv` (grouped) | Table 2, Figure 1 | Random forest MAE 1.47 ± 0.10; best baseline (B1) 3.54 ± 0.19 |
| `results/e1_summary.csv` (shuffled) | Results text | Random forest 1.38; linear 1.42 |
| `results/e6_by_queue.csv` | Table 3, Figure 2 | Category D: RF MAE 1.67, B1 MAE 8.46 |
| `results/e2_summary.csv`, `e5_summary.csv` | Table 4, Figure 3 | No queue report: RF 3.48 with the dismissal feature vs. 3.69 without |
| `results/e4_summary.csv` | Figure 4 | n = 30: best model MAE 1.67 ± 0.32 |
| `results/cleaning_summary.json` | Methods text | Exactly 12 rows removed per dataset; mean 239.6 clean rows |

---

## 7. Rebuilding the intake form

The form is a Google Apps Script web app that writes to a Google Sheet. It stores no names, IDs or emails.

1. Create a new Google Sheet, then open **Extensions → Apps Script**.
2. Replace `Code.gs` with `intake_form/Code.gs`. Add an HTML file named `Index` and paste in `intake_form/Index.html`.
3. In **Project Settings**, check *Show "appsscript.json"*, then paste in `intake_form/appsscript.json`.
4. Run `setup()` once and authorize it. This creates the `log` tab and its header row.
5. **Deploy → New deployment → Web app**. Set *Execute as:* Me.
6. Open the `/exec` URL on a phone, choose a line length, tap **I just got in line**, then tap **I just tapped my card**. A new row appears in the sheet.

---

## 8. Changes since Checkpoint 2 (in response to instructor feedback)

1. **Cleaning consistency check.** The wait is now recomputed from the timestamps; reversed or mismatched rows are rejected. (Previously a reversed-timestamp row with a positive supplied wait survived.)
2. **Grouped evaluation.** All observations from one service date stay in the same fold.
3. **Dismissal-feature ablation.** Every model is run with and without the feature, in surge and no-surge worlds, with and without the queue report.
4. **Generator and settings included.** The full generator code, settings file and seeds are provided.
5. **The 13.8-minute entry** in `wait_log.csv` is a form-test submission, not an observed dining-hall wait.

---

## 9. AI assistance disclosure

An AI assistant (Claude, Anthropic) was used for this component. It drafted the Python code in this package (`waitsim.py`, the experiment scripts, `make_figures.py`, `run_all.py`), the Colab demo notebook, and this README, and it helped debug the Apps Script intake form. The author reviewed the code, ran the experiments, and checked the outputs against the numbers reported in the paper.
