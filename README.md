# Football Prediction System

Information system for predicting the outcomes of football matches with machine
learning. Diploma project: *«Інформаційна система прогнозування результатів
спортивних подій на основі машинного навчання»*.

The system does more than say "home / draw / away". It:

* turns historical Football-Data results into pre-match features without data leakage;
* trains four specialised **XGBoost** models (result, home goals, away goals, corners);
* derives probabilities for 1X2, double chance, total goals, BTTS, exact score,
  European handicap and total corners from those models;
* compares the probabilities with market odds (kept separate from the ML part) and
  picks a **main** and an optional **risk** prediction per match;
* explains every prediction with **SHAP** in plain language and as a detailed chart;
* stores every prediction, settles it after the match and reports profit / ROI;
* shows everything in a multi-page Streamlit web interface.

---

## Architecture

```
Football-Data CSV (results + fixtures)
        │  src/data/download.py
        ▼
Cleaning, match identity, deduplication           src/data/cleaning.py, dataset.py
        ▼
Pre-match features (day-by-day, leak-free)        src/features/builder.py, state.py
        ▼
Feature sets per model                            src/features/sets.py
        ▼
4 × XGBoost (Hyperopt on a validation season)     src/modelling/
        ▼
Probabilities: P(H/D/A), λ_home, λ_away, μ_corners
        ▼
Distributions → markets                           src/markets/distributions.py, markets.py
        ▼
Odds layer: market / derived / simulated, EV      src/markets/odds.py
        ▼
Main & risk prediction rules                      src/markets/recommendation.py
        ▼
Prediction store (snapshots, markets, picks)      src/prediction/
        ▼
Settlement → profit / ROI analytics               src/prediction/store.py, src/analytics/
        ▼
SHAP explanations (on demand)                     src/explain/
        ▼
Streamlit UI                                      app.py, views/, src/ui/
```

```
app.py                      Streamlit entry point (navigation)
views/                      pages: matches, match_details, history, leagues, teams, analytics, model_analysis
scripts/                    command-line entry points (update data, build features, retrain, daily pipeline)
src/config.py               all settings: paths, seasons, thresholds, market lines
src/leagues.py              league registry (one line per division)
src/storage.py              LocalStorage: CSV / Parquet / joblib / JSON under local_data/
src/data/                   download, normalisation, deduplication
src/features/               feature builder, feature sets, human-readable names
src/modelling/              tasks, preprocessing, Hyperopt training, evaluation, model registry
src/markets/                score / corner distributions, markets, odds, recommendations
src/prediction/             prediction engine, prediction store and settlement
src/explain/                SHAP explanations and plain-language summaries
src/analytics/              team / league statistics, ROI statistics
src/ui/                     theme, cached data access, UI components
tests/                      unit tests; tests/e2e browser tests (Playwright)
local_data/                 data: results/, fixtures/, processed/, models/, predictions/
```

## Data source

[Football-Data.co.uk](https://www.football-data.co.uk) — free CSV files with results,
match statistics (shots, shots on target, corners, fouls, cards) and bookmaker odds.

* Historical results: `local_data/results/<season>/<division>.csv` (e.g. `2526/E0.csv`).
* Future fixtures: `local_data/fixtures/fixtures.csv` (teams, date, time, odds — no result).

All 22 main divisions are registered in `src/leagues.py` (England 1–5, Scotland 1–4,
Spain 1–2, Italy 1–2, Germany 1–2, France 1–2, Netherlands, Belgium, Portugal,
Turkey, Greece). Adding a division is one `League(...)` line; nothing else in the
code refers to specific league codes. When a league lacks some statistic (e.g. the
English National League has no corner data) the affected market is shown as
*unavailable* for that league instead of being invented.

### Match identity and deduplication

`match_id = league + date + home team + away team`. Kick-off time is not part of the
identity because Football-Data occasionally reports different times for the same match
in fixtures and results, and a club never plays two league matches on one day.
Only true duplicates (same id) are removed; a played row wins over a fixture row. The
same pairing in another season, or a second meeting in the same season, stays.
A fixture replaced by a result published under another date (rescheduled match) is
dropped as stale.

## Feature engineering and leakage protection

Matches are processed **day by day**: features of every match on day *D* are read from
the state built from matches *before D*; only then are the played matches of *D* added.
So a match never sees its own result, simultaneous matches of a round do not see each
other, and future fixtures get features but never update tables, form or targets.
Tests in `tests/test_features.py` check exactly these properties.

| Group | Examples |
|---|---|
| League table (season, reset each season) | games, points, points/game, goal difference, position, relative position |
| Season statistics | goals for / against, shots, shots on target, corners for / against, fouls, cards |
| Home / away | home team's home record, away team's away record |
| Form (last 5 / 10 / 15, across seasons) | win / draw / loss rates, PPG, goals, conceded, GD, shots, SoT, corners |
| Event frequencies (last 10) | BTTS, over 1.5 / 2.5 / 3.5, clean sheets, failed to score |
| Trends | last 5 minus season average: scoring, conceding, shots, corners |
| Rest days | days since previous match (capped at 30), difference |
| Head-to-head (last 5 meetings before the match) | wins / draws / losses, goals, BTTS rate, over 2.5 rate |
| Derived | attack vs defence, position / points / form gaps, corner volume |
| League context | running league averages (goals, corners, draw rate) |

**Missing data.** Missing values stay `NaN` — XGBoost learns a default direction for
missing values at every split, which is exactly the right behaviour for "no H2H yet",
"fewer than 15 matches", "start of the season" or "no corner data". Rows are not
dropped, and counters such as `h_form15_n`, `h2h_n` and `h_season_games` tell the model
how much history stands behind the other numbers.

**Encoding.** The only categorical feature is the league. A scikit-learn
`OneHotEncoder(handle_unknown="ignore")` is fitted on the training rows and saved
inside each model bundle, so training, validation, test and live prediction always
produce the same columns in the same order; an unknown league becomes an all-zero block.

**Odds are not features.** Bookmaker odds are kept only as market information. The
feature builder already computes margin-free implied probabilities (`market_prob_*`);
switching `USE_ODDS_FEATURES = True` in `src/config.py` adds them to every model's
feature set without any other change.

## Machine-learning models

| Model | Algorithm | Target | Objective | Hyperopt metric | Output |
|---|---|---|---|---|---|
| Result | `XGBClassifier` | H / D / A | `multi:softprob` | log loss | P(home), P(draw), P(away) |
| Home goals | `XGBRegressor` | home goals | `count:poisson` | Poisson deviance | λ_home |
| Away goals | `XGBRegressor` | away goals | `count:poisson` | Poisson deviance | λ_away |
| Corners | `XGBRegressor` | total corners | `count:poisson` | Poisson deviance | μ_corners |

Every model has its own feature set (`src/features/sets.py`): the result model leans
on table, form and strength gaps; the goal models on attack / defence, shots and
scoring frequencies; the corner model on corner statistics and shot volume.

*Why log loss for the classifier?* All markets and the EV calculation consume
probabilities. Log loss rewards well-calibrated probabilities, while accuracy and
macro F1 only look at the most likely class.
*Why Poisson deviance for the regressors?* Their predictions are used as Poisson
rates; Poisson deviance is the proper loss for count targets.

### Temporal evaluation

| Period | Seasons (current season 2026/27) | Use |
|---|---|---|
| Train | 2021/22 – 2023/24 | fitting during Hyperopt |
| Validation | 2024/25 | Hyperopt objective + early stopping only |
| Test | 2025/26 | final evaluation, never used for tuning |
| Current | 2026/27 | out-of-sample live predictions |

The split moves automatically with the calendar (`season_split` in
`src/modelling/pipeline.py`). Hyperopt (TPE) runs 30 trials for the classifier and 20
for each regressor with early stopping on the validation season. Two model versions are
saved:

* `backtest_<season>` — fitted on train + validation, evaluated on test; its predictions
  of the test season fill the history as *backtest*;
* `prod_<date>` — the same parameters refitted on all completed seasons; it predicts
  the current season.

Model Analysis also reports two reference forecasts on the test season: the naive
class-frequency forecast and the bookmaker's implied probabilities (for comparison only).

## Prediction markets

All goal markets come from **one score matrix**:

1. `P(i, j) = Pois(i; λ_home) · Pois(j; λ_away)` for 0–10 goals each;
2. the home-win / draw / away-win regions of the matrix are rescaled to the
   classifier's probabilities (independent Poisson misprices draws, the classifier is
   trained on exactly that target), keeping the score shape inside each region.

From that matrix:

* **1X2** — the three regions (equal to the classifier output);
* **Double chance** — 1X = P(H)+P(D), X2 = P(D)+P(A), 12 = P(H)+P(A);
* **Total goals** over / under 1.5, 2.5, 3.5 — sums over cells with `i + j > line`;
* **BTTS** — sum over cells with `i > 0 and j > 0`;
* **Exact score** — the five most probable cells (no per-score models);
* **Handicap** — European 3-way handicap ±1, ±2: the line is added to the home goals,
  home / draw / away of the adjusted score; three outcomes cover all scores, so there is
  never a refund.

**Corners**: total corners follow a negative-binomial distribution with mean μ_corners
and a dispersion estimated on validation residuals (Poisson when not over-dispersed).
Over / under 8.5, 9.5, 10.5 are stored; the match page has a slider for any line.

## Odds, EV and recommendations

For every selection the system keeps apart:

* **model probability** `p`;
* **fair odds** `1 / p`;
* **odds** and **odds source**:
  * `market` — real bookmaker average from Football-Data (1X2, over / under 2.5);
  * `derived` — computed from the real 1X2 and O/U 2.5 prices of the same match: a
    Poisson score model is fitted to the bookmaker probabilities and the bookmaker's
    margin is applied (double chance, other totals, BTTS, exact score, handicap);
  * `simulated` — no market information (corners, or matches without odds): a naive
    bookmaker prices the event at its recent league frequency plus a 6 % margin;
* **EV** = `p · odds − 1` per unit stake.

Derived and simulated prices are always labelled; history and analytics can be
filtered by odds source.

**Main prediction** (rule in `src/markets/recommendation.py`):

1. candidates: all priced selections except exact scores with `p ≥ 45 %` and
   `1.30 ≤ odds < 3.00`;
2. if any candidate has `EV ≥ +3 %`, choose the highest
   `EV × market reliability × odds-source reliability` → *value* pick;
3. otherwise choose the most probable candidate → *confidence* pick (labelled "not a
   value bet").

**Risk prediction** (optional): `3.0 ≤ odds ≤ 8.0`, `p ≥ 22 %`, `EV ≥ +5 %`, highest
weighted EV; not shown when nothing qualifies. All thresholds live in `src/config.py`.

## Prediction history and ROI

`local_data/predictions/`:

* `snapshots.parquet` — one row per predicted match (probabilities, expected goals and
  corners, main / risk selection, model version, source);
* `markets.parquet` — every priced selection;
* `picks.parquet` — stored predictions (one match → many predictions): the most likely
  selection of 1X2, total 2.5, BTTS, corners 9.5 and exact score, plus main and risk.

Each pick has a status `UPCOMING → WON / LOST` (`VOID` if corners were not reported or
the match never took place) and a unit-stake profit: win `odds − 1`, loss `−1`.
ROI = total profit / number of settled predictions × 100.

Prediction origins:

* `live` — generated before kick-off;
* `backfill` — current-season match predicted after the fact by a model trained only on
  earlier seasons, from pre-match features (still out-of-sample);
* `backtest` — test season, predicted by the backtest model.

A prediction is frozen once its match date has passed; re-running the pipeline only
refreshes upcoming matches.

## SHAP explanations

TreeSHAP values are computed on demand with the model version that produced the stored
prediction, from the stored pre-match features.

* **Why this prediction?** — a sentence and the main factors for / against in plain
  language (feature names are mapped to readable labels in `src/features/names.py`).
* **Detailed SHAP** — horizontal bar chart, top 10 / 20 / all, positive and negative
  contributions coloured separately.
* **Global SHAP** (Model Analysis) — mean |SHAP| on the test season for each model,
  next to XGBoost gain importance.

Market → model mapping: 1X2 / handicap → result-model class; double chance → the
excluded class with opposite sign; totals / BTTS → sum of both goal models (sign
flipped for under / no); corners → corners model.

## Web interface

| Page | Content |
|---|---|
| Matches | leagues grouped by country (left), date strip and calendar (top), match cards with 1X2, total and BTTS probabilities and odds (centre), day summary (right) |
| Match Details | header, result and settled predictions, main / risk prediction, why this prediction, detailed SHAP, all markets, corners slider, team comparison, last 5 / 10 / 15, home / away, H2H, rest days |
| Prediction History | filters (dates, league, market, type, status, odds source, origin), daily / weekly / monthly / yearly summary, prediction table |
| Leagues | table with form, upcoming matches, recent results, prediction performance |
| Teams | search, position, form 5 / 10 / 15, home / away, trends, matches |
| Analytics | overall ROI, by type, by league, by market, by period; Advanced Analytics charts |
| Model Analysis | periods, metrics with plain explanations, confusion matrix, calibration, benchmarks, global SHAP, Hyperopt search, derived markets |

## Installation

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

The repository already contains downloaded data, trained models and predictions, so
the interface works right away.

## Updating data and predictions

```bash
python scripts/run_pipeline.py              # download current season + fixtures,
                                            # rebuild features, predict, settle
python scripts/run_pipeline.py --offline    # same without downloading
python scripts/update_data.py --all-seasons # re-download every season
python scripts/build_features.py            # only rebuild features
```

Updating data and retraining are separate: the daily pipeline never changes the models.

## Retraining

```bash
python scripts/retrain_models.py            # full Hyperopt search (≈10–15 min)
python scripts/retrain_models.py --quick    # 5 trials per model, for a smoke test
```

Retrain once a season has finished: the split moves forward by one season, the new
production model includes the finished season, and the backtest history is
regenerated. Stored live predictions are never rewritten.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                     # unit tests
python -m playwright install chromium
pytest tests/e2e --e2e                     # browser tests: starts the app and drives every page
```

Unit tests cover deduplication, leakage (same-day matches, future fixtures, invariance
to later data), odds exclusion from features, probability sums, exact score, totals,
handicap, settlement and ROI.

## Possible extensions

Odds as model features (one flag), Asian handicap, cards and corner handicap markets,
a simple baseline model for comparison, probability calibration (isotonic) on the
validation season, scheduled pipeline runs.
