# FootPredict

Football prediction website built on machine learning. Diploma project:
*«Інформаційна система прогнозування результатів спортивних подій на основі машинного навчання»*.

FootPredict:

* turns historical Football-Data results into leak-free pre-match features;
* trains four specialised **XGBoost** models (match result, home goals, away goals, total corners);
* derives consistent probabilities for 1X2, double chance, totals, BTTS, exact score, handicap and corners;
* attaches **real bookmaker odds only** (never derived or simulated prices) and computes expected value;
* picks one **Main prediction** per match and, when it qualifies, an optional **Risk prediction**;
* explains every prediction with **SHAP** in plain language (EN / UA);
* stores every prediction, settles it after the match and reports profit / ROI of the Main strategy;
* runs as a multi-page Streamlit website with an English / Ukrainian switch and dark / light theme.

Live: <https://football-prediction-diploma.streamlit.app>

---

## Architecture

```
Football-Data CSV (history, Bet365 prices)      API-Football / The Odds API (fresh fixtures, results, odds)
        │  src/data/                                   │  src/providers/  (cached, quota-aware)
        └──────────────► unified match table ◄─────────┘   team-name aliases, no fuzzy matching
                               │
                 leak-free pre-match features            src/features/
                               │
            4 × XGBoost  (Hyperopt on the validation season)     src/modelling/
                               │   + draw-aware decision rule (chosen on validation)
          P(H/D/A), λ_home, λ_away, μ_corners
                               │
     score matrix (Poisson, aligned with 1X2) → markets           src/markets/
                               │
          real odds → EV → Main / Risk prediction                 src/markets/odds.py, recommendation.py
                               │
     prediction store → settlement → Main profit / ROI            src/prediction/, src/analytics/
                               │
             SHAP explanation (on demand, per model version)      src/explain/
                               │
                     Streamlit UI (EN / UA, dark / light)         app.py, views/, src/ui/
```

| Path | Content |
|---|---|
| `app.py` | entry point: brand, navigation, language and theme switches |
| `views/` | pages: matches, match details, history, leagues, teams, analytics, model analysis |
| `scripts/` | `run_pipeline.py` (daily update), `retrain_models.py`, `update_data.py`, `build_features.py`, `migrate_real_odds.py` |
| `src/config.py` | all settings (paths, seasons, API, markets, recommendation thresholds) |
| `src/leagues.py` | league registry with provider ids, season calendar and full-support flag |
| `src/data/` | Football-Data download, normalisation, match identity, deduplication |
| `src/providers/` | API-Football, The Odds API, Football-Data prices, aliases, cache, collection |
| `src/features/` | feature builder, feature sets per model, human-readable names (EN / UA) |
| `src/modelling/` | tasks, preprocessing, Hyperopt training, draw decision rule, evaluation, registry |
| `src/markets/` | distributions, markets, real-odds layer, main / risk rules |
| `src/prediction/` | prediction engine, prediction store, settlement |
| `src/explain/` | SHAP and the plain-language explanation |
| `src/analytics/` | team / league statistics, profit and ROI |
| `src/ui/` | theme (CSS variables), translations, components, cached data access |
| `tests/` | unit tests; `tests/e2e/` Playwright browser tests |
| `local_data/` | results, fixtures, live results, odds, features, models, predictions |

## Data sources

**Football-Data** ([football-data.co.uk](https://www.football-data.co.uk)) is the historical source for the
models: results with shots, shots on target, corners, fouls and cards, plus real **Bet365** prices for 1X2,
total goals 2.5 and the Asian handicap column (only half lines ±0.5 / ±1.5 / ±2.5 are used — they are
equivalent to a 2-way handicap without refunds).

**API providers** fix Football-Data's publishing delay and add real odds for more markets:

| Provider | Used for | Free plan | Calls per daily run |
|---|---|---|---|
| [API-Football](https://www.api-football.com) (api-sports.io, v3) | fixtures and final scores of the last 2 days and next 3 days, Bet365 odds for 1X2, totals, BTTS, double chance, exact score, European handicap, half-line Asian handicap, corners | 100 requests / day | 6 fixture requests (one per day, all leagues) + 1–2 odds pages per league with upcoming matches ≈ 25–45 |
| [The Odds API](https://the-odds-api.com) (v4) | additional real odds (h2h, totals, spreads) for leagues API-Football did not price; fixtures and scores if API-Football is not configured | 500 credits / month | 3 credits per league request, capped at 20 credits per run |

Priority when several providers price the same selection: API-Football → The Odds API → Football-Data.
Responses are cached in `local_data/api/` with a quota log; the website never calls an API — it only reads
files written by the pipeline. A provider name that cannot be mapped to a Football-Data team name is skipped
and written to `local_data/api/unmatched_teams.csv`; add it to `src/providers/aliases.py`.

### Supported leagues — full support only

21 divisions: England 1–4, Scotland 1–4, Spain 1–2, Italy 1–2, Germany 1–2, France 1–2, Netherlands,
Belgium, Portugal, Turkey, Greece. Every enabled league has several seasons of results with shots, shots on
target, corners and cards (tested in `tests/test_rules.py`), so every model and market works.

Not included: the English National League (only ≈5 % of matches have corner statistics) and the 16 extra
Football-Data leagues (Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway,
Poland, Romania, Russia, Sweden, Switzerland, USA), whose files contain goals and closing 1X2 odds only — no
shots, corners or cards. The registry already supports calendar-year seasons (`calendar="calendar_year"`)
should a complete source for such leagues become available.

## Features and leakage protection

Matches are processed **day by day**: the features of every match on day *D* come from matches before *D*;
only then are the results of *D* added. Future fixtures and API rows get features but never update tables,
form or targets. Groups: league table, season statistics, home / away, form 5 / 10 / 15, venue form,
event frequencies, trends, rest days, head-to-head, attack-vs-defence and strength gaps, corner pressure
(corner and shot shares), league context. Missing values stay `NaN` (XGBoost handles them natively).
Odds are **not** model inputs (`USE_ODDS_FEATURES = False`).

## Models

| Model | Algorithm | Target | Objective | Hyperopt metric |
|---|---|---|---|---|
| Result | `XGBClassifier` | H / D / A | `multi:softprob` | log loss |
| Home goals | `XGBRegressor` | home goals | `count:poisson` | Poisson deviance |
| Away goals | `XGBRegressor` | away goals | `count:poisson` | Poisson deviance |
| Corners | `XGBRegressor` | total corners | `count:poisson` + negative-binomial dispersion | Poisson deviance |

Temporal evaluation (current season 2026/27): **train** 2021/22–2023/24, **validation** 2024/25 (Hyperopt,
early stopping, every design decision), **test** 2025/26 (evaluated once), **current** season predicted
out-of-sample. Two model sets are saved: the evaluation model (train + validation, scored on test, fills the
test-season history) and the production model (all completed seasons).

### Draws

The classifier's draw probabilities are well calibrated, but a draw is rarely the single most likely outcome,
so a plain arg-max almost never predicts one. Options compared on the validation season only: draw class
weights (better recall, worse log loss), temperature / vector scaling (no gain), blending with the Poisson
outcome probabilities (negligible gain) and a **decision rule**. The decision rule was chosen: probabilities
(log loss, Brier) stay unchanged, the named outcome is `argmax(P(H), m·P(D), P(A))`, with `m` chosen on
validation as the highest-macro-F1 value whose predicted draw share does not exceed the training draw share.

### Corners

Validation comparison: the v2 feature set, + corner-specific features (venue form, corner difference and
share, conceded-corner trend, league home / away corners), + shot-based pressure features, and separate home
+ away corner models. The single total-corners model with corner and shot features was best; separate models
were not better and were not adopted.

## Markets

All goal markets come from one score matrix: `Pois(i; λ_home) · Pois(j; λ_away)`, with the home-win / draw /
away-win regions rescaled to the classifier probabilities, so 1X2, double chance, totals 1.5 / 2.5 / 3.5,
BTTS, exact scores and handicaps agree with each other.

**Handicap** (home team's perspective): whole lines ±1, ±2 are 3-way European handicaps (home / draw / away
after adding the line to the home goals); half lines ±0.5, ±1.5, ±2.5 are 2-way (a draw after the handicap
is impossible). Settlement is tested for both types.

## Odds, EV and the Main prediction

* **Odds** = a real bookmaker price from a provider; if a selection has no real price, the interface shows
  no odds and no EV — nothing is derived or simulated.
* **EV** = model probability × real odds − 1 (theoretical edge, not profit).
* **Main prediction** — the best prediction for every match: among selections with probability 45–80 %,
  a selection with real odds 1.30–3.00 and EV ≥ +3 % wins (highest EV × market reliability); otherwise the
  highest probability × market reliability. It always exists when the match has a prediction.
* **Risk prediction** — optional, real odds only: odds 3.0–8.0, probability ≥ 22 %, EV ≥ +5 %, reliable
  market. Hidden when nothing qualifies.

## History, settlement and ROI

Each match stores many predictions: market forecasts (1X2 — draw-aware —, total 2.5, BTTS, double chance,
corners 9.5, exact score, handicap), the Main and the Risk prediction. After the match: `WON / LOST`
(`VOID` only for a match that never took place). Fixed stake 1 unit: win `odds − 1`, loss `−1`, void `0`
(not staked). **ROI = net profit / amount staked × 100**, counting only predictions with real odds; the hit
rate counts every settled prediction. **Analytics defaults to Main predictions only** (one per match) — this
is the FootPredict strategy result; Risk, all market forecasts and single markets are separate views.

`scripts/migrate_real_odds.py` removed the derived / simulated prices of the v2 history (the predictions and
their results are kept; they no longer count in profit / ROI).

## Website

Matches (leagues, date strip, team search, 1X2 + total probabilities with real odds), Match Details
(Main / Risk, markets, team comparison, recent form, home / away, H2H, rest days, "Why this prediction?" with
relative influence bars, technical SHAP in an expander), Prediction History (completed predictions by default,
period presets, group-by, advanced filters), Leagues, Teams (period selector drives statistics, charts and
match list), Analytics (Main strategy, best / weakest market with a minimum sample), Model Analysis (metrics
with explanations, draw section, baseline comparison, calibration, global SHAP, feature groups).

## Installation

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## API setup

1. Register at <https://dashboard.api-football.com/register> (free plan) and copy the API key.
2. Register at <https://the-odds-api.com> (free plan) and copy the API key.
3. Copy `.env.example` to `.env` (never commit `.env`) and fill in:

   ```
   API_FOOTBALL_KEY=your-api-football-key
   ODDS_API_KEY=your-odds-api-key
   ```

4. Run `python scripts/run_pipeline.py`.
5. For the scheduled cloud update add the same two values as GitHub repository secrets
   (Settings → Secrets and variables → Actions) or with `gh secret set API_FOOTBALL_KEY`.

Without keys the pipeline still works with Football-Data only (results, fixtures and Bet365 prices for
1X2, total 2.5 and half-line handicaps).

## Updating and retraining

```bash
python scripts/run_pipeline.py              # APIs + Football-Data -> features -> settle -> predict
python scripts/run_pipeline.py --offline    # local files only
python scripts/retrain_models.py            # retrain the four models (≈15 min)
```

* `run_pipeline.py` updates data and predictions; it never retrains. Run it once or twice a day, before
  match days. Order: providers → Football-Data → features → settlement (once) → predictions.
* `retrain_models.py` re-tunes and refits the models; run it weekly or after a few hundred new results, and
  always when a season ends (the train / validation / test split moves forward).

### Deployment and synchronisation

The deployed app (Streamlit Community Cloud) is built from `main`. The workflow
`.github/workflows/update-data.yml` is the single writer of `local_data/`: twice a day it runs the pipeline
with the API keys from the repository secrets and commits the new data to `main`; Streamlit redeploys on
the push. A local checkout of `main` is synchronised with `git pull` — the footer of every page shows the
data timestamp, and `E2E_BASE_URL=<app url>/~/+ pytest tests/e2e --e2e -k sync` checks that the deployed app
serves exactly the data of the checkout.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                     # unit tests
python -m playwright install chromium
pytest tests/e2e --e2e                     # browser tests (starts the app)
E2E_BASE_URL=https://football-prediction-diploma.streamlit.app/~/+ pytest tests/e2e --e2e   # deployed app
```
