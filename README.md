# FootPredict

Football prediction website built on machine learning. Diploma project:
*«Інформаційна система прогнозування результатів спортивних подій на основі машинного навчання»*.

FootPredict:

* turns historical Football-Data results into leak-free pre-match features;
* trains four specialised **XGBoost** models (match result, home goals, away goals, total corners) with Elo and
  opponent-adjusted features;
* derives consistent probabilities for 1X2, double chance, totals, BTTS, exact score, handicap and corners;
* attaches **real bookmaker odds only** (never derived or simulated prices) and computes expected value;
* picks a **Main prediction** and a **Risk prediction** for every match it can predict;
* explains every prediction with **SHAP** in plain language (EN / UA);
* records predictions **before kick-off only** (every pre-match version is archived, the last one before
  kick-off counts), settles them — Asian handicaps with refunds — and reports profit / ROI;
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
| `scripts/` | `run_pipeline.py` (update, modes results / odds / full), `retrain_models.py`, `decision_study.py`, `odds_credits.py`, `check_team_aliases.py`, `migrate_prediction_history.py`, `update_data.py`, `build_features.py`, `migrate_real_odds.py` |
| `src/config.py` | all settings (paths, seasons, API, markets, recommendation thresholds) |
| `src/leagues.py` | league registry with provider ids, season calendar and full-support flag |
| `src/data/` | Football-Data download, normalisation, match identity, deduplication |
| `src/providers/` | API-Football, The Odds API, Football-Data prices, aliases, cache, credit ledger, collection |
| `src/features/` | feature builder, Elo ratings, feature sets per model, feature dictionary, readable names (EN / UA) |
| `src/modelling/` | tasks, preprocessing, Hyperopt training, draw decision rule, evaluation, registry |
| `src/markets/` | distributions, markets, real-odds layer, main / risk rules |
| `src/prediction/` | prediction engine, prediction store, settlement |
| `src/explain/` | football facts, SHAP and the plain-language explanation |
| `src/analytics/` | team / league statistics, profit and ROI |
| `src/ui/` | theme (CSS variables), translations, components, cached data access |
| `tests/` | unit tests; `tests/e2e/` Playwright browser tests |
| `local_data/` | results, fixtures, live results, odds, features, models, predictions |

## Data sources

Only **real bookmaker prices** are used. A selection without a published price shows no odds and no EV;
nothing is derived, estimated or simulated, and every price keeps its provider, bookmaker and time.

| Source | Used for | Free plan | Role |
|---|---|---|---|
| [Football-Data](https://www.football-data.co.uk) | history: results, shots, corners, cards; Bet365 1X2, total 2.5, Asian handicap (half and whole lines) | free download | model training, historical prices |
| [OddsPapi](https://oddspapi.io) v4 | fixtures and real odds of many bookmakers for 1X2, double chance, totals, BTTS, correct score, Asian / European handicap, corner totals; historical odds since January 2026 | 250 requests / month; historical odds free | **main odds provider** (see below) |
| [API-Football](https://www.api-football.com) v3 | fixtures and final scores (yesterday … tomorrow); club badge URLs | 100 requests / day; seasons 2022–2024 only for odds and team lists | fresh results |
| [The Odds API](https://the-odds-api.com) v4 | 1X2 / totals per league, BTTS, double chance, goal and corner lines per match (UK bookmakers) | 500 credits / month | fallback (`ODDS_API_ENABLED`) |

When several providers price a selection: OddsPapi → API-Football → The Odds API → Football-Data. The website
never calls an API: it only reads files written by the pipeline, so opening it costs no quota.

### OddsPapi: setup and quota

1. Create a free account at <https://oddspapi.io> and copy the API key.
2. Locally: add `ODDSPAPI_KEY=...` to `.env` (git-ignored). For the cloud update: GitHub → Settings → Secrets and
   variables → Actions → *New repository secret* `ODDSPAPI_KEY`. Never put the key in code, tests or commits.
3. `python scripts/odds_credits.py` shows the requests left (no API call).

Costs per odds update (`--mode odds`): the market catalogue and the tournament list are fetched once and cached
(months); **one** `/fixtures` request lists the fixtures of the next 4 days in all leagues; prices come from
`/historical-odds` (free, one call per match with a 5-second pause) or, if that endpoint does not serve upcoming
matches, from `/odds` (1 request per match within `ODDSPAPI_RUN_BUDGET`). Safeguards
(`local_data/api/oddspapi/ledger.json`, committed): a match is not asked again within 12 h — also when the
answer was "no odds" — but a temporary error (network, 5xx) may be retried; a monthly reserve of 20 requests is
never spent; the API's own counter (`/account`, free) is read every run.

### Historical odds (2026)

OddsPapi keeps every price snapshot since January 2026. `scripts/oddspapi_history.py` downloads, per league,
the fixtures (1 request per league) and then the snapshots of each played match (free endpoint); the pre-match
price is the last snapshot before kick-off, never a later one. Training still uses Football-Data seasons
2021/22 onward; the 2026 prices are used to evaluate predictions. A historical prediction is only evaluated
with the model whose training data ended before that match (the evaluation model for 2025/26, the production
model for 2026/27 matches), never with a model that saw the result.

### Team and match matching

One internal team id — `"<country>|<Football-Data name>"` — and, per provider, the provider's team ids
(`local_data/teams/provider_ids.json`). A provider id is learned only from an unambiguous name match inside the
right league (`src/providers/aliases.py`: accents, prefixes, aliases, unique word-subset match) and never
re-assigned; an id that contradicts the name is refused and reported. Odds are attached to a match only when
league, date and both teams agree. Unmatched names: `local_data/api/unmatched_teams.csv`;
`python scripts/check_team_aliases.py` checks the cached responses.

### The Odds API credits (fallback)

Costs: `/events` 0 credits; `/sports/{league}/odds` 1 credit per market (h2h + totals = 2 per league);
`/events/{id}/odds` 1 credit per returned market (≈4 per match for BTTS, double chance, goal and corner lines).
The free 500 credits cover the 1X2 / totals of every league a few times a week plus extended markets for about
100 top-league matches a month — full coverage of all markets is not possible for free.

Safeguards (`src/providers/ledger.py`, committed as `local_data/api/the_odds_api/ledger.json` because the CI
cache does not persist): a league is not asked again within 20 h and a match's extended markets are bought once
— also when the answer was empty; every run has a credit budget (`--credit-budget`, default 60) and a monthly
reserve of 30 credits is never spent. `python scripts/odds_credits.py` prints the remaining credits and the
last requests without calling the API.

### Supported leagues — full support only

21 divisions: England 1–4, Scotland 1–4, Spain 1–2, Italy 1–2, Germany 1–2, France 1–2, Netherlands,
Belgium, Portugal, Turkey, Greece. Every enabled league has several seasons of results with shots, shots on
target, corners and cards (tested in `tests/test_rules.py`), so every model and market works.

Not included: the English National League (only ≈5 % of matches have corner statistics) and the 16 extra
Football-Data leagues (Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway,
Poland, Romania, Russia, Sweden, Switzerland, USA), whose files contain goals and closing 1X2 odds only — no
shots, corners or cards. The registry already supports calendar-year seasons (`calendar="calendar_year"`)
should a complete source for such leagues become available.

UEFA competitions and full calendars (cups) were researched: there is no free, licence-clean source with the
statistics the models need (shots, corners), so they are not included.

## Features and leakage protection

Matches are processed **day by day**: the features of every match on day *D* come from matches before *D*;
only then are the results of *D* added. Future fixtures and API rows get features but never update tables,
form, ratings or targets. Groups: league table, season statistics, home / away, form 5 / 10 / 15, venue form,
event frequencies, trends, rest days, head-to-head, attack-vs-defence and strength gaps, corner pressure,
league context, **Elo ratings** (K = 20, home advantage 60, goal-difference multiplier, 80 % carried over
between seasons, tier-dependent start) and **opponent-adjusted form** (average opponent Elo, results against
the Elo expectation, goals / corners minus what those opponents usually allow). Missing values stay `NaN`.
Odds are **not** model inputs (`USE_ODDS_FEATURES = False`). Model Analysis has a searchable feature dictionary.

Ablation on the validation season (same tuned parameters, lower is better): result log loss 1.0148 → 1.0125
with Elo + opponent-adjusted form; home goals Poisson deviance 1.1132 → 1.1092 with Elo; away goals 1.1281 →
1.1242 with Elo + opponent-adjusted form; corners unchanged (1.1725 — the new features did not help).

## Models

| Model | Algorithm | Target | Objective | Hyperopt metric |
|---|---|---|---|---|
| Result | `XGBClassifier` | H / D / A | `multi:softprob` | log loss |
| Home goals | `XGBRegressor` | home goals | `count:poisson` | Poisson deviance |
| Away goals | `XGBRegressor` | away goals | `count:poisson` | Poisson deviance |
| Corners | `XGBRegressor` | total corners | `count:poisson` + negative-binomial dispersion | Poisson deviance |

Temporal evaluation (current season 2026/27): **train** 2021/22–2023/24, **validation** 2024/25 (Hyperopt,
early stopping and every design decision), **test** 2025/26, **current** season predicted out-of-sample. Two
model sets are saved: the evaluation model (train + validation, scored on test, fills the test-season history)
and the production model (all completed seasons).

The test season has been looked at during development (v2–v4), so it is no longer a strictly untouched test.
`scripts/decision_study.py` therefore also runs a **walk-forward** check (train on all earlier seasons, evaluate
the next): result-model log loss 1.006 (2023/24), 1.012 (2024/25), 1.015 (2025/26) against 1.07–1.08 for the
frequency baseline.

### Decision rules (validation season only)

`scripts/decision_study.py` fits the models on the train seasons, predicts 2024/25 (7,129 matches) and compares
the rules; results are saved in `local_data/models/decision_study.json` (not shown on the website):

* **Draw rule** `argmax(P(H), 1.4·P(D), P(A))` — draw recall 0.2 % → 26 %, but accuracy 49.7 % → 47.6 %; the
  1X2 return at real odds moves only within noise (−5.9 % vs −5.2 %, ≈1.2 pp standard error). **Removed**: the named outcome is the most
  probable one; draw probabilities are unchanged and used by every market.
* **Calibration** — expected calibration error ≈ 1 % for 1X2, over 2.5, BTTS and double chance; temperature
  scaling changes log loss by less than 0.0001. Not adopted (not needed).
* **Dixon-Coles** low-score correction — rho unstable between the two halves of the season (−0.06 / +0.007), no
  gain in exact-score log loss or totals / BTTS Brier. Implemented (`DIXON_COLES`), **off**.

### v5 study of the XGBoost models (validation 2024/25, `experiments/xgb_v5`)

Paired bootstrap over matches (95 % interval of the loss difference):

* **Elo**: removing Elo makes every model clearly worse — 1X2 log loss 1.0118 → 1.0152, home goals Poisson
  deviance 1.1087 → 1.1161, away goals 1.1240 → 1.1283 (intervals exclude 0). Elo stays inside the models but is
  never shown to users as an argument.
* **Draws**: P(draw) is well calibrated (mean 25.8 % vs 26.0 % actual) but almost never the largest of the three
  (maximum 0.36, median gap to the favourite 0.19), so an honest arg-max names a draw in ~0.1 % of matches. Draws
  are more frequent between close teams (30 % vs 20 %), yet draw-specific features and a separate binary draw
  model (best blend weight 0) did not improve the probabilities. Draws therefore appear as Risk predictions,
  not by inflating P(draw).
* **Feature selection** (top 30 / 60 / 100 by gain) and **per-league corner dispersion**: no improvement.
* **Calibration** of Asian handicap lines (ECE ≤ 1.6 %, refund rates as predicted) and corner lines (ECE 1.7–2.5 %).

### Corners

Total corners use a Poisson XGBoost model with negative-binomial dispersion estimated on validation residuals.
Compared on validation: corner-specific features, shot-pressure features, separate home / away models (v3) and
Elo + opponent-adjusted corners (v4); the single total model with corner and shot features stays best.

## Markets

All goal markets come from one score matrix: `Pois(i; λ_home) · Pois(j; λ_away)`, with the home-win / draw /
away-win regions rescaled to the classifier probabilities, so 1X2, double chance, totals 1.5 / 2.5 / 3.5,
BTTS, exact scores and handicaps agree with each other.

### Handicap

* **Asian handicap, half lines** (±0.5, ±1.5, ±2.5): two outcomes, no refund — `HANDICAP_<line>`.
* **Asian handicap, whole lines** (±1, ±2): win / **refund (push)** / lose — the stake is returned when the
  result lands exactly on the line. FootPredict shows P(win), P(refund) and P(lose); EV =
  P(win)·odds − (1 − P(refund)).
* **European 3-way handicap** (whole lines, home / draw / away after the handicap) is a different market with
  different prices: stored separately as `EHANDICAP_<line>` and never mixed with the Asian one.
* Quarter lines (±0.25, ±0.75) are not supported.

Settlement: `WON`, `LOST` or `PUSH` (refund, profit 0, not a staked bet in ROI and not counted in the hit rate).
Stored v4 whole-line rows were European handicaps and were renamed by `scripts/migrate_v5_handicap.py`.

## Main and Risk predictions

Both exist for every match with model probabilities and are shown side by side on Match Details (Main green,
Risk orange). Neither ever invents a price: "effective odds" are the real odds when they exist, otherwise the
fair odds 1/p, used only to rank.

* **Main** — candidates with probability 40–85 % (no exact scores). Score = market reliability × probability ×
  odds preference × (1 + 0.5·EV) for a real price, × 0.8 without one. The odds preference is 1 for
  **1.40–2.50** and decays smoothly outside (1.37 or 2.60 barely lose anything, 1.10 a lot), so a near-certain
  1.10 is not chosen for its hit rate and a big EV alone does not win.
* **Risk** — the most sensible higher-odds alternative: any selection except the Main one and anything that
  wins on exactly the same scores (e.g. "home −0.5" when Main is "home win"), probability ≥ 12 %, preference
  for odds 3.5–8.0 (soft), EV bonus but **no EV requirement**. Validation 2024/25: present for 7,126 of 7,129
  matches, average odds 3.2, hit rate 31 %, ROI −6.1 %.
* The weights were chosen on the validation season only (`scripts/decision_study.py`: 36 Main and 72 Risk
  variants, real Bet365 prices). Main ROI −3.6 % vs −3.7 % for the v4 rule — the same within noise; on the test
  season the new rule gave −6.7 % vs −4.9 % (≈2 standard errors, and the test season cannot be used to choose).
  No rule beats the bookmaker margin.
* **EV** = probability × real odds − 1 (with the refund term for Asian whole lines): a theoretical edge.

## History, settlement and ROI

**Pre-match versions.** A prediction is created while the match has not kicked off (UK time). Every later
pre-match refresh (new data or odds) is a new **version** with its generation time; all versions are kept in
`local_data/predictions/versions.parquet`, and the history and ROI use the **last version before kick-off**.
After kick-off nothing is created or changed; a price collected after kick-off is never used as a pre-match
price; a postponed match gets a new version for its new kick-off.

**Prediction History** shows only predictions recorded before kick-off. The after-the-fact model evaluation
(test season 2025/26 and current-season backfill) is kept separately and shown on Model Analysis.

Stake 1 unit with a real price: win `odds − 1`, loss `−1`, refund / void `0` (not staked).
**ROI = net profit / number of staked predictions × 100**; the hit rate counts won / lost predictions (with or
without odds). Highest / lowest ROI market follows the active filters; below 30 priced predictions it is marked
"Small sample".

## Website

* **Matches** — leagues (Ukrainian names in UA), date strip, team search, club badges, 1X2 + totals with odds.
* **Match Details** — score; **Main and Risk side by side**; *Why this prediction?* (facts for that exact market,
  then the 2–4 factors that support it, a factor against only when it really matters); *Why this risk
  prediction?*; *Explain another market*; markets (Asian handicap: half lines and whole lines with refund);
  team comparison, form, H2H, rest days.
* **Prediction History** — recorded predictions only; period presets incl. previous month, a specific month /
  year and custom dates; Main / Risk / Market forecasts; summary by period (12 rows, *Show all*); scores as 2:1.
* **Leagues** — season selector drives table, matches and FootPredict statistics (season or all time).
* **Teams** — badge, equal KPI cards, statistics grouped Results / Attack / Defence / Corners, paging.
* **Analytics** — six charts: cumulative profit, rolling ROI, rolling hit rate, ROI by league, ROI by market,
  hit rate by market (with sample sizes); highest / lowest ROI market.
* **Model Analysis** — test-season metrics with explanations, draws, calibration, confusion matrix, model
  features (SHAP chart, gain in an expander, searchable feature dictionary), hyperparameter search, derived
  markets, historical evaluation.

Club badges: URLs from API-Football team data (`scripts/build_team_logos.py`, map in
`local_data/teams/logos.json`; images are not stored in the repository), shown unmodified and small; a neutral
shield is shown when no badge is known. Badges are the clubs' trademarks.

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
2. Register at <https://oddspapi.io> (free plan) and copy the API key.
3. Optional fallback: <https://the-odds-api.com> (free plan).
4. Copy `.env.example` to `.env` (never commit `.env`) and fill in:

   ```
   API_FOOTBALL_KEY=your-api-football-key
   ODDSPAPI_KEY=your-oddspapi-key
   ODDS_API_KEY=your-odds-api-key
   ```

5. Run `python scripts/run_pipeline.py --mode results` (free) or `--mode odds` (uses the odds quotas).
6. For the cloud update (GitHub Actions) add the same values as GitHub repository secrets
   (Settings → Secrets and variables → Actions).

Without keys the pipeline still works with Football-Data only (results, fixtures and Bet365 prices for
1X2, total 2.5 and half-line handicaps).

## Updating and retraining

```bash
python scripts/run_pipeline.py --mode results              # fixtures + results, 0 Odds API credits
python scripts/run_pipeline.py --mode odds --credit-budget 30   # + real odds within the budget
python scripts/run_pipeline.py --mode full                 # both
python scripts/run_pipeline.py --mode odds --odds-league E0          # one league
python scripts/run_pipeline.py --mode odds --odds-date 2026-10-18    # one match day
python scripts/run_pipeline.py --mode odds --odds-match <match_id> --force   # re-price one match
python scripts/oddspapi_history.py --since 2026-01-01      # historical prices of played 2026 matches
python scripts/run_pipeline.py --offline                   # local files only
python scripts/odds_credits.py                             # credits left (no API call)
python scripts/retrain_models.py                           # retrain the four models (≈1 h)
python scripts/decision_study.py                           # re-run the validation study of Main / Risk
python scripts/regenerate_predictions.py                   # re-apply the rules without retraining
```

* `run_pipeline.py` updates data and predictions; it never retrains. Order: providers → Football-Data →
  features → settlement → predictions of matches that have not kicked off.
* `retrain_models.py` re-tunes and refits the models; run it when a season ends (the split moves forward) or
  after a few hundred new results, then `decision_study.py`.

### GitHub Actions and deployment

The deployed app (Streamlit Community Cloud) is built from `main`. `.github/workflows/update-data.yml` has
**no schedule** (to save credits): start it in *Actions → Update data and predictions → Run workflow* and
choose the mode — `results` (free; after match days for results, before them for fixtures), `odds` (run about
a day before the matches, and again a few hours before kick-off for late prices) or `full` — and optionally one
league, one day or one match and *force*. It commits the new data to `main`, prints the
credit ledger, and Streamlit redeploys. A local checkout is synchronised with `git pull`;
`E2E_BASE_URL=<app url>/~/+ pytest tests/e2e --e2e -k sync` checks that the deployed app serves the same data.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                     # unit tests
python -m playwright install chromium
pytest tests/e2e --e2e                     # browser tests (starts the app)
E2E_BASE_URL=https://football-prediction-diploma.streamlit.app/~/+ pytest tests/e2e --e2e   # deployed app
```
