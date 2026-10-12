# FootPredict

Football prediction website built on machine learning. Diploma project:
*«Інформаційна система прогнозування результатів спортивних подій на основі машинного навчання»*.

FootPredict:

* turns historical Football-Data results into leak-free pre-match features;
* trains four specialised **XGBoost** models (match result, home goals, away goals, total corners) with Elo and
  opponent-adjusted features;
* derives consistent probabilities for 1X2, double chance, totals, BTTS, exact score, handicap and corners;
* attaches **real bookmaker odds only** (never derived or simulated prices) and computes expected value;
* picks one **Main prediction** per match and, when it qualifies, an optional **Risk prediction**;
* explains every prediction with **SHAP** in plain language (EN / UA);
* records predictions **before kick-off only** (frozen once the match starts), settles them and reports profit /
  ROI of the Main strategy separately for recorded predictions and for historical model evaluation;
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
nothing is derived, estimated or simulated, and the history keeps the provider and bookmaker of every price.

| Source | Used for | Free plan / licence | Status |
|---|---|---|---|
| [Football-Data](https://www.football-data.co.uk) | history of results, shots, corners, cards; Bet365 1X2, total 2.5, Asian handicap (half lines used), pre-match and closing | free download | base of the models (confirmed) |
| [The Odds API](https://the-odds-api.com) v4 | fixtures (free), 1X2 + totals per league, and per match BTTS, double chance, goal lines 0.5–5.5, corner lines (UK bookmakers) | 500 credits / month | used, budgeted (confirmed live) |
| [API-Football](https://www.api-football.com) v3 | fixtures and final scores (yesterday … tomorrow) | 100 requests / day; free odds only for seasons 2022–2024 | used for results; odds off (confirmed) |
| The Odds API historical, Sportmonks, football-data.org, apifootball.com, odds-api.io, Pinnacle, Betfair | — | paid, no useful free odds, paused free keys, academic / personal only | not used (documentation) |

Priority when several providers price the same selection: The Odds API (fresh pre-match) → Football-Data.
Every response is cached in `local_data/api/`; the website never calls an API — it only reads files written by
the pipeline. Pre-match odds are archived with their collection time and never replaced after kick-off.

### The Odds API credits

Costs: `/events` 0 credits; `/sports/{league}/odds` 1 credit per market (h2h + totals = 2 per league);
`/events/{id}/odds` 1 credit per returned market (≈4 per match for BTTS, double chance, goal and corner lines).
The free 500 credits cover the 1X2 / totals of every league a few times a week plus extended markets for about
100 top-league matches a month — full coverage of all markets is not possible for free.

Safeguards (`src/providers/ledger.py`, committed as `local_data/api/the_odds_api/ledger.json` because the CI
cache does not persist): a league is not asked again within 20 h and a match's extended markets are bought once
— also when the answer was empty; every run has a credit budget (`--credit-budget`, default 60) and a monthly
reserve of 30 credits is never spent. `python scripts/odds_credits.py` prints the remaining credits and the
last requests without calling the API.

### Team names

Provider names are mapped to Football-Data names by `src/providers/aliases.py`: normalisation (accents,
apostrophes, club prefixes such as FC / KVC / SpVgg / TSG, numbers), an alias table, and a unique word-subset
match inside the same league (ambiguous names are never guessed). Unmatched names are written to
`local_data/api/unmatched_teams.csv`; `python scripts/check_team_aliases.py` checks all cached responses.

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
the next): result-model log loss 1.007 (2023/24), 1.013 (2024/25), 1.015 (2025/26) against 1.07–1.08 for the
frequency baseline.

### Decision rules (validation season only)

`scripts/decision_study.py` fits the models on the train seasons, predicts 2024/25 (7,129 matches) and compares
the rules; results are on the Model Analysis page (tab *Decision rules*):

* **Draw rule** `argmax(P(H), 1.4·P(D), P(A))` — draw recall 0.6 % → 27 %, but accuracy 49.6 % → 47.1 % and the
  1X2 return at real odds practically unchanged (−6.0 % vs −5.9 %). **Removed**: the named outcome is the most
  probable one; draw probabilities are unchanged and used by every market.
* **Calibration** — expected calibration error ≈ 1 % for 1X2, over 2.5, BTTS and double chance; temperature
  scaling improves log loss by 0.0002. Not adopted (not needed).
* **Dixon-Coles** low-score correction — rho unstable between the two halves of the season (−0.06 / +0.01), no
  gain in exact-score log loss or totals / BTTS Brier. Implemented (`DIXON_COLES`), **off**.

### Corners

Total corners use a Poisson XGBoost model with negative-binomial dispersion estimated on validation residuals.
Compared on validation: corner-specific features, shot-pressure features, separate home / away models (v3) and
Elo + opponent-adjusted corners (v4); the single total model with corner and shot features stays best.

## Markets

All goal markets come from one score matrix: `Pois(i; λ_home) · Pois(j; λ_away)`, with the home-win / draw /
away-win regions rescaled to the classifier probabilities, so 1X2, double chance, totals 1.5 / 2.5 / 3.5,
BTTS, exact scores and handicaps agree with each other.

**Handicap** (home team's perspective): whole lines ±1, ±2 are 3-way European handicaps (home / draw / away
after adding the line to the home goals); half lines ±0.5, ±1.5, ±2.5 are 2-way (a draw after the handicap
is impossible). Settlement is tested for both types.

## Odds, EV and the Main prediction

* **Odds** = a real bookmaker price from a provider; if a selection has no real price, the interface shows
  no odds and no EV.
* **EV** = model probability × real odds − 1 (theoretical edge, not profit).
* **Main prediction** — one per match. Candidates: probability 40–85 %, no exact scores. Score =
  market reliability × probability × odds preference, where the preference is 1 for odds **1.40–2.50** and
  decays smoothly outside (a soft preference, not a hard band). A candidate with real odds and EV ≥ 0 is
  preferred (value pick); without one the best candidate on probability is used (fair odds 1 / p for the
  preference). Chosen on validation among 144 variants (softness, EV weight, minimum EV, minimum probability):
  ROI −4.2 % with 78 % of Main predictions priced, vs −5.8 % / 56 % for the v3 rule (hard band 1.30–3.00,
  EV ≥ 3 %). Every variant loses against the bookmaker margin; the difference is within about one standard error.
* **Risk prediction** — optional, real odds only: odds 3.0–8.0, probability ≥ 22 %, EV ≥ +10 %, reliable
  market; hidden when nothing qualifies.

## History, settlement and ROI

**Pre-match only.** The pipeline predicts a match as `live` only while its kick-off (UK time) is in the
future; once the match has started its recorded prediction is frozen and never rewritten. Every snapshot
stores kick-off, generation time (UTC), first generation time, model version, Main / Risk, probabilities,
odds and their provider / bookmaker. `scripts/migrate_prediction_history.py` moved 64 v3 "live" predictions
that had been generated after kick-off to `backfill`.

Two datasets are never mixed:

* **Recorded predictions** (`live`) — saved before kick-off: the real track record;
* **Historical evaluation** (`backtest` test season, `backfill` current season) — the model run afterwards on
  pre-match data.

After the match: `WON / LOST` (`VOID` only for a match that never took place). Stake 1 unit: win `odds − 1`,
loss `−1`. **ROI = net profit / number of staked predictions × 100**, counting only predictions with real odds;
the hit rate counts every settled prediction. Best / weakest market (or league, inside one market) follows the
active filters and needs at least 30 priced predictions.

## Website

* **Matches** — leagues (Ukrainian names in UA), date strip, team search, 1X2 + totals with real odds.
* **Match Details** — header → result (score, Main / Risk won or lost, odds, profit) → Main → *Why this
  prediction?* (football facts first, then 3–5 model factors with their share of influence) → Risk → *Why
  this risk prediction?* → markets (one handicap table with a tooltip) → team comparison, form, home / away,
  H2H, rest days.
* **Prediction History** — Recorded vs Historical evaluation, filters Period / Prediction type (Main by
  default; Market forecasts explained) / League / Market, summary by period always visible.
* **Leagues**, **Teams** — statistics grouped Results / Attack / Defence / Corners, readable charts, paging.
* **Analytics** — filters Prediction set / Period / League / Data; performance over time with its own grouping.
* **Model Analysis** — metrics with ↑ / ↓ (higher / lower is better) and explanations, baselines,
  calibration, feature dictionary, SHAP as a share of total influence, decision-rule study, walk-forward.

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

4. Run `python scripts/run_pipeline.py --mode results` (free) or `--mode odds` (spends credits).
5. For the cloud update (GitHub Actions) add the same two values as GitHub repository secrets
   (Settings → Secrets and variables → Actions) or with `gh secret set API_FOOTBALL_KEY`.

Without keys the pipeline still works with Football-Data only (results, fixtures and Bet365 prices for
1X2, total 2.5 and half-line handicaps).

## Updating and retraining

```bash
python scripts/run_pipeline.py --mode results              # fixtures + results, 0 Odds API credits
python scripts/run_pipeline.py --mode odds --credit-budget 30   # + real odds within the budget
python scripts/run_pipeline.py --mode full                 # both
python scripts/run_pipeline.py --offline                   # local files only
python scripts/odds_credits.py                             # credits left (no API call)
python scripts/retrain_models.py                           # retrain the four models (≈1 h)
python scripts/decision_study.py                           # re-run the validation study
```

* `run_pipeline.py` updates data and predictions; it never retrains. Order: providers → Football-Data →
  features → settlement → predictions of matches that have not kicked off.
* `retrain_models.py` re-tunes and refits the models; run it when a season ends (the split moves forward) or
  after a few hundred new results, then `decision_study.py`.

### GitHub Actions and deployment

The deployed app (Streamlit Community Cloud) is built from `main`. `.github/workflows/update-data.yml` has
**no schedule** (to save credits): start it in *Actions → Update data and predictions → Run workflow* and
choose the mode — `results` (free; before / after match days), `odds` (spends credits — run about a day before
the matches you care about) or `full` — and the credit budget. It commits the new data to `main`, prints the
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
