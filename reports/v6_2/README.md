# FootPredict v6.2 — research reports

Thirteen research reports and one summary, in Russian. Each report is a standalone document with a title page, contents, goal, hypotheses, data, methods, results with confidence intervals, negative results, limitations, A / B / C recommendations and the commands to reproduce it. Every number in a PDF is read from the result files of its experiment folder at build time (`experiments/v6_2/<study>/results/`); each folder also has a short `findings.md`. The shared PDF layout is in `scripts/report_kit.py`; the common rules are in `experiments/v6_2/README.md`.

Start with **[Research_Summary.pdf](Research_Summary.pdf)**: it compares the conclusions of all thirteen studies, answers the twelve summary questions, lists every A / B / C decision with what is already in production, and the pre-registered forward tests.

| Report | What it studies | Code, results, figures | Build |
|---|---|---|---|
| [Research_Summary.pdf](Research_Summary.pdf) | All thirteen studies side by side: what was learned, which data helped, xG, models, markets, Main over-estimation, ROI, the bookmaker comparison, negative results, what to adopt / not adopt, future work; A / B / C table, forward hypotheses, caveats, disagreements between studies | `experiments/v6_2/summary/` | `python -m experiments.v6_2.summary.build_pdf` |
| [XG_Research.pdf](XG_Research.pdf) | xG / xGA (available only in 2026/27): audit of the columns, pre-match xG ratings, 18 variants in a weekly expanding window, effect on Main, corners, the shot-based proxy on the long history | `experiments/v6_2/xg/` | `python -m experiments.v6_2.xg.run`, then `python -m experiments.v6_2.xg.build_pdf` |
| [Data_Quality_Research.pdf](Data_Quality_Research.pdf) | The data chain from sources to predictions: raw files, team histories, an independent recomputation of the rolling features, the prediction store, clubs without history, newcomer variants, production defects D1–D6 | `experiments/v6_2/data_quality/` | `python -m experiments.v6_2.data_quality.run`, then `python -m experiments.v6_2.data_quality.build_pdf` |
| [Odds_Timing_Research.pdf](Odds_Timing_Research.pdf) | When each Main / Risk price was obtained (recorded at prediction time, added later, closing, retrospective), ROI by price class, opening → closing drift, model → line movement, CLV | `experiments/v6_2/odds_timing/` | `python -m experiments.v6_2.odds_timing.run && python -m experiments.v6_2.odds_timing.build_pdf` |
| [Bookmaker_Benchmark_Research.pdf](Bookmaker_Benchmark_Research.pdf) | Margin removal methods, model vs. the de-margined Bet365 / Pinnacle / Betfair prices (1X2, total 2.5, Asian handicap), encompassing tests, model–market hybrids, value betting, the multi-market check of 2026/27 | `experiments/v6_2/bookmaker/` | `bash experiments/v6_2/bookmaker/launch.sh run`, `python -m experiments.v6_2.bookmaker.preregister`, then `python -m experiments.v6_2.bookmaker.build_pdf` |
| [Main_Calibration_Research.pdf](Main_Calibration_Research.pdf) | Over-estimation of the chosen Main: calibration after selection, before selection, ranking rules, fewer markets, league corrections; the shown Main of 2026/27 | `experiments/v6_2/main_calibration/` | `python -m experiments.v6_2.main_calibration.run`, `… .preregister`, then `… .build_pdf` |
| [Risk_Selection_Research.pdf](Risk_Selection_Research.pdf) | Why the model EV of Risk is positive while ROI is negative (selection effect, favourite–longshot, margin), twelve selection rules, EV with refunds on Asian lines | `experiments/v6_2/risk/` | `python -m experiments.v6_2.risk.data && python -m experiments.v6_2.risk.run`, `… .preregister`, then `… .build_pdf` |
| [Market_Models_Research.pdf](Market_Models_Research.pdf) | Separate models per market (binary, goal-difference and total multiclass, exact score, corners) and stacking against the current derivation from the score matrix; coherence and cost | `experiments/v6_2/market_models/` | `python -m experiments.v6_2.market_models.data`, `… .direct`, `… .run`, then `… .build_pdf` |
| [Goal_Distributions_Research.pdf](Goal_Distributions_Research.pdf) | Score distributions behind the markets (Poisson, Dixon–Coles, bivariate Poisson, negative binomial, Conway–Maxwell–Poisson, Frank copula, Maher ratings, XGBoost + Maher blend), with and without reconciliation to 1X2; production code-path checks | `experiments/v6_2/goal_distributions/` | `python -m experiments.v6_2.goal_distributions.maher`, `… .run`, then `… .build_pdf` |
| [League_Specialization_Research.pdf](League_Specialization_Research.pdf) | How evenly the global model works across the 21 leagues; calibration per league / group with shrinkage, separate and fine-tuned models, goal multipliers, transfer between leagues | `experiments/v6_2/leagues/` | `timeout 21600 python -m experiments.v6_2.leagues.run`, `… .decisions`, then `… .build_pdf` |
| [Temporal_Features_Research.pdf](Temporal_Features_Research.pdf) | Leakage tests of the time features, then short windows, exponential averages, opponent adjustment, home / away form, slopes, Elo trend, fixture density, streaks; simple averages vs. dynamics; form vs. the market | `experiments/v6_2/temporal/` | `timeout 21600 python -m experiments.v6_2.temporal.run`, `… .decisions`, then `… .build_pdf` |
| [Prediction_Uncertainty_Research.pdf](Prediction_Uncertainty_Research.pdf) | Seed / bootstrap / subsample ensembles, Venn–Abers and conformal sets: does uncertainty predict that a Main loses more than stated; filters, bagging, version stability | `experiments/v6_2/uncertainty/` | `bash experiments/v6_2/uncertainty/launch.sh ensemble`, `… .prepare && … .run`, `… .figures`, then `… .build_pdf` |
| [Forward_Validation_Research.pdf](Forward_Validation_Research.pdf) | The registry of frozen B hypotheses, append-only shadow predictions next to production (`src/research/`), a replay of the mechanics, storage and run time, when each hypothesis can be judged | `experiments/v6_2/forward/` | `python -m pytest tests/test_shadow.py -q`, `python -m experiments.v6_2.forward.simulate && … .run`, `… .figures`, then `… .build_pdf` |
| [Data_Sources_Report.pdf](Data_Sources_Report.pdf) | What Football-Data, API-Football, OddsPapi and The Odds API deliver; API-Football lead time; OddsPapi coverage and cost; the daily run time; a recheck of the 2026/27 Main / Risk history | `experiments/v6_2/data_sources/` | `python -m experiments.v6_2.data_sources.run && python -m experiments.v6_2.data_sources.build_pdf` |

`…` stands for `python -m experiments.v6_2.<study>`. Run everything from the repository root with the project `.venv`. Long runs use `n_jobs=2`, a `timeout` and a log file; large intermediate files go to `experiments/v6_2/<study>/cache/` (not in git). `Report.check` renders every page to `reports/v6_2/<name>_pages/` (not in git) for a visual check.

## Common methodology

- **Baseline** is FootPredict v6.1 as it runs in production (same features, parameters, market rules and Main / Risk selection).
- **Time order.** Every model is fitted only on matches before the ones it predicts (walk-forward by season, or by week / month inside 2026/27). No feature uses the match's own result, statistics, xG or closing price.
- **Prices.** Only real bookmaker prices that existed when the decision was made: Football-Data opening / closing columns for past seasons, the recorded prices and the OddsPapi archive for 2026/27.
- **Seasons.** Variants are selected on 2023/24 and 2024/25. 2025/26 is reported "for information". 2026/27 was already looked at by the v6.1 studies, so it is **not** an untouched holdout and is never called one. xG exists only from 2026/27, so the xG study works inside that season.
- **Same matches, intervals, corrections.** Compared variants are evaluated on the same matches, with bootstrap 95 % intervals resampled by match; with many variants p-values are Holm-corrected and the number of variants tried is reported (about 211 across the studies; no correction is applied between studies).
- **Decisions.** A — ready to adopt; B — promising, needs forward evidence; C — not recommended. Research never changes production by itself.
- **Forward validation.** B hypotheses were frozen on 09.11.2026 in `results/preregistration.json` files committed before the matches they will be judged on; the daily run records them as shadows (`src/research/`, `local_data/research/`). No forward result exists yet; the earliest dates are in the summary and in `Forward_Validation_Research.pdf`.

## Outcome for production

| Area | Decision |
|---|---|
| Models (1X2, goals, corners) | unchanged — no variant improved them robustly; xG X5, the XGBoost + Maher goal blend and the stacked total model are B (forward test) |
| Main | selection unchanged; calibration of the shown probability is B (shadows `maincal_ps_platt_monthly`, `maincal_ps_group`) |
| Risk | selection unchanged; market-stacked rule and market probability are B (shadows); refund fix for an unpriced whole handicap line adopted |
| League specialisation, new time features, separate market models, uncertainty filters | not adopted (C) |
| Price timing | adopted: every stored Main / Risk price carries its timing class and UTC time, the price tooltip says which it is, History adds the ROI of prices on offer at prediction time |
| Data fixes | adopted: D1–D6 (temporary club spellings, stale id conflicts, withdrawn results, a match moved to an earlier day, VOID without a price, API-Football aliases), Edinburgh City, OddsPapi request counting, price of a moved match |
| Forward shadows | adopted: recorded in the daily run (stage `shadow`), never read by the website |
| Showing the de-margined market probability next to the model | recommended (A), not yet done |
| Exact daily start at 03:17 | code ready (`workflow_dispatch` under the once-a-day guard); needs an outside cron set up by the owner |

None of the studied changes improved Main or Risk ROI significantly. The reasons (market efficiency, margin, selection effect) are in `Research_Summary.pdf`, section 9. The data and pricing fixes are listed in the main `README.md` ("Data diagnostics and history fixes (v6.2)"); the design changes in "Design v6.2, the logo and Streamlit branding (v6.2)".

## Other data in this folder

- `data/contrast_before.json`, `data/contrast_after.json` — text-contrast audits (WCAG 4.5:1, large text 3:1) of every page in both themes at 1920×1080, measured in a real browser with `scripts/ui_contrast_audit.py`: v6.1 had 192 texts below the threshold in the dark theme and 26 in the light one, v6.2 none.
- `screenshots/before/` and `screenshots/after/` — the seven pages (Matches, a match, History, Leagues, Teams, Analytics, Model Analysis) before (v6.1 design) and after (v6.2), both themes, at 1366×768 and 1920×1080, each also at 125 % zoom (56 images per folder). Made with `scripts/ui_screenshots.py`.
