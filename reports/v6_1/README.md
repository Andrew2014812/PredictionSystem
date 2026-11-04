# FootPredict v6.1 — research reports

Six research reports in Russian. Each one is a standalone document with a title page, contents, methods, every experiment marked as run or not run, results, limitations, recommendations and the commands to reproduce it. Every number in a PDF is read from the result files of its experiment folder at build time. The shared PDF layout is in `scripts/report_kit.py`.

| Report | What it studies | Code, results, figures | Build |
|---|---|---|---|
| [Main_Strategy_Research.pdf](Main_Strategy_Research.pdf) | Why Main loses money at real odds (calibration, winner's curse, information in the market), strategies A–G, betting decision vs. prediction, bookmakers and CLV, other learners | `experiments/v6_1/main/` | `python -m experiments.v6_1.main.build_pdf` |
| [Risk_Strategy_Research.pdf](Risk_Strategy_Research.pdf) | Risk strategies (value, calibrated, market-specific, conservative, high-odds), odds bands, EV floors, calibration of rare outcomes | `experiments/v6_1/risk/` | `python -m experiments.v6_1.risk.build_pdf` |
| [Draw_Research.pdf](Draw_Research.pdf) | The draw rule in use vs. arg-max, new draw models, draw as Main, the confusion matrix shown on Model Analysis | `experiments/v6_1/draw/` | `python -m experiments.v6_1.draw.build_pdf` |
| [Corners_Research.pdf](Corners_Research.pdf) | Corners models (XGBoost, CatBoost, LightGBM, GLMs, hierarchical, home/away, ensembles), distributions, line calibration, real-price check | `experiments/v6_1/corners/` | `python -m experiments.v6_1.corners.build_pdf` |
| [Explainability_Research.pdf](Explainability_Research.pdf) | How to explain the probability of a derived market (SHAP of the source model, the model chain, sensitivity, attribution for a composition of models, football facts) and what the *Why this prediction?* card shows | `experiments/v6_1/explain/` | `python -m experiments.v6_1.explain.build_pdf` |
| [Odds_Coverage_Audit.pdf](Odds_Coverage_Audit.pdf) | Real-odds coverage of 2026/27 by league, market and line; the reason for every missing price; Barcelona / Espanyol, Turkey, Scotland; API checks and quotas; before / after the v6.1 fixes | `experiments/v6_1/odds/`, `reports/v6_1/data/` | `python scripts/v61_odds_audit.py --with-repairs --out reports/v6_1/data/odds_audit_after.json`, then `python -m experiments.v6_1.odds.build_pdf` |

## Common methodology

- **Walk-forward.** Every model is refitted only on seasons before the season it predicts, with the production parameters.
- **Selection.** Variants are chosen on 2023/24 and 2024/25 only. 2025/26 was looked at in earlier development, so it is reported "for information" and is never called independent.
- **Independent check.** The 2026/27 season played so far is the independent check. Uses the stored production history: probabilities computed before kick-off, and real prices recorded at the time or restored from the archive.
- **Pre-registration.** Main and Risk pre-registered their chosen strategy and the adoption criteria in `results/preregistration.json`. Those files were committed before the 2026/27 evaluation.
- **Prices.** Only real pre-match bookmaker prices are used. Every compared strategy is evaluated on the same matches, with bootstrap 95 % confidence intervals. A negative result is reported as such.

## Outcome for production

| Area | Decision |
|---|---|
| Main | unchanged |
| Risk | unchanged |
| Draw rule | unchanged |
| Corners model | unchanged |

None of these met its pre-registered adoption criteria on the independent season. The reasons and the rules a future change would have to meet are in each report.

The data fixes found by the odds audit are in the code; the report lists them.

## Other data in this folder

- `data/` — the audits before and after the fixes and the API checks of 04.11.2026.
- `screenshots/before/` and `screenshots/after/` — every page, both themes, at 1366×768 and 1920×1080, and 1366×768 at 125 % zoom. Made with `scripts/ui_screenshots.py`.
