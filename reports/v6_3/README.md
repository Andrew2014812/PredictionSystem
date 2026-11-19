# FootPredict v6.3 — research reports

Thirteen research reports and one summary, in Russian. Every report replays the whole season 2026/27 (two decision
moments: `close` — last quote before kick-off, primary; `t09` — prices known at the daily run) and compares its
candidates with Baseline B (production models and Main rule replayed) and Baseline A (the site's history), with paired
bootstrap intervals and Holm's correction. Every number in a PDF is read at build time from the result files of its
experiment folder (`experiments/v6_3/<study>/results/`); each folder also has a short `findings.md` and the
pre-registered B candidates (`results/preregistration.json`). Shared replay and methodology: `experiments/v6_3/README.md`
and `experiments/v6_3/replay/`; the PDF layout is `scripts/report_kit.py`.

Start with **[FootPredict_v6.3_Research_Summary.pdf](FootPredict_v6.3_Research_Summary.pdf)**.

| Report | What it studies | Code, results, figures | Build |
|---|---|---|---|
| [FootPredict_v6.3_Research_Summary.pdf](FootPredict_v6.3_Research_Summary.pdf) | All 13 studies side by side, every B candidate against Baseline B, the key table "Current Production vs Best Validated Experimental Candidate", 12 pre-planned combinations (stage 6), model improvements, Squad Strength, Recent Season Weighting, statistical reliability, implemented / rejected, open problems, conclusions for the diploma | `experiments/v6_3/summary/` | `… summary.run`, then `… summary.build_pdf` |
| [Main_Selection_Research.pdf](Main_Selection_Research.pdf) | Study 1 — rules for the one Main per match: odds floors (1.40 with a not-a-bet fallback), real price mandatory, preferred odds, probability power, EV weight, calibrated EV, market reliabilities, market probability, over-estimation corrections, alternative rankings | `experiments/v6_3/main_selection/` | `… main_selection.prepare`, `… .run`, `… .figures`, `… .preregister`, `… .build_pdf` |
| [Probability_Calibration_Research.pdf](Probability_Calibration_Research.pdf) | Study 2 — Platt, temperature, beta, isotonic and histogram calibration by scope and fitting source; calibration of the chosen Main and the winner's curse | `experiments/v6_3/calibration/` | `… calibration.run`, `… .figures`, `… .preregister`, `… .build_pdf` |
| [Market_Aware_Main_Research.pdf](Market_Aware_Main_Research.pdf) | Study 3 — de-margining methods, information of the model beyond the market, model–market blends, market only, disagreement filter, blended EV | `experiments/v6_3/market_aware/` | `… market_aware.run`, `… .build_pdf` |
| [Market_Profitability_Research.pdf](Market_Profitability_Research.pdf) | Study 4 — ROI of every market group (all priced candidates and Main), market priority, reliability boosts, exclusions, empirical-Bayes and lower-bound history rules | `experiments/v6_3/market_profitability/` | `… market_profitability.run`, `… .figures`, `… .preregister`, `… .build_pdf` |
| [Timing_CLV_Research.pdf](Timing_CLV_Research.pdf) | Study 5 — Main 48 h before kick-off, at the daily run and at the close; closing line value; market movement towards the model; live prediction versions | `experiments/v6_3/timing/` | `… timing.run`, `… .build_pdf` |
| [Quality_Coverage_Research.pdf](Quality_Coverage_Research.pdf) | Study 6 — model vs market by data sufficiency (early season, newcomers, phantom clubs, missing statistics, leagues, prices at T), feature-group stability, mitigations (market blends, priors) | `experiments/v6_3/quality/` | `… quality.oos`, `… .run`, `… .build_pdf` |
| [Recent_Season_Weighting_Research.pdf](Recent_Season_Weighting_Research.pdf) | Study 7 — sample weights of the current / last season, exponential decay, windows, monthly and every-250-match refits, recency + current form | `experiments/v6_3/recent_weighting/` | `heavy.sh … recent_weighting.fits`, then `… .run`, `… .build_pdf` |
| [Alternative_ML_Research.pdf](Alternative_ML_Research.pdf) | Study 8 — LightGBM, CatBoost, random forest, extra trees, GLM, HistGradientBoosting, ensembles and monthly refits against production XGBoost on all four tasks | `experiments/v6_3/alt_ml/` | `… alt_ml.tune`, `… .fit26`, `… .run`, `… .build_pdf` |
| [Meta_Selection_Research.pdf](Meta_Selection_Research.pdf) | Study 9 — a meta-model (logistic / boosting) that picks the Main among candidates, with permutation and in-sample controls | `experiments/v6_3/meta_selection/` | `… calibration.run` (first), `… meta_selection.run`, `… .figures`, `… .preregister`, `… .build_pdf` |
| [Goal_Model_Research.pdf](Goal_Model_Research.pdf) | Study 10 — score-matrix construction (reconciliation, Dixon–Coles, negative binomial, IPF), dynamic ratings, rating blends, one team-goals model, monthly refits | `experiments/v6_3/goals/` | `… goals.run --stage ratings`, `heavy.sh … goals.base`, `… .run --stage eval`, `… .build_pdf` |
| [Corners_Research.pdf](Corners_Research.pdf) | Study 11 — new corner features, XGBoost / LightGBM / CatBoost / Poisson GLM / ensembles, home–away models with a copula, count distributions, line calibration, corner betting and Main | `experiments/v6_3/corners/` | `… corners.data`, `… .fit`, `… .run`, `… .build_pdf` |
| [Squad_Strength_Research.pdf](Squad_Strength_Research.pdf) | Study 12 — data sources for line-ups and player values, typical XI, player and team strength, dynamic and pre-season squad features on earlier seasons and on 2026/27 | `experiments/v6_3/squad/` | `… squad.build_features`, `… .fits`, `… .run`, `… .build_pdf` |
| [Market_Consensus_Research.pdf](Market_Consensus_Research.pdf) | Study 13 — availability of several bookmakers, consensus rules on Football-Data 2021/22–2026/27, decision-time consensus, model + consensus blends, dispersion filter, best price | `experiments/v6_3/consensus/` | `… consensus.run`, `… .build_pdf` |

`…` stands for `.venv/bin/python -m experiments.v6_3.<study>` (`heavy.sh` is `experiments/v6_3/heavy.sh <timeout> <log>`);
run everything from the repository root. Caches and logs (`experiments/v6_3/*/cache/`, `logs/`) are not in git;
`Report.check` renders every page to `reports/v6_3/<name>_pages/` for a visual check.

Other folders: `design/` — the v6.3 design notes (Main result indicator variants), `screenshots/` — the site before
and after the v6.3 interface changes.
