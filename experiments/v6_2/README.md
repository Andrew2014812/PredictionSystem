# FootPredict v6.2 — research experiments

Each study has its own folder `experiments/v6_2/<study>/` and its own report in `reports/v6_2/`.
Every number in a report is read from the result files of its folder when the PDF is built (`scripts/report_kit.py`).

| Folder | Report |
|---|---|
| `xg/` | `XG_Research.pdf` |
| `data_quality/` | `Data_Quality_Research.pdf` |
| `odds_timing/` | `Odds_Timing_Research.pdf` |
| `bookmaker/` | `Bookmaker_Benchmark_Research.pdf` |
| `main_calibration/` | `Main_Calibration_Research.pdf` |
| `risk/` | `Risk_Selection_Research.pdf` |
| `market_models/` | `Market_Models_Research.pdf` |
| `goal_distributions/` | `Goal_Distributions_Research.pdf` |
| `leagues/` | `League_Specialization_Research.pdf` |
| `temporal/` | `Temporal_Features_Research.pdf` |
| `uncertainty/` | `Prediction_Uncertainty_Research.pdf` |
| `forward/` | `Forward_Validation_Research.pdf` |
| `data_sources/` | `Data_Sources_Report.pdf` |
| `summary/` | `Research_Summary.pdf` |

## Layout of a study folder

- `run.py` — runs every experiment of the study (`python -m experiments.v6_2.<study>.run`), writes `results/`.
- `build_pdf.py` — builds the report from `results/` and `figures/` (`python -m experiments.v6_2.<study>.build_pdf`).
- `results/*.json`, `results/*.csv` — raw results; `results/run_info.json` holds the git SHA, the data version
  (row counts and mtimes of the input files), library versions, parameters and the run time.
- `figures/*.png` — the charts used by the report.
- `cache/` — large intermediate files (ignored by git).
- `findings.md` — a short plain-text summary of the results and the A / B / C decision.

## Common methodology

1. **Baseline** is FootPredict v6.1 as it runs in production: the same features, parameters, market rules
   and Main / Risk selection.
2. **Time order.** Every model is fitted only on matches played before the matches it predicts
   (walk-forward by season, or by month inside 2026/27). No feature of a match uses that match's result,
   statistics, xG or closing price.
3. **Prices.** A decision may use only a price that existed when the decision was made. Real bookmaker
   prices only: Football-Data closing / pre-match columns for past seasons, the stored recorded prices and
   the OddsPapi archive (`local_data/odds/odds.parquet`, `collected_at` / `priced_at`) for 2026/27.
   No derived or invented prices.
4. **Seasons.**
   - Variants are selected on 2023/24 and 2024/25.
   - 2025/26 was used by earlier studies: it is reported "for information", never as independent.
   - 2026/27 was also used repeatedly by the v6.1 studies. It is **not** an untouched holdout and is never
     called one.
   - The only truly independent check is forward: matches played after a hypothesis was frozen
     (`results/preregistration.json`, committed before those matches). See `forward/`.
   - xG exists only from 2026/27, so the xG study works inside that season (time-ordered splits) and says so.
5. **Same matches.** Every compared variant is evaluated on the same matches.
6. **Uncertainty.** Bootstrap 95 % intervals (resampled by match). With many variants, p-values are
   corrected (Holm) and the number of variants tried is reported.
7. **Breakdowns** by league, market and season where the sample allows; small samples are marked.
8. **Decisions** use three categories: A — ready to adopt; B — promising, needs forward evidence;
   C — not recommended. Research never changes production by itself.
9. **No paid API calls.** Studies use local data only.
10. **Resources.** The machine is shared: at most 2 threads per model (`n_jobs=2`), read only the needed columns,
    long runs in the background with a `timeout` and a log file.

## Reports

Russian, A4, built with `scripts/report_kit.py` (table text at least 8 pt, at most 11 columns per table;
split wider tables). Each report has: goal, problem, hypotheses, data, methods with formulas, parameters,
experiments, results with tables and charts, negative results, limitations, conclusions, recommendations
(A / B / C) and the commands to reproduce it. Rendered pages are checked (`Report.check`) for cut text.
