# FootPredict v6.3 — research experiments

Main goal: find out whether better probabilities, models, data and a better Main rule give a more accurate and
more profitable Main prediction. Every study replays the whole season 2026/27 on its own data; the website's
history is never rewritten.

## Shared replay (`replay/`)

`python -m experiments.v6_3.replay.build` builds the inputs every study reads (`replay/cache/`, not committed):

| File | Content |
|---|---|
| `matches.parquet` | every 2026/27 match of the feature table with its decision time T |
| `odds_long.parquet` | every stored real price of these matches before kick-off (OddsPapi, The Odds API, Football-Data) |
| `prices_{decision,early,closing}.parquet` | one price per selection at T / 48 h before kick-off / last before kick-off |
| `prices_books_decision.parquet` | the latest price per bookmaker at T (consensus studies) |
| `outputs_production.parquet` | the production models' outputs (p 1X2, λ home, λ away, μ corners) for every match |
| `candidates_{t09,close}.parquet` | every selection of every match (production market mathematics) with its price, EV, result and profit |
| `baseline_a.parquet` | Baseline A — the Main of the website's unified history (live + backfill) |
| `baseline_b_{t09,close}.parquet` | Baseline B — production models + production Main rule replayed |

Code: `replay/data.py` (matches, decision times, prices), `replay/engine.py` (production engine with any model
outputs; refitting a task; monthly walk-forward), `replay/evaluate.py` (settlement, Main by rule / by score,
summaries, paired comparison, per-match table).

### Two decision moments

The OddsPapi archive stores mostly the last quote before kick-off (median 1 quote per selection and bookmaker;
for about half of the selections the first stored quote is later than the morning run). So:

* **close** — the decision is made just before kick-off with the last quote before kick-off. Prices are complete
  (99.96 % of Main priced), the features are the same pre-match features. **Primary universe** of the selection
  studies: it covers the whole season and matches the website's own prices.
* **t09** — the decision is made at the daily run, T = min(match day 09:00 UK, kick-off − 30 min), with only the
  prices known at T (Football-Data 1X2 / total 2.5 / Asian line for every match, OddsPapi where an earlier quote
  is stored). About a quarter of the chosen Main have no known price at T; they are settled but are not bets.
  Used as a robustness check: a rule must not win only at closing prices.

A price published after the decision moment is never used for that decision.

### Baselines

* **Baseline A** — the website's history: one Main per match (live when recorded before kick-off, otherwise
  backfill), stored price and result, exactly as Prediction History counts it.
* **Baseline B** — production replayed: production models `prod_20261012` (trained on 2021/22–2025/26, so on
  matches before every 2026/27 match) and the production Main rule, on every 2026/27 match, at the decision moment
  of the universe. **All paired comparisons use Baseline B of the same universe.**

## Methodology

1. **No future information.** A feature of a match uses only earlier matches (the production feature builder,
   verified in v6.2). A refitted model is trained only on matches played before its cut-off (monthly walk-forward:
   matches of month M are predicted by models fitted on everything played before the 1st of M). Any parameter
   fitted inside 2026/27 (calibration maps, market reliabilities, meta-models) is fitted only on matches settled
   before the predicted match (expanding window by month).
2. **Selection of variants.** Where a variant has parameters, they are chosen on earlier seasons (2023/24, 2024/25
   walk-forward tables of v6.1 / v6.2) or on months of 2026/27 before the evaluated month — never on the whole
   season that is then reported.
3. **2026/27 is not a holdout.** It was examined by v6.1 / v6.2. A result on it is a chronological check on an
   already studied season; a rule that looks better is frozen (preregistration) and judged on future matches.
4. **Same matches.** Methods are compared on the same matches; matches a method cannot handle are counted and
   shown, never dropped silently.
5. **Uncertainty.** Bootstrap 95 % intervals by match, paired bootstrap for differences, Holm correction across
   the variants of a study; the number of variants tried is reported.
6. **ROI** as production computes it: unit stake, WON / LOST with a real price are bets, PUSH / VOID return the
   stake, a prediction without a real price is not a bet. EV = p · odds − (1 − P(push)); the formula is not changed.
7. **Decisions.** A — adopt (convincing paired improvement on the chronological check, stable across months and
   leagues, enough bets, tests pass); B — promising, frozen for a forward check; C — rejected.
8. **Resources.** The machine is shared: at most 2 threads per model, one heavy job at a time per study, long runs
   in the background with a timeout and a log.
9. **No paid API calls.** Studies use local data only, unless a study states otherwise and the owner agreed.

## Layout of a study folder

`run.py` (experiments → `results/`), `build_pdf.py` (report → `reports/v6_3/<Name>.pdf`), `results/*.json|csv`
(raw results, `run_info.json` with git SHA / data version / parameters), `figures/*.png`, `findings.md` (short
summary and A / B / C decision), `cache/` and `logs/` (not committed). Every study publishes a per-match table of
its best candidate and of Baseline B (`results/per_match_*.csv`, fields of Prompt 9 §5).
