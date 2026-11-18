"""Build the shared replay of 2026/27: matches, prices, Baseline A and Baseline B.

    python -m experiments.v6_3.replay.build

Writes ``experiments/v6_3/replay/cache/*.parquet`` (inputs every study reads) and ``results/`` (baseline tables,
coverage, self-checks, run info).

* Baseline A — the website's history: the Main of every 2026/27 match of the unified history (live when recorded,
  otherwise backfill), with the stored price and result, exactly as Prediction History counts it.
* Baseline B — production replayed: the production models (prod_20261012) and the production Main rule on every
  2026/27 match, with the prices that existed at the decision time T (``data.py``). All studies compare with B
  on the same matches; A is reported next to it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as D
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV
from src.analytics.performance import summarize, unified_history

CACHE, RESULTS = C.SHARED_CACHE, C.SHARED_RESULTS


# Two decision moments, both honest (features are the same pre-match features in both):
#   t09   — the daily run's moment T = min(match day 09:00 UK, kick-off − 30 min), with the prices known at T;
#   close — the last moment before kick-off, with the last quote before kick-off (what the OddsPapi archive mostly
#           stores: a median of 1 quote per selection, so most prices of the archive are unknown at T).
UNIVERSES = {"t09": "decision", "close": "closing"}


def archive_depth(odds: pd.DataFrame, m: pd.DataFrame) -> dict:
    o = odds.loc[odds["provider"] == "oddspapi"]
    q = o.groupby(["match_id", "market", "selection", "bookmaker"])["collected_at"].agg(["size", "min"])
    t = q.index.get_level_values(0).map(m.set_index("match_id")["decision_time"])
    return {"selections_x_books": int(len(q)), "median_quotes": float(q["size"].median()),
            "mean_quotes": float(q["size"].mean()), "share_first_quote_after_T": float((q["min"] > t).mean())}


def build_universe(universe: str, m, res, out, sets, prices_key: str, a: pd.DataFrame) -> dict:
    batch = EN.batch_from_outputs(m, out, D.lookup(sets[prices_key]), EN.PRODUCTION_VERSION)
    extra = {k: v for k, v in sets.items() if k != prices_key}
    cands = EV.candidates(batch.markets, m, res, extra)
    cands.to_parquet(CACHE / f"candidates_{universe}.parquet")
    main_engine = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
    b = cands.merge(main_engine, on=["match_id", "market", "selection"])
    b["method"] = f"baseline_b_{universe}"
    b.to_parquet(CACHE / f"baseline_b_{universe}.parquet")
    sample = cands.loc[cands["match_id"].isin(m["match_id"].iloc[::10])]
    again = EV.main_by_rule(sample)
    chk = again.merge(b[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_engine"))
    same_rule = float(((chk["market"] == chk["market_engine"]) & (chk["selection"] == chk["selection_engine"])).mean())
    played = b.loc[b["match_id"].isin(res.index)]
    EV.per_match_table(played, res, f"baseline_b_{universe}", EN.PRODUCTION_VERSION).to_csv(
        RESULTS / f"baseline_b_{universe}_main.csv", index=False)
    return {"summary": EV.summary(played), "roi_ci": EV.roi_with_ci(played), "vs_baseline_a": EV.compare(played, a),
            "rows": int(len(b)), "played": int(len(played)), "priced": int(played["odds"].notna().sum()),
            "status": played["status"].value_counts().to_dict(), "groups": played["group"].value_counts().to_dict(),
            "unpriced_groups": played.loc[played["odds"].isna(), "group"].value_counts().to_dict(),
            "candidates": int(len(cands)), "providers": sets[prices_key]["provider"].value_counts().to_dict(),
            "self_check_rule_equals_engine": same_rule, "self_check_matches": int(len(chk))}


def baseline_a(matches: pd.DataFrame) -> pd.DataFrame:
    picks = pd.read_parquet(C.DATA_FILES["picks"])
    u = unified_history(picks.loc[picks["season"] == C.CURRENT])
    main = u.loc[u["role"] == "main"].copy()
    main["bet"] = (main["odds"].notna() & main["status"].isin(["WON", "LOST"]) & main["profit"].notna()).astype(int)
    main["method"] = "baseline_a"
    return main


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"{name}.parquet")


def main() -> None:
    started = time.time()
    log = C.setup_logging("replay", CACHE.parent / "build.log")
    feats = D.load_features()
    m = D.season_matches(feats)
    res = D.results(m)
    log.info("season 2026/27: %d matches, %d played", len(m), len(res))
    odds = D.odds_long(m)
    sets = D.price_sets(m, odds)
    books = D.prices_at(odds, m.set_index("match_id")["decision_time"], by_book=True)
    m.to_parquet(CACHE / "matches.parquet")
    odds.to_parquet(CACHE / "odds_long.parquet")
    for k, v in sets.items():
        v.to_parquet(CACHE / f"prices_{k}.parquet")
    books.to_parquet(CACHE / "prices_books_decision.parquet")

    out = EN.production_outputs(m)
    out.to_parquet(CACHE / "outputs_production.parquet")
    a = baseline_a(m)
    a.to_parquet(CACHE / "baseline_a.parquet")
    a_tab = a.rename(columns={"source": "data_type"}).assign(method="baseline_a", note="")
    a_tab["priced_at"] = a_tab.get("odds_priced_at")
    a_tab.reindex(columns=EV.PER_MATCH_COLS).to_csv(RESULTS / "baseline_a_main.csv", index=False)

    report = {"coverage": D.coverage(m, sets["decision"]),
              "baseline_a": {"production_summary": summarize(a), "summary": EV.summary(a),
                             "roi_ci": EV.roi_with_ci(a)}}
    for universe, prices_key in UNIVERSES.items():
        report[universe] = build_universe(universe, m, res, out, sets, prices_key, a)
        log.info("baseline B (%s): %s", universe, report[universe]["summary"])

    # Baseline B vs the stored live predictions: same model, so the 1X2 probabilities should agree where the
    # features did not change after the prediction was recorded
    snaps = pd.read_parquet(C.DATA_FILES["snapshots"], columns=["match_id", "source", "p_home", "version"])
    live = snaps.loc[snaps["source"] == "live"].sort_values("version").drop_duplicates("match_id", keep="last")
    j = live.merge(out[["match_id", "p_home"]], on="match_id", suffixes=("_stored", "_replay"))
    report["self_check_outputs"] = {
        "replay_p_home_equals_stored_live(±0.005)": float((np.abs(j["p_home_stored"] - j["p_home_replay"]) < 0.005).mean()),
        "live_matches_compared": int(len(j))}
    report["archive_depth"] = archive_depth(odds, m)
    C.write_json(RESULTS / "baseline.json", report)
    C.write_json(RESULTS / "run_info.json", C.run_info("replay", started, {
        "decision_time": "min(match day 09:00 UK, kick-off - 30 min)", "early": "kick-off - 48 h",
        "closing": "last quote before kick-off", "production_version": EN.PRODUCTION_VERSION}))
    log.info("baseline A: %s", report["baseline_a"]["summary"])


if __name__ == "__main__":
    main()
