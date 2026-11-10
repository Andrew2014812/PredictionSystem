"""The Odds API: what is stored, which predictions still depend on it and whether a fallback would add anything.

Inputs: ``local_data/odds/odds.parquet`` (provider = the_odds_api), ``local_data/api/the_odds_api/ledger.json`` and
``quota.log`` (committed), the unified prediction history and the OddsPapi quotes of the same selections.
"""
from __future__ import annotations

import json

import pandas as pd

from . import common as C


def stored(o: pd.DataFrame) -> dict:
    t = o.loc[o["provider"] == "the_odds_api"]
    feats = pd.read_parquet(C.DATA / "processed" / "features.parquet", columns=["match_id", "league", "season"])
    lg = t["match_id"].map(feats.drop_duplicates("match_id").set_index("match_id")["league"])
    return {"rows": int(len(t)), "matches": int(t["match_id"].nunique()),
            "first_collected": str(t["collected_at"].min()), "last_collected": str(t["collected_at"].max()),
            "by_group": t.groupby("group")["match_id"].nunique().to_dict(),
            "by_bookmaker": t["bookmaker"].value_counts().to_dict(),
            "by_league": lg.fillna("?").groupby(lg.fillna("?")).size().to_dict(),
            "matches_by_league": t.assign(league=lg.fillna("?")).groupby("league")["match_id"].nunique().to_dict()}


def credits() -> dict:
    ledger = json.loads((C.API / "the_odds_api" / "ledger.json").read_text())
    log = pd.read_json(C.API / "the_odds_api" / "quota.log", lines=True)
    log["day"] = log["at"].astype(str).str[:10]
    log["kind"] = log["url"].str.extract(r"/(events/[^/]+/odds|odds|events)$")[0].str.replace(r"events/[^/]+/odds",
                                                                                            "event odds", regex=True)
    by_day = log.groupby("day").agg(requests=("url", "size"), credits=("cost", "sum"),
                                    remaining_after=("remaining", "last")).reset_index()
    C.save_csv("toa_requests_per_day.csv", by_day)
    return {"ledger_month": ledger.get("month"), "ledger_month_credits": ledger.get("month_credits"),
            "ledger_remaining": ledger.get("remaining"), "ledger_checked_at": ledger.get("checked_at"),
            "log_requests": int(len(log)), "log_credits": float(log["cost"].sum()),
            "log_first": str(log["at"].min()), "log_last": str(log["at"].max()),
            "by_kind": log.groupby("kind").agg(requests=("url", "size"), credits=("cost", "sum")).reset_index().to_dict("records"),
            "days": by_day.to_dict("records")}


def dependence(o: pd.DataFrame) -> dict:
    """Predictions (unified history, 2026/27) priced by The Odds API, and whether OddsPapi quoted the same
    selection before the prediction was generated (A2-like) or before kick-off (B-like)."""
    from src.analytics.performance import summarize, unified_history
    from src.markets.markets import market_group
    p = pd.read_parquet(C.PICKS_FILE)
    h = unified_history(p)
    h = h.loc[(h["season"] == C.SEASON)]
    t = h.loc[h["provider"] == "the_odds_api"].copy()
    op = o.loc[o["provider"] == "oddspapi", ["match_id", "market", "selection", "collected_at", "kickoff"]]
    op = op.loc[op["collected_at"] < op["kickoff"]]
    gen = C.uk_to_utc(op["collected_at"]).to_numpy()
    op = op.assign(quote_utc=gen)
    first = op.groupby(["match_id", "market", "selection"])["quote_utc"].min().rename("first_odp_quote_utc")
    t = t.join(first, on=["match_id", "market", "selection"])
    t["odp_any"] = t["first_odp_quote_utc"].notna()
    t["odp_before_generation"] = t["odp_any"] & (t["first_odp_quote_utc"] <= pd.to_datetime(t["generated_at"]))
    t["group"] = t["market"].map(market_group)
    roles = {}
    for role, g in t.groupby("role"):
        s = summarize(g)
        roles[role] = {"predictions": int(len(g)), "settled": s["settled"], "bets": s["bets"], "profit": s["profit"],
                       "roi": s["roi"], "odp_same_selection": int(g["odp_any"].sum()),
                       "odp_before_generation": int(g["odp_before_generation"].sum())}
    C.save_csv("toa_dependent_picks.csv", t[["pick_id", "match_id", "league", "role", "market", "selection", "odds",
                                              "bookmaker", "status", "profit", "generated_at", "odp_any",
                                              "odp_before_generation"]])
    return {"by_role": roles, "matches": int(t["match_id"].nunique()),
            "dates": [str(t["date"].min()), str(t["date"].max())],
            "by_league_main_risk": t.loc[t["role"].isin(["main", "risk"])].groupby("league").size().to_dict(),
            "by_group_main_risk": t.loc[t["role"].isin(["main", "risk"])].groupby("group").size().to_dict(),
            "share_of_main_risk_bets": None}


def run() -> dict:
    from .oddspapi import load_odds
    o = load_odds()
    out = {"stored": stored(o), "credits": credits(), "dependence": dependence(o),
           "plan_credits_per_month": 500}                      # free plan (README, "Data sources")
    # share of all priced Main / Risk of 2026/27 that rest on The Odds API
    from src.analytics.performance import unified_history
    h = unified_history(pd.read_parquet(C.PICKS_FILE))
    h = h.loc[(h["season"] == C.SEASON) & h["role"].isin(["main", "risk"]) & h["odds"].notna()]
    out["dependence"]["share_of_main_risk_bets"] = float((h["provider"] == "the_odds_api").mean())
    out["dependence"]["main_risk_priced_total"] = int(len(h))
    C.save_json("the_odds_api.json", out)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str)[:6000])
