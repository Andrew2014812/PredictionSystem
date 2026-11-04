"""Invariants of the one prediction history, checked on every run and by tests.

Each check returns a list of problems (empty = fine); nothing is repaired or deleted here, so a problem is
reported instead of silently "fixed" by dropping rows.

* one Main and one Risk per match, no prediction stored twice;
* every settled prediction belongs to a known match and carries that match's result;
* a price is a real bookmaker quote: a bookmaker is named, it is not a market average or a fair price,
  and it was quoted before kick-off;
* profit and ROI use bets only (real price, won / lost);
* the Leagues page adds up to the History page (same predictions, same bets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .performance import group_summary, summarize, unified_history

SETTLED = ("WON", "LOST", "PUSH")


def check_history(picks: pd.DataFrame, matches: pd.DataFrame | None = None) -> dict[str, list]:
    history = unified_history(picks)
    out: dict[str, list] = {}
    if history.empty:
        return out
    roles = history.loc[history["role"].isin(["main", "risk"])]
    per = roles.groupby(["match_id", "role"]).size().unstack(fill_value=0).reindex(columns=["main", "risk"],
                                                                                 fill_value=0)
    for role in ("main", "risk"):
        if role in per:
            bad = per.index[per[role] > 1].tolist()
            if bad:
                out[f"more than one {role} per match"] = bad[:20]
    if {"main", "risk"} <= set(per.columns):
        lonely = per.index[(per["main"] == 1) & (per["risk"] == 0)].tolist()
        if lonely:
            out["main without a risk"] = lonely[:20]
    if "pick_id" in history:
        dup = history.loc[history["pick_id"].duplicated(), "pick_id"].tolist()
        if dup:
            out["prediction stored twice"] = dup[:20]

    priced = history.loc[history["odds"].notna()]
    if "bookmaker" in priced:
        nameless = priced.loc[priced["bookmaker"].isna() | (priced["bookmaker"] == "Market average"), "pick_id"]
        if len(nameless):
            out["price without a real bookmaker"] = nameless.tolist()[:20]
    if "real_odds" in priced:
        fake = priced.loc[~priced["real_odds"].fillna(False).astype(bool), "pick_id"]
        if len(fake):
            out["price not marked as real"] = fake.tolist()[:20]
    profit_without_price = history.loc[history["odds"].isna() & history["profit"].notna(), "pick_id"]
    if len(profit_without_price):
        out["profit without a price"] = profit_without_price.tolist()[:20]

    if matches is not None and not matches.empty:
        known = matches.set_index("match_id")
        done = history.loc[history["status"].isin(SETTLED)]
        if "odds_priced_at" in priced and "kickoff" in known:
            kick = pd.to_datetime(priced["match_id"].map(known["kickoff"]), errors="coerce")
            late = priced.loc[pd.to_datetime(priced["odds_priced_at"], errors="coerce") > kick, "pick_id"]
            if len(late):
                out["price quoted after kick-off"] = late.tolist()[:20]
        unknown = sorted(set(done["match_id"]) - set(known.index))
        # a settled prediction of a moved match carries the id of its first date: accept a played match
        # of the same pairing (see store._rescheduled_result)
        if unknown:
            pairs = set(zip(known["league"], known["home_team"], known["away_team"]))
            first = done.drop_duplicates("match_id").set_index("match_id")
            unknown = [m for m in unknown
                       if (first.at[m, "league"], first.at[m, "home_team"], first.at[m, "away_team"]) not in pairs]
            if unknown:
                out["settled prediction of an unknown match"] = unknown[:20]
        if {"home_goals", "away_goals"} <= set(known.columns):
            res = done.drop_duplicates("match_id").set_index("match_id")
            res = res.loc[res.index.isin(known.index)]
            k = known.loc[res.index]
            wrong = res.index[(k["home_goals"].notna()) & ((res["home_goals"] != k["home_goals"])
                                                         | (res["away_goals"] != k["away_goals"]))].tolist()
            if wrong:
                out["result differs from the match"] = wrong[:20]

    for role in ("main", "risk"):
        part = history.loc[history["role"] == role]
        s = summarize(part)
        bets = part.loc[part["status"].isin(["WON", "LOST"]) & part["odds"].notna() & part["profit"].notna()]
        if s["bets"] != len(bets) or (len(bets) and not np.isclose(s["profit"], bets["profit"].sum())):
            out[f"{role} ROI not computed from the bets"] = [s["bets"], len(bets)]
        by_league = group_summary(part, "league")
        if not by_league.empty and int(by_league["bets"].sum()) != s["bets"]:
            out[f"{role}: leagues do not add up to the history"] = [int(by_league["bets"].sum()), s["bets"]]
    return out
