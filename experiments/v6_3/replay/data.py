"""Matches, results and decision-time prices of the season 2026/27 replay.

Decision time. Every replayed prediction is made at one moment per match, the same for every method:

    T = min(match day 09:00 UK, kick-off − 30 min)

09:00 UK is about when the daily run finishes (it starts 07:15–09:20 UTC in practice, see v6.2 Data_Sources_Report);
for a match that kicks off before 09:30 the decision moves to 30 minutes before kick-off.

Prices. Only real bookmaker prices that existed at T (``collected_at`` <= T, UK time):
* OddsPapi quotes (bet365, Pinnacle, William Hill) — timestamped by the bookmaker's quote time;
* The Odds API prices already stored (no new credits);
* Football-Data prices of the season's fixture / result files (Bet365 pre-match price, published before the match;
  the file has no time, production treats it as pre-match and so does the replay — lowest priority).
One price per selection, as production chooses it (``src/providers/schema.best_prices``: provider priority, then
the latest quote). Two diagnostic price sets are built the same way: CLOSING (last quote before kick-off) and EARLY
(last quote at least 48 h before kick-off). They are never used as if available at T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src.providers.football_data import odds_from_matches
from src.providers.schema import PROVIDER_PRIORITY

DECISION_HOUR = 9
DECISION_LEAD = pd.Timedelta(minutes=30)
EARLY_LEAD = pd.Timedelta(hours=48)
MATCH_COLS = ["match_id", "league", "country", "season", "date", "time", "kickoff", "home_team", "away_team", "played",
              "home_goals", "away_goals", "total_corners", "result", "odds_home", "odds_draw", "odds_away",
              "odds_over25", "odds_under25", "odds_ah_line", "odds_ah_home", "odds_ah_away"]


def load_features(columns: list[str] | None = None) -> pd.DataFrame:
    f = pd.read_parquet(C.DATA_FILES["features"], columns=columns)
    if "date" in f:
        f["date"] = pd.to_datetime(f["date"])
    if "kickoff" in f:
        f["kickoff"] = pd.to_datetime(f["kickoff"], errors="coerce")
    return f


def season_matches(features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every match of 2026/27 in the feature table (played or not) with its decision time."""
    f = features if features is not None else load_features()
    m = f.loc[f["season"] == C.CURRENT].copy()
    m["decision_time"] = decision_time(m)
    return m.sort_values(["date", "kickoff", "match_id"]).reset_index(drop=True)


def decision_time(m: pd.DataFrame) -> pd.Series:
    day = pd.to_datetime(m["date"]).dt.normalize() + pd.Timedelta(hours=DECISION_HOUR)
    ko = pd.to_datetime(m["kickoff"], errors="coerce")
    t = day.where(ko.isna() | (ko - DECISION_LEAD >= day), ko - DECISION_LEAD)
    return t


def results(m: pd.DataFrame) -> pd.DataFrame:
    """Played matches with the score and corners used for settlement (the same table production settles with)."""
    r = m.loc[m["played"].astype(bool), ["match_id", "home_goals", "away_goals", "total_corners"]]
    return r.set_index("match_id")


def odds_long(m: pd.DataFrame) -> pd.DataFrame:
    """All stored real prices of the season's matches, strictly before kick-off (UK time)."""
    api = pd.read_parquet(C.DATA_FILES["odds"])
    api = api.loc[api["match_id"].isin(set(m["match_id"]))].copy()
    fd = odds_from_matches(m)
    fd["collected_at"] = pd.to_datetime(fd["collected_at"])
    allp = pd.concat([api, fd], ignore_index=True)
    allp["collected_at"] = pd.to_datetime(allp["collected_at"])
    ko = pd.to_datetime(allp["match_id"].map(m.set_index("match_id")["kickoff"]))
    allp = allp.loc[ko.isna() | (allp["collected_at"] < ko) | (allp["provider"] == "football_data")]
    allp["bookmaker"] = allp["bookmaker"].astype(str)
    return allp.reset_index(drop=True)


def _best(prices: pd.DataFrame) -> pd.DataFrame:
    rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
    df = prices.assign(_rank=prices["provider"].map(rank).fillna(len(rank)))
    df = df.sort_values(["match_id", "market", "selection", "_rank", "collected_at"],
                        ascending=[True, True, True, True, False])
    return df.drop_duplicates(["match_id", "market", "selection"]).drop(columns="_rank")


def prices_at(odds: pd.DataFrame, when: pd.Series, by_book: bool = False) -> pd.DataFrame:
    """Best price per (match, market, selection) among quotes collected at or before ``when[match_id]``.
    ``by_book``: the latest quote per bookmaker instead (for consensus studies)."""
    t = odds["match_id"].map(when)
    ok = odds.loc[t.notna() & (odds["collected_at"] <= t)]
    if by_book:
        ok = ok.sort_values("collected_at").drop_duplicates(["match_id", "market", "selection", "bookmaker"],
                                                            keep="last")
        return ok.reset_index(drop=True)
    return _best(ok).reset_index(drop=True)


def lookup(prices: pd.DataFrame) -> dict:
    """match_id -> {(market, selection): quote} in the form ``predict_matches`` expects."""
    out: dict[str, dict] = {}
    for r in prices.itertuples(index=False):
        out.setdefault(r.match_id, {})[(r.market, r.selection)] = {
            "odds": float(r.odds), "bookmaker": r.bookmaker, "provider": r.provider, "priced_at": r.collected_at}
    return out


def price_sets(m: pd.DataFrame, odds: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """The three price sets: decision (at T), early (T_early = kick-off − 48 h) and closing (last before kick-off)."""
    odds = odds if odds is not None else odds_long(m)
    by = m.set_index("match_id")
    ko = pd.to_datetime(by["kickoff"]).fillna(pd.to_datetime(by["date"]) + pd.Timedelta(hours=15))
    return {
        "decision": prices_at(odds, by["decision_time"]),
        "early": prices_at(odds, ko - EARLY_LEAD),
        "closing": prices_at(odds, ko - pd.Timedelta(seconds=1)),
    }


def coverage(m: pd.DataFrame, decision_prices: pd.DataFrame) -> dict:
    played = m["played"].astype(bool)
    priced = set(decision_prices.loc[decision_prices["market"] == "1X2", "match_id"])
    return {"season_matches": int(len(m)), "played": int(played.sum()), "not_played": int((~played).sum()),
            "played_with_1x2_price_at_T": int(m.loc[played, "match_id"].isin(priced).sum()),
            "played_with_corners": int(m.loc[played, "total_corners"].notna().sum()),
            "first_date": str(m["date"].min().date()), "last_played": str(m.loc[played, "date"].max().date())}


def as_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s


__all__ = ["season_matches", "results", "odds_long", "prices_at", "price_sets", "lookup", "coverage", "np"]
