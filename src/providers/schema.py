"""Unified schema of real bookmaker odds collected from any provider.

One row = one price of one selection at one bookmaker:

    match_id, provider, provider_fixture_id, bookmaker,
    market, line, selection, odds, collected_at

``market`` / ``selection`` use the system's own keys (see markets/markets.py),
e.g. ("TOTAL_2.5", "OVER"), ("HANDICAP_-0.5", "H"), ("EXACT_SCORE", "2-1").
Only prices published by a bookmaker are stored — nothing is derived.
"""
from __future__ import annotations

import pandas as pd

ODDS_COLUMNS = ["match_id", "provider", "provider_fixture_id", "bookmaker", "market", "line",
                "selection", "odds", "collected_at"]

# When several providers price the same selection, the first one wins.
PROVIDER_PRIORITY = ("api_football", "the_odds_api", "football_data")


def empty_odds() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def odds_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return empty_odds()
    df = pd.DataFrame(records).reindex(columns=ODDS_COLUMNS)
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.loc[df["odds"] > 1.0]
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    return df.reset_index(drop=True)


def best_prices(odds: pd.DataFrame) -> pd.DataFrame:
    """One price per (match, market, selection): provider priority, then latest."""
    if odds.empty:
        return odds
    rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
    df = odds.assign(_rank=odds["provider"].map(rank).fillna(len(rank)))
    df = df.sort_values(["match_id", "market", "selection", "_rank", "collected_at"],
                        ascending=[True, True, True, True, False])
    return df.drop_duplicates(["match_id", "market", "selection"]).drop(columns="_rank")


def odds_lookup(odds: pd.DataFrame) -> dict[str, dict[tuple[str, str], dict]]:
    """match_id -> {(market, selection): {"odds", "bookmaker", "provider"}}."""
    out: dict[str, dict] = {}
    for r in best_prices(odds).itertuples():
        out.setdefault(r.match_id, {})[(r.market, r.selection)] = {
            "odds": float(r.odds), "bookmaker": r.bookmaker, "provider": r.provider}
    return out
