"""Data providers: Football-Data (history + Bet365 prices), API-Football and
The Odds API (fresh fixtures, results and real bookmaker odds)."""
from __future__ import annotations

import pandas as pd

from .collect import load_api_odds
from .football_data import odds_from_matches
from .schema import empty_odds, odds_lookup


def all_real_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """Every real price known for ``matches`` (Football-Data + API providers)."""
    frames = [odds_from_matches(matches)]
    api = load_api_odds()
    if not api.empty:
        api = api.loc[api["match_id"].isin(matches["match_id"])]
        frames.append(pre_match_only(api, matches))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_odds()


def pre_match_only(odds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Drop prices collected at or after kick-off: they are not pre-match prices.

    ``collected_at`` and ``kickoff`` are both naive UK local time. Football-Data
    prices are pre-match by definition (published before the match)."""
    if odds.empty or "kickoff" not in matches:
        return odds
    kickoff = matches.set_index("match_id")["kickoff"]
    ko = pd.to_datetime(odds["match_id"].map(kickoff))
    return odds.loc[ko.isna() | (pd.to_datetime(odds["collected_at"]) < ko)]


def real_odds_lookup(matches: pd.DataFrame) -> dict:
    odds = all_real_odds(matches)
    return odds_lookup(odds) if not odds.empty else {}
