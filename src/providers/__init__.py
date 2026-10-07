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
        frames.append(api.loc[api["match_id"].isin(matches["match_id"])])
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_odds()


def real_odds_lookup(matches: pd.DataFrame) -> dict:
    odds = all_real_odds(matches)
    return odds_lookup(odds) if not odds.empty else {}
