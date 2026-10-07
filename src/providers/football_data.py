"""Real Bet365 prices published in the Football-Data CSV files.

Covered markets: 1X2, total goals 2.5 and the half-line of the Asian
handicap column (±0.5, ±1.5, ±2.5 are equivalent to a 2-way handicap without
refunds). Whole and quarter Asian lines are skipped: their settlement
(refunds, split stakes) differs from the system's handicap markets.
"""
from __future__ import annotations

import pandas as pd

from ..data.cleaning import ODDS_BOOKMAKER
from ..markets.markets import handicap_market, is_half_line
from .schema import odds_frame

PROVIDER = "football_data"


def odds_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Odds records from canonical match rows (results and fixtures)."""
    records = []
    cols = ["match_id", "date", "odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25",
            "odds_ah_line", "odds_ah_home", "odds_ah_away"]
    for r in matches.reindex(columns=cols).itertuples(index=False):
        base = {"match_id": r.match_id, "provider": PROVIDER, "provider_fixture_id": None,
                "bookmaker": ODDS_BOOKMAKER, "collected_at": r.date}

        def add(market, line, selection, price):
            if price == price and price is not None and price > 1.0:
                records.append({**base, "market": market, "line": line, "selection": selection,
                                "odds": float(price)})

        add("1X2", None, "H", r.odds_home)
        add("1X2", None, "D", r.odds_draw)
        add("1X2", None, "A", r.odds_away)
        add("TOTAL_2.5", 2.5, "OVER", r.odds_over25)
        add("TOTAL_2.5", 2.5, "UNDER", r.odds_under25)
        line = r.odds_ah_line
        if line == line and line is not None and is_half_line(float(line)):
            market = handicap_market(float(line))
            add(market, float(line), "H", r.odds_ah_home)
            add(market, float(line), "A", r.odds_ah_away)
    return odds_frame(records)
