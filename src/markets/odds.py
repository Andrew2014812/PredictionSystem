"""Odds layer: kept strictly separate from the ML prediction.

For every selection the system distinguishes:

* model probability ``p`` — output of the models (markets.py);
* real odds — a price published by a bookmaker and collected by a data
  provider (Football-Data, API-Football, The Odds API). Nothing is derived
  or simulated: a selection without a real price simply has no odds;
* expected value per unit stake, only when real odds exist:
  ``EV = p * odds - 1``; for an Asian whole-line handicap, where the stake is
  refunded with probability ``p_push``, ``EV = p_win * odds - (1 - p_push)``
  (win pays odds - 1, refund pays 0, loss costs 1). EV is a theoretical edge
  at the quoted price, not a profit.
"""
from __future__ import annotations

from dataclasses import dataclass

from .markets import Selection, market_group


@dataclass
class PricedSelection:
    market: str
    selection: str
    probability: float
    odds: float | None = None
    bookmaker: str | None = None
    provider: str | None = None
    push_probability: float = 0.0

    @property
    def group(self) -> str:
        return market_group(self.market)

    @property
    def has_odds(self) -> bool:
        return self.odds is not None and self.odds > 1.0

    @property
    def fair_odds(self) -> float | None:
        """1 / p — used in technical analysis only, never shown as a price."""
        return 1 / self.probability if self.probability > 0 else None

    @property
    def ev(self) -> float | None:
        return self.probability * self.odds - (1 - self.push_probability) if self.has_odds else None

    @property
    def lose_probability(self) -> float:
        return max(0.0, 1.0 - self.probability - self.push_probability)

    def to_dict(self) -> dict:
        return {"market": self.market, "selection": self.selection, "group": self.group,
                "probability": self.probability, "odds": self.odds if self.has_odds else None,
                "bookmaker": self.bookmaker, "provider": self.provider, "ev": self.ev,
                "push_probability": self.push_probability}


def price(selections: list[Selection], match_odds: dict[tuple[str, str], dict] | None) -> list[PricedSelection]:
    """Attach the real price of every selection that has one."""
    match_odds = match_odds or {}
    out = []
    for s in selections:
        quote = match_odds.get((s.market, s.selection))
        out.append(PricedSelection(s.market, s.selection, s.probability,
                                   quote["odds"] if quote else None,
                                   quote.get("bookmaker") if quote else None,
                                   quote.get("provider") if quote else None,
                                   s.push_probability))
    return out
