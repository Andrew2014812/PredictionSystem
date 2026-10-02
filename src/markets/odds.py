"""Odds layer: kept strictly separate from the ML prediction.

For every selection four numbers are distinguished:

* model probability  ``p``           - output of the models (markets.py)
* fair odds          ``1 / p``       - the price at which a bet on p has zero EV
* odds + odds_source - the price used for EV and profit, one of
    - ``market``    real bookmaker average from Football-Data (1X2, O/U 2.5)
    - ``derived``   computed from the real 1X2 and O/U 2.5 prices of the same
                    match: a Poisson score model is fitted to the bookmaker
                    probabilities, the other goal markets are read from it and
                    the bookmaker's own margin is applied
    - ``simulated`` no market information at all (corners, or a match without
                    odds): a "naive bookmaker" prices the event at its recent
                    frequency in the league plus ``config.SIMULATED_MARGIN``
* expected value     ``EV = p * odds - 1`` per unit stake

Derived and simulated prices are never presented as bookmaker odds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .. import config
from .distributions import outcome_probabilities, score_matrix
from .markets import Selection, exact_score_probability, goal_markets, market_group

SOURCE_LABELS = {
    "market": "Bookmaker odds",
    "derived": "Derived from bookmaker odds",
    "simulated": "Simulated odds",
}


@dataclass
class PricedSelection:
    market: str
    selection: str
    probability: float
    odds: float | None
    odds_source: str | None

    @property
    def group(self) -> str:
        return market_group(self.market)

    @property
    def fair_odds(self) -> float | None:
        return 1 / self.probability if self.probability > 0 else None

    @property
    def implied_probability(self) -> float | None:
        return 1 / self.odds if self.odds else None

    @property
    def ev(self) -> float | None:
        return self.probability * self.odds - 1 if self.odds else None

    def to_dict(self) -> dict:
        return {"market": self.market, "selection": self.selection, "group": self.group,
                "probability": self.probability, "odds": self.odds, "odds_source": self.odds_source,
                "fair_odds": self.fair_odds, "implied_probability": self.implied_probability,
                "ev": self.ev}


def _valid(x) -> bool:
    return x is not None and x == x and x > 1.0


def real_odds(match: dict) -> dict[tuple[str, str], float]:
    """Bookmaker prices available for the match in the source data."""
    out = {}
    for sel, col in (("H", "odds_home"), ("D", "odds_draw"), ("A", "odds_away")):
        if _valid(match.get(col)):
            out[("1X2", sel)] = float(match[col])
    if _valid(match.get("odds_over25")):
        out[("TOTAL_2.5", "OVER")] = float(match["odds_over25"])
    if _valid(match.get("odds_under25")):
        out[("TOTAL_2.5", "UNDER")] = float(match["odds_under25"])
    return out


def market_margin(match: dict) -> float:
    """Bookmaker overround of the 1X2 market (fallback: simulated margin)."""
    odds = [match.get(c) for c in ("odds_home", "odds_draw", "odds_away")]
    if all(_valid(o) for o in odds):
        return max(sum(1 / o for o in odds) - 1, 0.0)
    return config.SIMULATED_MARGIN


def fit_market_score_matrix(match: dict) -> np.ndarray | None:
    """Poisson score matrix whose 1X2 (and O/U 2.5) matches the bookmaker."""
    odds = [match.get(c) for c in ("odds_home", "odds_draw", "odds_away")]
    if not all(_valid(o) for o in odds):
        return None
    inv = np.array([1 / o for o in odds])
    target = inv / inv.sum()
    over_target = None
    if _valid(match.get("odds_over25")) and _valid(match.get("odds_under25")):
        o, u = 1 / match["odds_over25"], 1 / match["odds_under25"]
        over_target = o / (o + u)

    def loss(log_lams):
        m = score_matrix(*np.exp(log_lams), max_goals=8)
        err = np.sum((outcome_probabilities(m) - target) ** 2)
        if over_target is not None:
            i, j = np.indices(m.shape)
            err += (m[i + j > 2.5].sum() - over_target) ** 2
        return err

    res = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    return score_matrix(*np.exp(res.x))


def price(selections: list[Selection], match: dict,
          market_matrix: np.ndarray | None,
          base_goal_probs: dict[tuple[str, str], float],
          base_matrix: np.ndarray | None,
          base_corner_probs: dict[tuple[str, str], float]) -> list[PricedSelection]:
    """Attach odds to every selection, preferring market > derived > simulated."""
    real = real_odds(match)
    margin = market_margin(match)
    derived = {}
    if market_matrix is not None:
        derived = {(s.market, s.selection): s.probability for s in goal_markets(market_matrix, top_scores=0)}

    def with_margin(p: float, m: float) -> float | None:
        return round(1 / (p * (1 + m)), 2) if p > 1e-4 else None

    out = []
    for s in selections:
        key = (s.market, s.selection)
        odds, source = None, None
        if key in real:
            odds, source = real[key], "market"
        elif s.group == "EXACT_SCORE" and market_matrix is not None:
            odds, source = with_margin(exact_score_probability(market_matrix, s.selection), margin), "derived"
        elif key in derived:
            odds, source = with_margin(derived[key], margin), "derived"
        elif s.group == "CORNERS" and key in base_corner_probs:
            odds, source = with_margin(base_corner_probs[key], config.SIMULATED_MARGIN), "simulated"
        elif s.group == "EXACT_SCORE" and base_matrix is not None:
            odds = with_margin(exact_score_probability(base_matrix, s.selection), config.SIMULATED_MARGIN)
            source = "simulated"
        elif key in base_goal_probs:
            odds, source = with_margin(base_goal_probs[key], config.SIMULATED_MARGIN), "simulated"
        out.append(PricedSelection(s.market, s.selection, s.probability, odds,
                                   source if odds else None))
    return out
