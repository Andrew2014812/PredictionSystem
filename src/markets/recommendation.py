"""Main prediction and risk prediction: formal selection rules.

MAIN PREDICTION — the system's best prediction for the match; it exists for
every match that has a model prediction.

    Candidates: every selection except exact scores whose probability lies in
    [MAIN_MIN_PROBABILITY, MAIN_MAX_PROBABILITY] (the upper bound removes
    near-certain, uninformative selections such as "over 0.5").

    Every candidate gets the score

        reliability(market) x (p + MAIN_EV_WEIGHT x EV) x odds_preference(odds)

    odds_preference is 1 inside MAIN_PREFERRED_ODDS (1.40–2.50) and decays
    smoothly outside it (a Gaussian in log-odds with width MAIN_ODDS_SOFTNESS)
    — a soft preference, not a hard range.

    1. Value pick — the best-scoring candidate WITH a real price and
       EV >= MAIN_MIN_EV.
    2. Otherwise a model pick — the best-scoring candidate on probability
       alone (EV = 0; the "odds" of the preference are the fair odds 1 / p).
    3. If no selection is in the probability band, the most probable 1X2
       outcome is used.

    The weights were chosen on the VALIDATION season only (see
    ``scripts/decision_study.py``); the test season is reported afterwards.

RISK PREDICTION — optional, real odds only: RISK_MIN_ODDS <= odds <=
RISK_MAX_ODDS, p >= RISK_MIN_PROBABILITY, EV >= RISK_MIN_EV and a market
reliability of at least RISK_MIN_RELIABILITY; highest EV x reliability.
Exact scores are excluded. If nothing qualifies there is no risk prediction.

Market reliability weights downgrade markets the models predict less
reliably (corners, handicap, exact score).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config
from .odds import PricedSelection


@dataclass
class Recommendation:
    role: str                 # "main" | "risk"
    kind: str                 # "value" (chosen on EV) | "model" (chosen on probability)
    pick: PricedSelection
    score: float

    def to_dict(self) -> dict:
        return {"role": self.role, "kind": self.kind, "score": self.score, **self.pick.to_dict()}


def reliability(sel: PricedSelection) -> float:
    return config.MARKET_RELIABILITY.get(sel.group, 0.8)


def odds_preference(odds: float, preferred=None, softness: float | None = None) -> float:
    """1 inside the preferred odds range, decaying smoothly (in log-odds) outside it."""
    low, high = preferred or config.MAIN_PREFERRED_ODDS
    softness = config.MAIN_ODDS_SOFTNESS if softness is None else softness
    if not odds or odds <= 1.0:
        return 0.0
    x = math.log(odds)
    d = math.log(low) - x if odds < low else (x - math.log(high) if odds > high else 0.0)
    return math.exp(-0.5 * (d / softness) ** 2) if softness > 0 else float(d == 0.0)


def main_score(sel: PricedSelection, ev_weight: float | None = None, softness: float | None = None) -> float:
    ev_weight = config.MAIN_EV_WEIGHT if ev_weight is None else ev_weight
    if sel.has_odds:
        return reliability(sel) * (sel.probability + ev_weight * sel.ev) * odds_preference(sel.odds, None, softness)
    return reliability(sel) * sel.probability * odds_preference(1 / sel.probability, None, softness)


def main_prediction(selections: list[PricedSelection], min_ev: float | None = None,
                    ev_weight: float | None = None, softness: float | None = None,
                    min_probability: float | None = None) -> Recommendation | None:
    if not selections:
        return None
    min_ev = config.MAIN_MIN_EV if min_ev is None else min_ev
    min_p = config.MAIN_MIN_PROBABILITY if min_probability is None else min_probability
    candidates = [s for s in selections if s.group != "EXACT_SCORE"
                  and min_p <= s.probability <= config.MAIN_MAX_PROBABILITY]
    value = [s for s in candidates if s.has_odds and s.ev >= min_ev]
    if value:
        best = max(value, key=lambda s: (main_score(s, ev_weight, softness), s.probability))
        return Recommendation("main", "value", best, main_score(best, ev_weight, softness))
    if candidates:
        best = max(candidates, key=lambda s: (main_score(s, 0.0, softness), s.has_odds))
        return Recommendation("main", "model", best, main_score(best, 0.0, softness))
    one_x_two = [s for s in selections if s.market == "1X2"]
    if not one_x_two:
        return None
    best = max(one_x_two, key=lambda s: s.probability)
    return Recommendation("main", "model", best, best.probability)


def risk_prediction(selections: list[PricedSelection], min_ev: float | None = None) -> Recommendation | None:
    min_ev = config.RISK_MIN_EV if min_ev is None else min_ev
    candidates = [s for s in selections if s.has_odds and s.group != "EXACT_SCORE"
                  and config.RISK_MIN_ODDS <= s.odds <= config.RISK_MAX_ODDS
                  and s.probability >= config.RISK_MIN_PROBABILITY
                  and s.ev >= min_ev
                  and reliability(s) >= config.RISK_MIN_RELIABILITY]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: (s.ev * reliability(s), s.probability))
    return Recommendation("risk", "value", best, best.ev * reliability(best))
