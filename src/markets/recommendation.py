"""Main prediction and risk prediction: formal selection rules.

MAIN PREDICTION — the system's best prediction for the match; it exists for
every match that has a model prediction.

    Candidates: every selection except exact scores whose probability lies in
    [MAIN_MIN_PROBABILITY, MAIN_MAX_PROBABILITY] (the upper bound removes
    near-certain, uninformative selections such as "over 0.5").

    1. Value rule — among candidates WITH real odds in
       [MAIN_MIN_ODDS, MAIN_MAX_ODDS) and EV >= MAIN_MIN_EV, choose the
       highest EV x market reliability.
    2. Otherwise — choose the candidate with the highest
       probability x market reliability (odds and EV are shown only if a real
       price exists).
    3. If no selection is in the probability band, the most probable 1X2
       outcome is used.

RISK PREDICTION — optional, real odds only: RISK_MIN_ODDS <= odds <=
RISK_MAX_ODDS, p >= RISK_MIN_PROBABILITY, EV >= RISK_MIN_EV and a market
reliability of at least RISK_MIN_RELIABILITY; highest EV x reliability.
Exact scores are excluded. If nothing qualifies there is no risk prediction.

Market reliability weights downgrade markets the models predict less
reliably (corners, handicap, exact score).
"""
from __future__ import annotations

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


def main_prediction(selections: list[PricedSelection]) -> Recommendation | None:
    if not selections:
        return None
    candidates = [s for s in selections if s.group != "EXACT_SCORE"
                  and config.MAIN_MIN_PROBABILITY <= s.probability <= config.MAIN_MAX_PROBABILITY]
    value = [s for s in candidates if s.has_odds and config.MAIN_MIN_ODDS <= s.odds < config.MAIN_MAX_ODDS
             and s.ev >= config.MAIN_MIN_EV]
    if value:
        best = max(value, key=lambda s: (s.ev * reliability(s), s.probability))
        return Recommendation("main", "value", best, best.ev * reliability(best))
    if candidates:
        best = max(candidates, key=lambda s: (s.probability * reliability(s), s.has_odds))
        return Recommendation("main", "model", best, best.probability * reliability(best))
    one_x_two = [s for s in selections if s.market == "1X2"]
    if not one_x_two:
        return None
    best = max(one_x_two, key=lambda s: s.probability)
    return Recommendation("main", "model", best, best.probability)


def risk_prediction(selections: list[PricedSelection]) -> Recommendation | None:
    candidates = [s for s in selections if s.has_odds and s.group != "EXACT_SCORE"
                  and config.RISK_MIN_ODDS <= s.odds <= config.RISK_MAX_ODDS
                  and s.probability >= config.RISK_MIN_PROBABILITY
                  and s.ev >= config.RISK_MIN_EV
                  and reliability(s) >= config.RISK_MIN_RELIABILITY]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: (s.ev * reliability(s), s.probability))
    return Recommendation("risk", "value", best, best.ev * reliability(best))
