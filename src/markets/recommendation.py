"""Main prediction and risk prediction: formal selection rules.

MAIN PREDICTION
    1. Candidates: every priced selection except exact scores, with
       p >= MAIN_MIN_PROBABILITY and MAIN_MIN_ODDS <= odds < MAIN_MAX_ODDS.
    2. Value rule: among candidates with EV >= MAIN_MIN_EV pick the highest
       score = EV x market reliability x odds-source reliability
       (kind = "value").
    3. Fallback: if no candidate has enough EV, pick the candidate with the
       highest probability among those with odds >= MAIN_MIN_ODDS
       (kind = "confidence"); the UI says that no value was found.
    4. No candidates at all -> no main prediction.

RISK PREDICTION (optional)
    Candidates with RISK_MIN_ODDS <= odds <= RISK_MAX_ODDS,
    p >= RISK_MIN_PROBABILITY and EV >= RISK_MIN_EV; pick the highest score
    (same formula). Exact scores are excluded. If nothing qualifies the block
    is not shown.

Reliability weights downgrade markets the models predict less reliably
(corners, handicap) and odds that are not real bookmaker prices.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .odds import PricedSelection


@dataclass
class Recommendation:
    role: str                 # "main" | "risk"
    kind: str                 # "value" | "confidence"
    pick: PricedSelection
    score: float

    def to_dict(self) -> dict:
        return {"role": self.role, "kind": self.kind, "score": self.score, **self.pick.to_dict()}


def reliability(sel: PricedSelection) -> float:
    return (config.MARKET_RELIABILITY.get(sel.group, 0.8)
            * config.ODDS_SOURCE_RELIABILITY.get(sel.odds_source or "", 0.0))


def _score(sel: PricedSelection) -> float:
    return (sel.ev or 0.0) * reliability(sel)


def _priced(selections: list[PricedSelection]) -> list[PricedSelection]:
    return [s for s in selections if s.odds and s.group != "EXACT_SCORE"]


def main_prediction(selections: list[PricedSelection]) -> Recommendation | None:
    candidates = [s for s in _priced(selections)
                  if s.probability >= config.MAIN_MIN_PROBABILITY
                  and config.MAIN_MIN_ODDS <= s.odds < config.MAIN_MAX_ODDS]
    if not candidates:
        return None
    value = [s for s in candidates if s.ev >= config.MAIN_MIN_EV]
    if value:
        best = max(value, key=lambda s: (_score(s), s.probability))
        return Recommendation("main", "value", best, _score(best))
    best = max(candidates, key=lambda s: (s.probability * reliability(s), s.odds))
    return Recommendation("main", "confidence", best, _score(best))


def risk_prediction(selections: list[PricedSelection]) -> Recommendation | None:
    candidates = [s for s in _priced(selections)
                  if config.RISK_MIN_ODDS <= s.odds <= config.RISK_MAX_ODDS
                  and s.probability >= config.RISK_MIN_PROBABILITY
                  and s.ev >= config.RISK_MIN_EV]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: (_score(s), s.probability))
    return Recommendation("risk", "value", best, _score(best))
