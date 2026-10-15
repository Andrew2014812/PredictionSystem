"""Main prediction and risk prediction: formal selection rules.

Both exist for every match that has model probabilities; neither ever
invents a price. A selection's "effective odds" are its real bookmaker odds
when one exists, otherwise the fair odds 1 / p (used only for ranking, never
shown as a price).

MAIN PREDICTION — the most defensible prediction of the match.

    Candidates: every selection except exact scores and European handicaps
    with probability in [MAIN_MIN_PROBABILITY, MAIN_MAX_PROBABILITY].

        score = reliability(market) x p^MAIN_PROBABILITY_POWER x odds_preference(effective odds)
                x (1 + MAIN_EV_WEIGHT x clip(EV))      with a real price
                x MAIN_UNPRICED_FACTOR                  without a real price

    odds_preference is 1 inside MAIN_PREFERRED_ODDS (about 1.40–2.50) and
    decays smoothly outside (a Gaussian in log-odds) — a 1.37 or 2.60 is
    barely penalised, a 1.10 strongly. Nothing is chosen on EV alone.

RISK PREDICTION — the most sensible higher-odds alternative.

    Candidates: every selection except European handicaps, the Main
    selection, any selection that wins on exactly the same scores as Main
    (e.g. "home -0.5" when Main is "home win") and any bet AGAINST Main (wins
    exactly when Main loses, e.g. "over 3.5" when Main is "under 3.5"), with
    p >= RISK_MIN_PROBABILITY. A different outcome of the same market (a draw
    when Main is a home win) is allowed: it is an alternative, not the opposite.

        score = reliability(market) x p^RISK_PROBABILITY_POWER x odds_preference(effective odds, RISK_PREFERRED_ODDS)
                x (1 + RISK_EV_WEIGHT x clip(EV))      with a real price
                x RISK_UNPRICED_FACTOR                  without a real price

    Positive EV is NOT required: a risk prediction is by definition less
    likely. The probability factor keeps it a meaningful model result rather
    than the longest price on the board.

All weights were chosen on the VALIDATION season only (scripts/decision_study.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .. import config
from .markets import settle_selection
from .odds import PricedSelection

EXCLUDED_GROUPS = {"EHANDICAP"}
EV_CLIP = 0.3


@dataclass(frozen=True)
class RuleParams:
    min_probability: float
    max_probability: float
    preferred: tuple[float, float]
    softness: float
    ev_weight: float
    unpriced_factor: float
    probability_power: float = 1.0      # < 1 weakens the pull towards the most likely (lowest-odds) selection

    def with_(self, **kw) -> "RuleParams":
        return replace(self, **kw)


def main_params() -> RuleParams:
    return RuleParams(config.MAIN_MIN_PROBABILITY, config.MAIN_MAX_PROBABILITY, config.MAIN_PREFERRED_ODDS,
                      config.MAIN_ODDS_SOFTNESS, config.MAIN_EV_WEIGHT, config.MAIN_UNPRICED_FACTOR,
                      config.MAIN_PROBABILITY_POWER)


def risk_params() -> RuleParams:
    return RuleParams(config.RISK_MIN_PROBABILITY, 1.0, config.RISK_PREFERRED_ODDS,
                      config.RISK_ODDS_SOFTNESS, config.RISK_EV_WEIGHT, config.RISK_UNPRICED_FACTOR,
                      config.RISK_PROBABILITY_POWER)


@dataclass
class Recommendation:
    role: str                 # "main" | "risk"
    kind: str                 # "priced" (real odds) | "model" (no real price yet)
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


def effective_odds(sel: PricedSelection) -> float:
    return sel.odds if sel.has_odds else 1.0 / max(sel.probability, 1e-9)


def rule_score(sel: PricedSelection, params: RuleParams) -> float:
    base = (reliability(sel) * sel.probability ** params.probability_power
            * odds_preference(effective_odds(sel), params.preferred, params.softness))
    if sel.has_odds:
        return base * (1 + params.ev_weight * float(np.clip(sel.ev, -EV_CLIP, EV_CLIP)))
    return base * params.unpriced_factor


def _choose(candidates: list[PricedSelection], params: RuleParams, role: str) -> Recommendation | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda s: (rule_score(s, params), s.has_odds, s.probability))
    return Recommendation(role, "priced" if best.has_odds else "model", best, rule_score(best, params))


def main_prediction(selections: list[PricedSelection], params: RuleParams | None = None) -> Recommendation | None:
    params = params or main_params()
    eligible = [s for s in selections if s.group not in EXCLUDED_GROUPS | {"EXACT_SCORE"}]
    candidates = [s for s in eligible if params.min_probability <= s.probability <= params.max_probability]
    rec = _choose(candidates, params, "main")
    if rec is not None:
        return rec
    one_x_two = [s for s in selections if s.market == "1X2"]          # no selection in the band
    if not one_x_two:
        return None
    best = max(one_x_two, key=lambda s: s.probability)
    return Recommendation("main", "priced" if best.has_odds else "model", best, best.probability)


# -- risk ------------------------------------------------------------------------------
_SCORES = [(h, a) for h in range(8) for a in range(8)]


@lru_cache(maxsize=None)
def _footprint(market: str, selection: str) -> tuple:
    """Settlement of a goal-market selection on every score up to 7:7."""
    return tuple(settle_selection(market, selection, h, a) for h, a in _SCORES)


def same_outcome(a: PricedSelection, b: PricedSelection) -> bool:
    """True if both selections win, lose and refund on exactly the same results."""
    if (a.market, a.selection) == (b.market, b.selection):
        return True
    if a.group == "CORNERS" or b.group == "CORNERS":
        return False
    return _footprint(a.market, a.selection) == _footprint(b.market, b.selection)


def opposite_outcome(a: PricedSelection, b: PricedSelection) -> bool:
    """True if ``a`` wins exactly when ``b`` loses (a bet against the other), e.g. over / under the
    same line, BTTS yes / no, double chance 1X / away win."""
    if a.group == "CORNERS" or b.group == "CORNERS":
        return a.market == b.market and a.selection != b.selection
    flip = {"WON": "LOST", "LOST": "WON", "PUSH": "PUSH"}
    return _footprint(a.market, a.selection) == tuple(flip.get(x, x) for x in _footprint(b.market, b.selection))


def risk_prediction(selections: list[PricedSelection], main: PricedSelection | None = None,
                    params: RuleParams | None = None) -> Recommendation | None:
    """Always returns a risk prediction when there is any eligible selection."""
    params = params or risk_params()
    eligible = [s for s in selections if s.group not in EXCLUDED_GROUPS
                and not (main is not None and (same_outcome(s, main) or opposite_outcome(s, main)))]
    candidates = [s for s in eligible if s.probability >= params.min_probability]
    return _choose(candidates or eligible, params, "risk")
