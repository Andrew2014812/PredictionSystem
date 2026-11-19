"""Vectorised Main selection: production's rule with any parameters, odds floors, market restrictions, other
probability columns and any alternative score — always exactly one Main per match.

Production rule (``src/markets/recommendation.py``), reproduced here on whole candidate tables (checked against
Baseline B and the stored backtest Main in ``run.py`` → ``self_check``):

    eligible  = group not in {EHANDICAP, EXACT_SCORE} and MIN_P <= p <= MAX_P (p = model probability, also
                when another probability column is used for ranking)
    score     = reliability(group) · p^power · pref(effective odds) · (1 + w_EV · clip(EV, ±0.3))   real price
              = reliability(group) · p^power · pref(effective odds) · UNPRICED_FACTOR              no real price
    Main      = argmax score (ties: real price, then higher p); no eligible selection → the most likely 1X2 outcome

A rule with a hard constraint (minimum real price, real price mandatory, restricted markets) that leaves a match
without an eligible selection keeps the STATISTICAL Main of that match (production's choice) but, when the
constraint is about the price, marks it "not a bet": it is shown and settled as a prediction, it is not counted
in ROI, and ``note`` says why. Nothing is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from src import config

PROD_RELIABILITY = dict(config.MARKET_RELIABILITY)
EV_CLIP = 0.3


@dataclass(frozen=True)
class Rule:
    name: str
    min_p: float = config.MAIN_MIN_PROBABILITY
    max_p: float = config.MAIN_MAX_PROBABILITY
    preferred: tuple = tuple(config.MAIN_PREFERRED_ODDS)
    softness: float = config.MAIN_ODDS_SOFTNESS
    ev_weight: float = config.MAIN_EV_WEIGHT
    unpriced: float = config.MAIN_UNPRICED_FACTOR
    power: float = config.MAIN_PROBABILITY_POWER
    reliability: tuple = tuple(sorted(PROD_RELIABILITY.items()))
    prob: str = "probability"               # probability column used for RANKING (shown p stays the model's)
    min_odds: float | None = None           # hard floor on the real price (implies a real price)
    require_price: bool = False             # only selections with a real price are eligible
    min_eff_odds: float | None = None       # floor on the effective price (real, or break-even when unpriced)
    groups: frozenset | None = None         # allowed market groups (None = all Main groups)
    priority: tuple = ()                    # lexicographic market priority (first group with an eligible selection)
    score: Callable | None = field(default=None, compare=False)   # f(c, p, ev) -> array replaces the formula
    multiplier: Callable | None = field(default=None, compare=False)  # f(c) -> array multiplies the score
    fallback_bet: bool = False              # price-constrained rule: is the statistical fallback Main a bet?
                                            # False = correct (not a bet); True = the constraint broken silently
    value_first_min_ev: float | None = None

    def with_(self, **kw) -> "Rule":
        return replace(self, **kw)

    @property
    def rel(self) -> dict:
        return dict(self.reliability)


PRODUCTION = Rule("production")


def with_reliability(rule: Rule, name: str, **groups) -> Rule:
    r = dict(PROD_RELIABILITY)
    r.update(groups)
    return rule.with_(name=name, reliability=tuple(sorted(r.items())))


def odds_pref(odds: np.ndarray, preferred, softness: float) -> np.ndarray:
    low, high = preferred
    x = np.log(np.maximum(odds, 1e-9))
    d = np.where(odds < low, np.log(low) - x, np.where(odds > high, x - np.log(high), 0.0))
    out = np.exp(-0.5 * (d / softness) ** 2) if softness > 0 else (d == 0).astype(float)
    return np.where(odds > 1.0, out, 0.0)


def rank_inputs(c: pd.DataFrame, rule: Rule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p used for ranking, EV with that p at the real price, effective odds with that p)."""
    p = c[rule.prob].to_numpy(float)
    p = np.where(np.isfinite(p), p, c["probability"].to_numpy(float))
    push = c["push_probability"].to_numpy(float)
    odds = c["odds"].to_numpy(float)
    has = c["has_odds"].to_numpy(bool)
    ev = np.where(has, p * odds - (1 - push), np.nan)
    eff = np.where(has, odds, np.maximum(1 - push, 1e-9) / np.maximum(p, 1e-9))
    return p, ev, eff


def rule_score(c: pd.DataFrame, rule: Rule) -> np.ndarray:
    p, ev, eff = rank_inputs(c, rule)
    if rule.score is not None:
        s = np.asarray(rule.score(c, p, ev), float)
    else:
        rel = c["group"].map(rule.rel).fillna(0.8).to_numpy(float)
        base = rel * np.power(np.clip(p, 0, None), rule.power) * odds_pref(eff, rule.preferred, rule.softness)
        has = c["has_odds"].to_numpy(bool)
        s = np.where(has, base * (1 + rule.ev_weight * np.clip(np.nan_to_num(ev), -EV_CLIP, EV_CLIP)),
                     base * rule.unpriced)
    if rule.multiplier is not None:
        s = s * np.asarray(rule.multiplier(c), float)
    return s


def eligible(c: pd.DataFrame, rule: Rule) -> np.ndarray:
    p = c["probability"].to_numpy(float)      # the band is a property of the SHOWN Main: model probability
    ok = ~c["excluded"].to_numpy(bool) & (p >= rule.min_p) & (p <= rule.max_p)
    if rule.groups is not None:
        ok &= c["group"].isin(rule.groups).to_numpy()
    if rule.require_price or rule.min_odds is not None:
        ok &= c["has_odds"].to_numpy(bool)
    if rule.min_odds is not None:
        ok &= np.nan_to_num(c["odds"].to_numpy(float)) >= rule.min_odds - 1e-9
    if rule.min_eff_odds is not None:
        _, _, eff = rank_inputs(c, rule)
        ok &= eff >= rule.min_eff_odds - 1e-9
    return ok


def _argmax(c: pd.DataFrame, mask: np.ndarray, score: np.ndarray, prio: np.ndarray | None = None) -> pd.Series:
    """Row index of the best masked row of every match (ties: real price, then higher p, then table order)."""
    idx = np.flatnonzero(mask & np.isfinite(score))
    if len(idx) == 0:
        return pd.Series(dtype=int)
    mi = c["mi"].to_numpy()[idx]
    keys = [idx, -c["probability"].to_numpy(float)[idx], -c["has_odds"].to_numpy(int)[idx], -score[idx]]
    if prio is not None:
        keys.append(prio[idx])
    keys.append(mi)
    order = idx[np.lexsort(keys)]
    first = ~pd.Series(c["mi"].to_numpy()[order]).duplicated().to_numpy()
    best = order[first]
    return pd.Series(best, index=c["mi"].to_numpy()[best])


def statistical_main(c: pd.DataFrame) -> pd.Series:
    """Production Main (row index by match index), including the 'most likely 1X2' fallback."""
    return choose(c, PRODUCTION)


def choose(c: pd.DataFrame, rule: Rule) -> pd.Series:
    """Row index of the rule's Main for every match that has an eligible selection (+ production's 1X2 fallback
    for the plain rule)."""
    s = rule_score(c, rule)
    ok = eligible(c, rule)
    prio = None
    if rule.priority:
        order = {g: i for i, g in enumerate(rule.priority)}
        prio = c["group"].map(order).fillna(len(order)).to_numpy(float)
    best = _argmax(c, ok, s, prio)
    if rule.value_first_min_ev is not None:
        _, ev, _ = rank_inputs(c, rule)
        v = _argmax(c, ok & c["has_odds"].to_numpy(bool) & (np.nan_to_num(ev, nan=-9) >= rule.value_first_min_ev), s)
        best = pd.concat([v, best.loc[~best.index.isin(v.index)]])
    plain = (rule.groups is None and not rule.require_price and rule.min_odds is None
             and rule.min_eff_odds is None)
    if plain:          # production: no selection in the band -> the most likely 1X2 outcome
        miss = ~np.isin(c["mi"].to_numpy(), best.index.to_numpy())
        x = _argmax(c, miss & (c["market"] == "1X2").to_numpy(), c["probability"].to_numpy(float))
        best = pd.concat([best, x])
    return best.sort_index()


def select(c: pd.DataFrame, rule: Rule, stat: pd.Series | None = None) -> pd.DataFrame:
    """One Main per match. Matches without an eligible selection keep the statistical (production) Main; when the
    rule constrains the price, that Main is NOT a bet (``bet`` 0, ``profit`` NaN, ``note`` explains)."""
    best = choose(c, rule)
    stat = statistical_main(c) if stat is None else stat
    miss = stat.loc[~stat.index.isin(best.index)]
    main = c.iloc[best.to_numpy()].assign(note="")
    if len(miss):
        fb = c.iloc[miss.to_numpy()].copy()
        price_rule = rule.min_odds is not None or rule.require_price or rule.min_eff_odds is not None
        if price_rule and not rule.fallback_bet:
            why = (f"fallback: no selection with a real price >= {rule.min_odds:.2f}" if rule.min_odds
                   else f"fallback: no selection with a price >= {rule.min_eff_odds:.2f}" if rule.min_eff_odds
                   else "fallback: no eligible selection with a real price")
            fb["note"] = why + " - statistical Main shown, not a bet"
            fb["bet"] = 0
            fb["profit"] = np.nan
        elif price_rule:
            fb["note"] = "fallback: constraint broken silently - statistical Main counted as a bet"
        else:
            fb["note"] = "fallback: no eligible selection - statistical Main"
        main = pd.concat([main, fb])
    main = main.sort_values("mi").reset_index(drop=True)
    main["method"] = rule.name
    assert main["match_id"].is_unique and len(main) == c["mi"].nunique(), rule.name
    return main
