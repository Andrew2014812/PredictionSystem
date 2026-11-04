"""Vectorised Main-selection rules on a long candidate table (candidates.py).

``rule_score`` is src/markets/recommendation.rule_score written over columns; ``pick`` is main_prediction():
the candidate with the highest score (ties: priced first, then the more probable), and — when no candidate
lies in the probability band — the most probable 1X2 outcome. self_check() in candidates.py confirms that the
vectorised current rule chooses the same selection as production.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src import config

EXCLUDED = {"EHANDICAP", "EXACT_SCORE"}
EV_CLIP = 0.3


@dataclass(frozen=True)
class Params:
    min_p: float = config.MAIN_MIN_PROBABILITY
    max_p: float = config.MAIN_MAX_PROBABILITY
    preferred: tuple = config.MAIN_PREFERRED_ODDS
    softness: float = config.MAIN_ODDS_SOFTNESS
    ev_weight: float = config.MAIN_EV_WEIGHT
    unpriced_factor: float = config.MAIN_UNPRICED_FACTOR
    power: float = config.MAIN_PROBABILITY_POWER
    value_first_min_ev: float | None = config.MAIN_VALUE_FIRST_MIN_EV
    reliability: tuple = tuple(sorted(config.MARKET_RELIABILITY.items()))
    priced_only: bool = False          # betting universe: only selections with a real price are candidates
    ev_clip: float = EV_CLIP

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def rel(self) -> dict:
        return dict(self.reliability)


CURRENT = Params()


def odds_preference(odds: np.ndarray, preferred=(1.40, 2.50), softness=0.15) -> np.ndarray:
    o = np.asarray(odds, float)
    lo, hi = preferred
    with np.errstate(all="ignore"):
        x = np.log(o)
        d = np.where(o < lo, np.log(lo) - x, np.where(o > hi, x - np.log(hi), 0.0))
        out = np.exp(-0.5 * (d / softness) ** 2) if softness > 0 else (d == 0).astype(float)
    return np.where((o > 1.0) & np.isfinite(o), out, 0.0)


def ev(p, odds, p_push) -> np.ndarray:
    return np.asarray(p, float) * np.asarray(odds, float) - (1 - np.asarray(p_push, float))


def rule_score(df: pd.DataFrame, params: Params = CURRENT, p_col: str = "p", odds_col: str = "odds",
               ev_values: np.ndarray | None = None) -> np.ndarray:
    """Production score; -inf for a selection that is not a candidate."""
    p = df[p_col].to_numpy(float)
    odds = df[odds_col].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    eff = np.where(priced, odds, 1.0 / np.clip(p, 1e-9, None))
    rel = df["group"].map(params.rel()).fillna(0.8).to_numpy(float)
    base = rel * np.clip(p, 0, None) ** params.power * odds_preference(eff, params.preferred, params.softness)
    e = ev(p, odds, df["p_push"].to_numpy(float)) if ev_values is None else ev_values
    with np.errstate(all="ignore"):
        bonus = 1 + params.ev_weight * np.clip(np.nan_to_num(e, nan=0.0), -params.ev_clip, params.ev_clip)
    score = np.where(priced, base * bonus, base * params.unpriced_factor)
    ok = (~df["group"].isin(EXCLUDED).to_numpy()) & (p >= params.min_p) & (p <= params.max_p)
    if params.priced_only:
        ok &= priced
    return np.where(ok, score, -np.inf)


def _production_order() -> dict:
    """Position of every selection in the list production builds (goal_markets + corner_markets): max() keeps the
    first of equal keys, so exact ties (e.g. Asian -1 and -1.5 with the same win probability) resolve the same way."""
    from src.markets.markets import corner_markets, goal_markets
    m = np.full((config.MAX_GOALS + 1, config.MAX_GOALS + 1), 1.0 / (config.MAX_GOALS + 1) ** 2)
    sels = goal_markets(m) + corner_markets(np.full(41, 1 / 41))
    return {(s.market, s.selection): i for i, s in enumerate(sels)}


_ORDER = None


def pick(df: pd.DataFrame, score: np.ndarray, p_col: str = "p", odds_col: str = "odds",
         priced_only: bool = False, fallback: bool = True) -> np.ndarray:
    """Row positions (one per match with a pick) of the best score; fallback = most probable 1X2 outcome."""
    odds = df[odds_col].to_numpy(float)
    priced = (np.isfinite(odds) & (odds > 1.0)).astype(int)
    p = df[p_col].to_numpy(float)
    mi = df["mi"].to_numpy()
    global _ORDER
    if _ORDER is None:
        _ORDER = _production_order()
    pos = np.array([_ORDER.get(k, 999) for k in zip(df["market"].to_numpy(), df["selection"].to_numpy())])
    order = np.lexsort((pos, -p, -priced, -np.nan_to_num(score, neginf=-1e18), mi))
    first = order[np.r_[True, mi[order][1:] != mi[order][:-1]]]
    ok = np.isfinite(score[first])
    chosen = list(first[ok])
    if fallback:
        missing = set(mi[first[~ok]])
        if missing:
            m = np.isin(mi, list(missing)) & (df["market"].to_numpy() == "1X2")
            if priced_only:
                m &= priced.astype(bool)
            sub = np.flatnonzero(m)
            if len(sub):
                o2 = sub[np.lexsort((-p[sub], mi[sub]))]
                f2 = o2[np.r_[True, mi[o2][1:] != mi[o2][:-1]]]
                chosen += list(f2)
    return np.array(sorted(chosen), dtype=int)


def choose(df: pd.DataFrame, params: Params = CURRENT, p_col: str = "p", odds_col: str = "odds",
           ev_values: np.ndarray | None = None) -> np.ndarray:
    """main_prediction() with optional value-first stage."""
    score = rule_score(df, params, p_col, odds_col, ev_values)
    if params.value_first_min_ev is not None:
        e = ev(df[p_col], df[odds_col], df["p_push"]) if ev_values is None else ev_values
        odds = df[odds_col].to_numpy(float)
        priced = np.isfinite(odds) & (odds > 1.0)
        vscore = np.where(priced & (np.nan_to_num(e, nan=-9) >= params.value_first_min_ev), score, -np.inf)
        mi = df["mi"].to_numpy()
        has_value = pd.Series(np.isfinite(vscore)).groupby(mi).transform("any").to_numpy()
        score = np.where(has_value, vscore, score)
    return pick(df, score, p_col, odds_col, params.priced_only)


def pick_current(match_rows: pd.DataFrame) -> pd.DataFrame:
    d = match_rows.reset_index(drop=True)
    return d.iloc[choose(d)]


# ------------------------------------------------------------------------------------------------ evaluation
def outcome(df: pd.DataFrame, idx: np.ndarray, price_col: str = "odds", p_col: str = "p") -> pd.DataFrame:
    """Chosen rows with won / priced / profit at ``price_col`` (PUSH and unknown: no bet)."""
    d = df.iloc[idx]
    price = d[price_col].to_numpy(float)
    status = d["status"].to_numpy()
    priced = np.isfinite(price) & (price > 1.0)
    decided = np.isin(status, [0, 1])
    profit = np.where(priced & decided, np.where(status == 1, price - 1, -1.0), np.nan)
    return pd.DataFrame({"mi": d["mi"].to_numpy(), "match_id": d["match_id"].to_numpy(),
                         "market": d["market"].to_numpy(), "selection": d["selection"].to_numpy(),
                         "group": d["group"].to_numpy(), "line": d["line"].to_numpy(),
                         "p": d[p_col].to_numpy(float), "p_raw": d["p"].to_numpy(float),
                         "p_push": d["p_push"].to_numpy(float),
                         "odds": price, "status": status, "decided": decided, "priced": priced,
                         "won": (status == 1).astype(float), "profit": profit,
                         "league": d["league"].to_numpy(), "date": d["date"].to_numpy(),
                         "row": idx})


def summary(o: pd.DataFrame, n_matches: int | None = None, boot: bool = True) -> dict:
    from . import common as C
    dec = o.loc[o["decided"]]
    bets = dec.loc[dec["priced"]]
    prof = bets["profit"].to_numpy(float)
    lo, hi = C.roi_ci(prof) if boot and len(prof) > 1 else (np.nan, np.nan)
    return {"matches": n_matches, "picks": int(len(o)), "settled": int(len(dec)),
            "hit_rate": float(dec["won"].mean()) if len(dec) else np.nan,
            "mean_prob": float(dec["p"].mean()) if len(dec) else np.nan,
            "bets": int(len(bets)), "priced_share": float(len(bets) / len(dec)) if len(dec) else np.nan,
            "bet_hit_rate": float(bets["won"].mean()) if len(bets) else np.nan,
            "bet_mean_prob": float(bets["p"].mean()) if len(bets) else np.nan,
            "avg_odds": float(bets["odds"].mean()) if len(bets) else np.nan,
            "profit": float(prof.sum()), "roi": float(prof.mean()) if len(prof) else np.nan,
            "roi_ci95": [lo, hi],
            "mean_ev": float(np.mean(bets["p"] * bets["odds"] - (1 - bets["p_push"]))) if len(bets) else np.nan}


def per_match_profit(o: pd.DataFrame, n_matches: int, mi_index: np.ndarray | None = None) -> np.ndarray:
    """Profit per match of the universe (NaN = no bet), for paired bootstrap across strategies."""
    out = np.full(n_matches, np.nan)
    pos = o["mi"].to_numpy() if mi_index is None else mi_index
    out[pos] = o["profit"].to_numpy(float)
    return out
