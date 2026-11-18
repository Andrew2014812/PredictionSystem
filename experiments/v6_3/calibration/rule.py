"""Vectorised production Main rule (``src.markets.recommendation.main_prediction``) on a candidate table.

Identical to ``evaluate.main_by_rule`` (checked in ``run.py`` on every match of both universes) but fast enough for
the many subset / calibration replays of the studies. Any probability column can drive the rule; EV inside the
rule is recomputed from that column: EV = p * odds - (1 - P(push)) — the production formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.markets.recommendation import EV_CLIP, EXCLUDED_GROUPS, RuleParams, main_params


def odds_pref(odds: np.ndarray, preferred=None, softness=None) -> np.ndarray:
    low, high = preferred or config.MAIN_PREFERRED_ODDS
    softness = config.MAIN_ODDS_SOFTNESS if softness is None else softness
    o = np.asarray(odds, float)
    x = np.log(np.where(o > 1.0, o, 2.0))
    d = np.where(o < low, np.log(low) - x, np.where(o > high, x - np.log(high), 0.0))
    out = np.exp(-0.5 * (d / softness) ** 2) if softness > 0 else (d == 0).astype(float)
    return np.where(o > 1.0, out, 0.0)


def rule_scores(c: pd.DataFrame, prob_col: str = "probability", params: RuleParams | None = None) -> pd.DataFrame:
    """score, eligibility and EV (on ``prob_col``) of every row."""
    params = params or main_params()
    p = c[prob_col].to_numpy(float)
    odds = pd.to_numeric(c["odds"], errors="coerce").to_numpy(float)
    push = c["push_probability"].to_numpy(float)
    has = np.isfinite(odds) & (odds > 1.0)
    ev = np.where(has, p * np.where(has, odds, 0) - (1 - push), np.nan)
    eff = np.where(has, odds, np.maximum(1 - push, 1e-9) / np.maximum(p, 1e-9))
    rel = c["group"].map(config.MARKET_RELIABILITY).fillna(0.8).to_numpy(float)
    base = rel * p ** params.probability_power * odds_pref(eff, params.preferred, params.softness)
    score = np.where(has, base * (1 + params.ev_weight * np.clip(np.nan_to_num(ev), -EV_CLIP, EV_CLIP)),
                     base * params.unpriced_factor)
    grp = c["group"].to_numpy()
    elig = (~np.isin(grp, list(EXCLUDED_GROUPS | {"EXACT_SCORE"}))) & (p >= params.min_probability) & (
        p <= params.max_probability)
    return pd.DataFrame({"score": score, "elig": elig, "has": has, "ev_rule": ev, "p_rule": p}, index=c.index)


def choose(c: pd.DataFrame, prob_col: str = "probability", params: RuleParams | None = None,
           mask: np.ndarray | None = None) -> pd.Index:
    """Index (into ``c``) of the Main of every match. ``mask`` optionally limits the candidates (subset studies);
    the fallback (no candidate in the band: the most likely 1X2 outcome) is production's."""
    params = params or main_params()
    s = rule_scores(c, prob_col, params)
    elig = s["elig"].to_numpy() & (np.ones(len(c), bool) if mask is None else np.asarray(mask, bool))
    if params.value_first_min_ev is not None:
        raise NotImplementedError("value-first rule is not used by these studies")
    d = pd.DataFrame({"m": c["match_id"].to_numpy(), "s": s["score"].to_numpy(), "h": s["has"].to_numpy(),
                      "p": s["p_rule"].to_numpy(), "o": np.arange(len(c))}, index=c.index)
    e = d.loc[elig].sort_values(["m", "s", "h", "p", "o"], ascending=[True, False, False, False, True])
    picked = e.drop_duplicates("m")
    missing = set(d["m"]) - set(picked["m"])
    if missing:
        f = d.loc[d["m"].isin(missing) & (c["market"].to_numpy() == "1X2")]
        f = f.sort_values(["m", "p"], ascending=[True, False], kind="mergesort").drop_duplicates("m")
        picked = pd.concat([picked, f])
    return picked.index


def main_table(c: pd.DataFrame, prob_col: str = "probability", params: RuleParams | None = None,
               method: str = "rule", mask=None) -> pd.DataFrame:
    """Chosen rows with ``stated_p`` (the probability the rule used) and ``ev_stated`` (EV on it)."""
    idx = choose(c, prob_col, params, mask)
    out = c.loc[idx].copy()
    odds = pd.to_numeric(out["odds"], errors="coerce")
    out["stated_p"] = out[prob_col].astype(float)
    out["ev_stated"] = np.where(odds > 1, out["stated_p"] * odds - (1 - out["push_probability"]), np.nan)
    out["method"] = method
    return out.sort_values("match_id").reset_index(drop=True)
