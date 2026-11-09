"""What each shadow computes for the upcoming matches of one production run.

Every function receives the production state of the run (``Context``: the stored predictions of the matches
that have not kicked off, their priced selections, the whole prediction history, the feature table) and the
frozen preregistration document of its shadow, and returns one row per (match, item):

    match_id, item, market, selection, p          the shadow's pick and its stated probability
    base_market, base_selection, base_p           the production pick / probability it is compared with
    odds, bookmaker, provider, priced_at          the real price of the shadow's pick at this moment (if any)
    base_odds, base_priced_at                     the real price of the production pick at this moment
    payload                                       variant-specific numbers (dict, stored as JSON)

Nothing here writes anything; production functions are only called, never changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..markets.odds import PricedSelection
from ..markets.recommendation import (EXCLUDED_GROUPS, opposite_outcome, risk_params, rule_score, same_outcome)

EPS = 1e-4
ROW_COLUMNS = ["match_id", "item", "market", "selection", "p", "base_market", "base_selection", "base_p", "odds",
               "bookmaker", "provider", "priced_at", "base_odds", "base_priced_at", "payload"]
COMPLETE_BOOK = {"1X2": 3}
NO_BOOK = {"DOUBLE_CHANCE", "EXACT_SCORE", "EHANDICAP"}


@dataclass
class Context:
    now: pd.Timestamp                    # UK local time (the clock of kick-off times)
    snapshots: pd.DataFrame              # stored production predictions of the matches to shadow (one per match)
    markets: pd.DataFrame                # their priced selections (production markets table rows)
    picks: pd.DataFrame                  # their stored picks (main / risk / market)
    history: pd.DataFrame                # every stored production pick (for shadows fitted on settled history)
    history_push: pd.DataFrame           # (match_id, market, selection, push_probability) of the history picks
    features: pd.DataFrame | None = None
    research_dir: Path | None = None
    model_meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ helpers
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _ts(x):
    return pd.Timestamp(x) if x is not None and not pd.isna(x) else pd.NaT


def selections(mk: pd.DataFrame) -> list[PricedSelection]:
    """PricedSelection objects of one match from its markets rows (real prices only)."""
    out = []
    for r in mk.itertuples(index=False):
        odds = _f(r.odds)
        out.append(PricedSelection(r.market, r.selection, float(r.probability),
                                   odds if odds == odds and odds > 1.0 else None, r.bookmaker, r.provider,
                                   float(_f(r.push_probability)) if _f(r.push_probability) == _f(r.push_probability)
                                   else 0.0, _ts(r.priced_at)))
    return out


def market_probability(sels: list[PricedSelection]) -> dict[tuple[str, str], float]:
    """Proportional de-vig of every complete real book of the match (Asian whole line: P(win | no refund))."""
    by_market: dict[str, list[PricedSelection]] = {}
    for s in sels:
        by_market.setdefault(s.market, []).append(s)
    out = {}
    for market, group in by_market.items():
        grp = group[0].group
        if grp in NO_BOOK:
            continue
        need = COMPLETE_BOOK.get(grp, 2)
        priced = [s for s in group if s.has_odds]
        if len(priced) != need or len(group) != need:
            continue
        inv = np.array([1.0 / s.odds for s in priced])
        for s, q in zip(priced, inv / inv.sum()):
            out[(s.market, s.selection)] = float(q)
    return out


def selection_key(sel: PricedSelection) -> str:
    """Selection type of the Risk study: 1X2:D, TOTAL_2.5:OVER, HANDICAP:H, CORNERS_9.5:UNDER, EXACT_SCORE."""
    g = sel.group
    if g == "EXACT_SCORE":
        return g
    base = sel.market if g in ("TOTAL", "CORNERS") else g
    return f"{base}:{sel.selection}"


def _pick(picks: pd.DataFrame, match_id: str, role: str) -> pd.Series | None:
    rows = picks.loc[(picks["match_id"] == match_id) & (picks["role"] == role)]
    return rows.iloc[-1] if len(rows) else None


def _row(match_id, item, market, selection, p, base, odds=np.nan, bookmaker=None, provider=None, priced_at=pd.NaT,
         payload=None) -> dict:
    return {"match_id": match_id, "item": item, "market": market, "selection": selection, "p": _f(p),
            "base_market": None if base is None else base.get("market"),
            "base_selection": None if base is None else base.get("selection"),
            "base_p": np.nan if base is None else _f(base.get("probability")),
            "odds": _f(odds), "bookmaker": bookmaker, "provider": provider, "priced_at": _ts(priced_at),
            "base_odds": np.nan if base is None else _f(base.get("odds")),
            "base_priced_at": pd.NaT if base is None else _ts(base.get("odds_priced_at")),
            "payload": payload or {}}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def _push(mk: pd.DataFrame, market: str, selection: str) -> float:
    r = mk.loc[(mk["market"] == market) & (mk["selection"] == selection), "push_probability"]
    v = _f(r.iloc[-1]) if len(r) else np.nan
    return 0.0 if v != v else v


# ------------------------------------------------------------------------------------------------ Main calibration
def _main_rows(ctx: Context, fn, name: str) -> pd.DataFrame:
    rows = []
    for mid, mk in ctx.markets.groupby("match_id", sort=False):
        main = _pick(ctx.picks, mid, "main")
        if main is None:
            continue
        push = _push(mk, main["market"], main["selection"])
        pc = float(main["probability"]) / max(1 - push, 1e-6)
        p_new, info = fn(main, pc, mid)
        rows.append(_row(mid, "main", main["market"], main["selection"], p_new, main, main.get("odds"),
                         main.get("bookmaker"), main.get("provider"), main.get("odds_priced_at"),
                         {"pc": pc, "push": push, "group": main["group"], **info}))
    return _frame(rows)


def ps_group(ctx: Context, doc: dict) -> pd.DataFrame:
    params = doc["hypotheses"]["H2_challenger"]["frozen_parameters"]

    def fn(main, pc, _mid):
        g = main["group"] if main["group"] in params else "pool"
        a, b = params[g]["a"], params[g]["b"]
        return float(sigmoid(a + b * logit(pc))), {"map": g, "a": a, "b": b}
    return _main_rows(ctx, fn, "ps_group")


def monthly_platt_fit(history: pd.DataFrame, history_push: pd.DataFrame, month_start: pd.Timestamp,
                      season: int) -> dict:
    """(a, b) of the H1 map for the month starting ``month_start``: logistic regression (C = 1e4) of WON / LOST on
    logit(pc) over every settled shown Main of the unified history of ``season`` dated before the month."""
    from sklearn.linear_model import LogisticRegression
    from ..analytics.performance import unified_history
    h = unified_history(history)
    h = h.loc[(h["role"] == "main") & (h["season"] == season) & h["status"].isin(["WON", "LOST"])
              & (pd.to_datetime(h["date"]) < month_start)]
    if h.empty:
        return {"n": 0, "a": np.nan, "b": np.nan, "reason": "no settled Main before the month"}
    h = h.merge(history_push, how="left", on=["match_id", "market", "selection"])
    pc = h["probability"].astype(float) / (1 - h["push_probability"].astype(float).fillna(0.0))
    y = (h["status"] == "WON").astype(float).to_numpy()
    if len(h) < 50 or y.min() == y.max():
        return {"n": int(len(h)), "a": np.nan, "b": np.nan, "reason": "fewer than 50 settled Mains"}
    lr = LogisticRegression(C=1e4, max_iter=500).fit(logit(pc.to_numpy()).reshape(-1, 1), y)
    return {"n": int(len(h)), "a": float(lr.intercept_[0]), "b": float(lr.coef_[0, 0]),
            "last_date": str(pd.to_datetime(h["date"]).max().date())}


def ps_platt_monthly(ctx: Context, doc: dict) -> pd.DataFrame:
    fits: dict = {}
    dates = ctx.snapshots.set_index("match_id")["date"]
    seasons = ctx.snapshots.set_index("match_id")["season"]

    def fn(main, pc, mid):
        month = pd.Timestamp(dates.get(mid)).normalize().replace(day=1)
        key = (month, int(seasons.get(mid)))
        if key not in fits:
            fits[key] = monthly_platt_fit(ctx.history, ctx.history_push, month, key[1])
        fit = fits[key]
        p = float(sigmoid(fit["a"] + fit["b"] * logit(pc))) if fit["a"] == fit["a"] else np.nan
        return p, {"month": str(month.date()), **fit}
    return _main_rows(ctx, fn, "ps_platt_monthly")


# ------------------------------------------------------------------------------------------------ Risk
def _production_main(sels: list[PricedSelection], main: pd.Series | None) -> PricedSelection | None:
    if main is None:
        return None
    for s in sels:
        if (s.market, s.selection) == (main["market"], main["selection"]):
            return s
    return None


def risk_pool(sels: list[PricedSelection], main: PricedSelection | None) -> list[PricedSelection]:
    """Candidates of src.markets.recommendation.risk_prediction (same eligibility, same probability floor)."""
    params = risk_params()
    eligible = [s for s in sels if s.group not in EXCLUDED_GROUPS
                and not (main is not None and (same_outcome(s, main) or opposite_outcome(s, main)))]
    cand = [s for s in eligible if s.probability >= params.min_probability]
    return cand or eligible


def risk_stacked(ctx: Context, doc: dict) -> pd.DataFrame:
    h1 = doc["hypotheses"]["H1_rule"]
    coef = h1["frozen_parameters"]                     # key -> [b1 (logit pc), b2 (logit qc), b0]
    params = risk_params()
    rows = []
    for mid, mk in ctx.markets.groupby("match_id", sort=False):
        risk = _pick(ctx.picks, mid, "risk")
        if risk is None:
            continue
        sels = selections(mk)
        main = _production_main(sels, _pick(ctx.picks, mid, "main"))
        pool = risk_pool(sels, main)
        q = market_probability(sels)
        tier1 = []
        for s in pool:
            qc = q.get((s.market, s.selection))
            if not s.has_odds or qc is None:
                continue
            b1, b2, b0 = coef.get(selection_key(s), coef["*"])
            pc = s.probability / max(1 - s.push_probability, 1e-6)
            ps = float(sigmoid(b1 * logit(pc) + b2 * logit(qc) + b0)) * (1 - s.push_probability)
            if ps * s.odds - (1 - s.push_probability) >= 0:
                tier1.append((s, ps, qc))
        if tier1:
            best = max(tier1, key=lambda t: (rule_score(PricedSelection(
                t[0].market, t[0].selection, t[1], t[0].odds, push_probability=t[0].push_probability), params),
                t[0].has_odds, t[1]))
            s, ps, qc = best
            rows.append(_row(mid, "risk", s.market, s.selection, ps, risk, s.odds, s.bookmaker, s.provider,
                             s.priced_at, {"tier": 1, "n_tier1": len(tier1), "qc": qc, "p_model": s.probability,
                                           "ev_stacked": ps * s.odds - (1 - s.push_probability),
                                           "push": s.push_probability}))
        else:
            rows.append(_row(mid, "risk", risk["market"], risk["selection"], risk["probability"], risk,
                             risk.get("odds"), risk.get("bookmaker"), risk.get("provider"), risk.get("odds_priced_at"),
                             {"tier": 2, "n_tier1": 0}))
    return _frame(rows)


def risk_market_probability(ctx: Context, doc: dict) -> pd.DataFrame:
    rows = []
    for mid, mk in ctx.markets.groupby("match_id", sort=False):
        risk = _pick(ctx.picks, mid, "risk")
        if risk is None:
            continue
        sels = selections(mk)
        q = market_probability(sels).get((risk["market"], risk["selection"]))
        push = _push(mk, risk["market"], risk["selection"])
        rows.append(_row(mid, "risk", risk["market"], risk["selection"], np.nan if q is None else q, risk,
                         risk.get("odds"), risk.get("bookmaker"), risk.get("provider"), risk.get("odds_priced_at"),
                         {"full_book": q is not None, "push": push,
                          "pc": float(risk["probability"]) / max(1 - push, 1e-6)}))
    return _frame(rows)


# ------------------------------------------------------------------------------------------------ bookmaker pool
def power_demargin(odds: np.ndarray) -> np.ndarray:
    """Power method (bookmaker study): q_i = pi_i^k with sum q_i = 1, bisection on k."""
    pi = np.clip(1.0 / np.asarray(odds, float), 1e-9, 1 - 1e-9)
    lo, hi = np.full((len(pi), 1), 0.5), np.full((len(pi), 1), 3.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = (pi ** mid).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot <= 1, mid, hi)
    q = pi ** ((lo + hi) / 2)
    return q / q.sum(1, keepdims=True)


def pool_predict(theta, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """softmax_k(a_k + b1 log q_k + b2 log p_k), a_1 = 0, theta = [a_2..a_K, b1, b2] (bookmaker study)."""
    theta = np.asarray(theta, float)
    k = q.shape[1]
    a = np.r_[0.0, theta[:k - 1]]
    z = a + theta[k - 1] * np.log(np.clip(q, 1e-6, 1)) + theta[k] * np.log(np.clip(p, 1e-6, 1))
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bookmaker_h2(ctx: Context, doc: dict) -> pd.DataFrame:
    fx = None
    if ctx.features is not None and "odds_home" in ctx.features:
        fx = ctx.features.set_index("match_id")
    rows = []
    for snap in ctx.snapshots.itertuples(index=False):
        mid = snap.match_id
        p = np.array([snap.p_home, snap.p_draw, snap.p_away], float)
        p = p / p.sum()
        mk = ctx.markets.loc[ctx.markets["match_id"] == mid]
        over = mk.loc[(mk["market"] == "TOTAL_2.5") & (mk["selection"] == "OVER"), "probability"]
        fixture = {}
        if fx is not None and mid in fx.index:
            r = fx.loc[mid]
            r = r.iloc[-1] if isinstance(r, pd.DataFrame) else r
            fixture = {k: _f(r.get(k)) for k in ("odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25")}
        base = {"market": "1X2", "selection": "HDA", "probability": np.nan}
        rows.append(_row(mid, "1x2", "1X2", "HDA", np.nan, base,
                         payload={"p_model": p.tolist(), "fixture_b365": [fixture.get(k) for k in
                                                                          ("odds_home", "odds_draw", "odds_away")]}))
        if len(over):
            po = float(over.iloc[-1])
            base = {"market": "TOTAL_2.5", "selection": "OVER", "probability": po}
            rows.append(_row(mid, "ou25", "TOTAL_2.5", "OVER", np.nan, base,
                             payload={"p_model": [po, 1 - po], "fixture_b365": [fixture.get("odds_over25"),
                                                                                fixture.get("odds_under25")]}))
    return _frame(rows)


# ------------------------------------------------------------------------------------------------ xG X5
def xg_x5(ctx: Context, doc: dict) -> pd.DataFrame:
    from . import xg_x5 as X
    return X.shadow_rows(ctx, doc)


COMPUTE = {"maincal_ps_platt_monthly": ps_platt_monthly, "maincal_ps_group": ps_group,
           "risk_stacked": risk_stacked, "risk_market_probability": risk_market_probability,
           "bookmaker_h2_pool": bookmaker_h2, "xg_x5": xg_x5}

_ = config
