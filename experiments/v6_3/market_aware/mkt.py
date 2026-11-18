"""Market mathematics shared by the three market studies of v6.3 (market-aware Main, timing & CLV, consensus).

* Quotes. ``odds_long`` of the shared replay (every stored real price strictly before kick-off: OddsPapi bet365 /
  Pinnacle / William Hill, The Odds API books, Football-Data Bet365). Each quote gets a ``book`` key; Football-Data
  Bet365 is kept apart from OddsPapi bet365 (``fd_bet365``) because it is another snapshot of the same bookmaker.
* Moments. A price is used at a moment only if ``collected_at <= moment`` (UK time, as the replay):
  ``t09`` = the replay decision time T, ``close`` = one second before kick-off, ``early`` = kick-off − 48 h.
* Complete books. A market probability exists only when ONE book quotes every selection of the market at or before
  the moment (each with its own latest quote). The book is then de-margined (proportional, power, Shin, odds-ratio /
  logarithmic, additive — the formulas of v6.2 ``odds_timing/common.py``, copied to avoid side effects).
* Asian handicaps. A two-way handicap book gives P(win | no refund); the model's probability is compared on the same
  conditional scale p / (1 − P(refund)). Double chance has no partition book: it is derived from the 1X2 book.
* Main rule. ``main_vec`` is production's Main rule (``src/markets/recommendation.py``) vectorised over the candidate
  table; ``self_check`` verifies it reproduces Baseline B exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV
from src import config
from src.markets.recommendation import EV_CLIP, RuleParams, main_params

CACHE = C.SHARED_CACHE
SIDES = {"1X2": ("H", "D", "A"), "TOTAL": ("OVER", "UNDER"), "CORNERS": ("OVER", "UNDER"), "BTTS": ("YES", "NO"),
         "HANDICAP": ("H", "A")}
BOOK_GROUPS = tuple(SIDES)
MONTHS = ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
METHODS_ORDER = ("power", "prop", "shin", "log", "add")
METHOD_RU = {"prop": "пропорциональный", "shin": "Шин", "power": "степенной", "log": "odds-ratio (логарифм.)",
             "add": "аддитивный"}


def group_of(market: str) -> str:
    return market.split("_")[0] if "_" in market and market.split("_")[0] in (
        "TOTAL", "HANDICAP", "CORNERS", "EHANDICAP") else market


# ------------------------------------------------------------------------------------------------ de-margining
def proportional(pi: np.ndarray) -> np.ndarray:
    return pi / pi.sum(1, keepdims=True)


def shin(pi: np.ndarray) -> np.ndarray:
    s = pi.sum(1, keepdims=True)
    lo, hi = np.zeros((len(pi), 1)), np.full((len(pi), 1), 0.5)

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / s) - z) / (2 * (1 - z))
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = probs(mid).sum(1, keepdims=True)
        hi = np.where(tot < 1, mid, hi)
        lo = np.where(tot >= 1, mid, lo)
    p = probs((lo + hi) / 2)
    return p / p.sum(1, keepdims=True)


def power(pi: np.ndarray) -> np.ndarray:
    pi = np.clip(pi, 1e-9, 1 - 1e-9)
    lo, hi = np.full((len(pi), 1), 0.5), np.full((len(pi), 1), 3.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = (pi ** mid).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot <= 1, mid, hi)
    q = pi ** ((lo + hi) / 2)
    return q / q.sum(1, keepdims=True)


def logarithmic(pi: np.ndarray) -> np.ndarray:
    pi = np.clip(pi, 1e-9, 1 - 1e-9)
    lg = np.log(pi / (1 - pi))
    lo, hi = np.full((len(pi), 1), -1.0), np.full((len(pi), 1), 3.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = (1 / (1 + np.exp(-(lg - mid)))).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot <= 1, mid, hi)
    q = 1 / (1 + np.exp(-(lg - (lo + hi) / 2)))
    return q / q.sum(1, keepdims=True)


def additive(pi: np.ndarray) -> np.ndarray:
    k = pi.shape[1]
    q = np.clip(pi - (pi.sum(1, keepdims=True) - 1) / k, 1e-4, None)
    return q / q.sum(1, keepdims=True)


METHODS = {"prop": proportional, "shin": shin, "power": power, "log": logarithmic, "add": additive}


def demargin(odds: np.ndarray, method: str = "power") -> np.ndarray:
    o = np.asarray(odds, float)
    out = np.full(o.shape, np.nan)
    ok = np.all(np.isfinite(o) & (o > 1.0), 1)
    if ok.any():
        out[ok] = METHODS[method](1.0 / o[ok])
    return out


# ------------------------------------------------------------------------------------------------ quotes
def book_key(provider: str, bookmaker: str) -> str:
    b = str(bookmaker).lower().replace(" ", "")
    if provider == "football_data":
        return "fd_bet365"
    if b in ("williamhill",):
        return "williamhill"
    return b


def load_odds() -> pd.DataFrame:
    """Every stored real quote of the season's matches for the market groups with a book (no exact scores,
    no European handicaps), with its book key and group."""
    o = pd.read_parquet(CACHE / "odds_long.parquet",
                        columns=["match_id", "provider", "bookmaker", "market", "selection", "odds", "collected_at"])
    o = o.loc[~o["market"].isin(["EXACT_SCORE"]) & ~o["market"].str.startswith("EHANDICAP")]
    o = o.loc[pd.to_numeric(o["odds"], errors="coerce") > 1.0].copy()
    o["group"] = o["market"].map(group_of)
    o["book"] = [book_key(p, b) for p, b in zip(o["provider"], o["bookmaker"])]
    return o.reset_index(drop=True)


def matches() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "matches.parquet", columns=["match_id", "league", "date", "kickoff", "home_team",
                                                               "away_team", "played", "decision_time"])


def moment(m: pd.DataFrame, which: str) -> pd.Series:
    """match_id -> decision moment of a universe (UK time)."""
    by = m.set_index("match_id")
    ko = pd.to_datetime(by["kickoff"]).fillna(pd.to_datetime(by["date"]) + pd.Timedelta(hours=15))
    if which == "t09":
        return pd.to_datetime(by["decision_time"])
    if which == "close":
        return ko - pd.Timedelta(seconds=1)
    if which == "early":
        return ko - pd.Timedelta(hours=48)
    raise ValueError(which)


def latest_quotes(odds: pd.DataFrame, when: pd.Series) -> pd.DataFrame:
    """The latest quote of every (match, market, selection, book) at or before ``when[match_id]``."""
    t = odds["match_id"].map(when)
    ok = odds.loc[t.notna() & (odds["collected_at"] <= t)]
    ok = ok.sort_values("collected_at").drop_duplicates(["match_id", "market", "selection", "book"], keep="last")
    return ok.reset_index(drop=True)


def merge_bet365(quotes: pd.DataFrame) -> pd.DataFrame:
    """One bookmaker = one book: bet365 of OddsPapi and of Football-Data become ``bet365`` with the later quote of
    each selection (Football-Data's quote time is a nominal 00:00 of the match day)."""
    q = quotes.copy()
    q["book"] = q["book"].replace({"fd_bet365": "bet365"})
    q = q.sort_values("collected_at").drop_duplicates(["match_id", "market", "selection", "book"], keep="last")
    return q.reset_index(drop=True)


def fair_books(quotes: pd.DataFrame, method: str = "power") -> pd.DataFrame:
    """Complete books -> long table (match, market, book, selection) with the fair probability ``q``, the book's
    margin, its latest / earliest quote time and the time gap between its selections."""
    out = []
    for grp, sides in SIDES.items():
        g = quotes.loc[quotes["group"] == grp]
        if g.empty:
            continue
        w = g.pivot_table(index=["match_id", "market", "book"], columns="selection", values="odds", aggfunc="last")
        t = g.pivot_table(index=["match_id", "market", "book"], columns="selection", values="collected_at",
                          aggfunc="last")
        if not all(s in w.columns for s in sides):
            continue
        w, t = w[list(sides)], t.reindex(columns=list(sides))
        ok = w.notna().all(1)
        w, t = w.loc[ok], t.loc[ok]
        o = w.to_numpy(float)
        q = demargin(o, method)
        tt = t.apply(pd.to_datetime)
        tmax, tmin = tt.max(1), tt.min(1)
        margin = (1 / o).sum(1) - 1
        for j, s in enumerate(sides):
            out.append(pd.DataFrame({"match_id": w.index.get_level_values(0), "market": w.index.get_level_values(1),
                                     "book": w.index.get_level_values(2), "selection": s, "group": grp,
                                     "odds": o[:, j], "q": q[:, j], "margin": margin,
                                     "t_max": tmax.to_numpy(), "t_min": tmin.to_numpy()}))
    f = pd.concat(out, ignore_index=True)
    f["gap_h"] = (pd.to_datetime(f["t_max"]) - pd.to_datetime(f["t_min"])).dt.total_seconds() / 3600
    return f.loc[np.isfinite(f["q"])].reset_index(drop=True)


def dc_from_1x2(fair: pd.DataFrame) -> pd.DataFrame:
    """Double-chance fair probabilities derived from each complete 1X2 book."""
    x = fair.loc[fair["market"] == "1X2"]
    w = x.pivot_table(index=["match_id", "book"], columns="selection", values="q", aggfunc="first")
    meta = x.drop_duplicates(["match_id", "book"]).set_index(["match_id", "book"])[["margin", "t_max", "t_min",
                                                                                     "gap_h"]]
    w = w.join(meta)
    rows = []
    for sel, (a, b) in {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}.items():
        rows.append(pd.DataFrame({"match_id": w.index.get_level_values(0), "book": w.index.get_level_values(1),
                                  "market": "DOUBLE_CHANCE", "selection": sel, "group": "DOUBLE_CHANCE",
                                  "odds": np.nan, "q": (w[a] + w[b]).to_numpy(), "margin": w["margin"].to_numpy(),
                                  "t_max": w["t_max"].to_numpy(), "t_min": w["t_min"].to_numpy(),
                                  "gap_h": w["gap_h"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


SINGLE_BOOK_ORDER = ("bet365", "fd_bet365", "pinnacle", "williamhill")


def single_book(fair: pd.DataFrame, order=SINGLE_BOOK_ORDER) -> pd.DataFrame:
    """One market probability per (match, market, selection): the first complete book in a fixed order
    (bet365 OddsPapi — the production book — then Football-Data Bet365, Pinnacle, William Hill, any other)."""
    rank = {b: i for i, b in enumerate(order)}
    f = fair.assign(_r=fair["book"].map(rank).fillna(len(rank)))
    first = f.sort_values(["match_id", "market", "_r"]).drop_duplicates(["match_id", "market"])[
        ["match_id", "market", "book"]]
    return f.merge(first, on=["match_id", "market", "book"]).drop(columns="_r")


def attach_q(cands: pd.DataFrame, fair_one: pd.DataFrame, prefix: str = "q") -> pd.DataFrame:
    """Add the market probability of each candidate (``q``; for Asian handicaps the no-refund probability
    ``q`` and the comparable win probability ``q_win = q · (1 − P_model(refund))``)."""
    f = pd.concat([fair_one, dc_from_1x2(fair_one)], ignore_index=True)
    cols = {"q": prefix, "book": f"{prefix}_book", "margin": f"{prefix}_margin", "t_max": f"{prefix}_time",
            "gap_h": f"{prefix}_gap_h"}
    f = f.drop_duplicates(["match_id", "market", "selection"])[["match_id", "market", "selection", *cols]]
    out = cands.drop(columns=[c for c in cols.values() if c in cands], errors="ignore").merge(
        f.rename(columns=cols), on=["match_id", "market", "selection"], how="left")
    return out


# ------------------------------------------------------------------------------------------------ candidates
def load_candidates(universe: str) -> pd.DataFrame:
    c = pd.read_parquet(CACHE / f"candidates_{universe}.parquet")
    c = c.loc[c["group"] != "EXACT_SCORE"].reset_index(drop=True)
    c["probability"] = c["probability"].astype(float)
    c["odds"] = c["odds"].astype(float)
    c["push_probability"] = c["push_probability"].astype(float)
    c["p_model"] = c["probability"]
    c["month"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m")
    return c


def results() -> pd.DataFrame:
    m = pd.read_parquet(CACHE / "matches.parquet", columns=["match_id", "played", "home_goals", "away_goals",
                                                             "total_corners"])
    return m.loc[m["played"].astype(bool)].set_index("match_id")[["home_goals", "away_goals", "total_corners"]]


def baseline(name: str) -> pd.DataFrame:
    b = pd.read_parquet(CACHE / f"{name}.parquet")
    return b


def cond(p, push):
    return np.asarray(p, float) / np.clip(1 - np.asarray(push, float), 1e-6, None)


# ------------------------------------------------------------------------------------------------ market table
def market_table(c: pd.DataFrame, qcol: str = "q", pcol: str = "p_model") -> pd.DataFrame:
    """One row per (match, market) of a book group with a complete market probability: model and market
    probabilities of every side (handicap: no-refund scale), the index of the winning side (NaN: push / unknown /
    not played), month."""
    rows = []
    for grp, sides in SIDES.items():
        g = c.loc[(c["group"] == grp) & c[qcol].notna()]
        if g.empty:
            continue
        p = g[pcol].to_numpy(float)
        if grp == "HANDICAP":
            p = cond(p, g["push_probability"])
        g = g.assign(_p=p)
        wp = g.pivot_table(index=["match_id", "market"], columns="selection", values="_p", aggfunc="first")
        wq = g.pivot_table(index=["match_id", "market"], columns="selection", values=qcol, aggfunc="first")
        ws = g.pivot_table(index=["match_id", "market"], columns="selection", values="status", aggfunc="first")
        if not all(s in wq.columns for s in sides):
            continue
        ok = wq[list(sides)].notna().all(1) & wp[list(sides)].notna().all(1)
        wp, wq, ws = wp.loc[ok, list(sides)], wq.loc[ok, list(sides)], ws.loc[ok].reindex(columns=list(sides))
        won = (ws == "WON").to_numpy()
        y = np.where(won.sum(1) == 1, won.argmax(1), -1)
        P = wp.to_numpy(float)
        P = np.clip(P / P.sum(1, keepdims=True), 1e-6, 1)     # handicap scale / rounding: a proper distribution
        meta = g.drop_duplicates(["match_id", "market"]).set_index(["match_id", "market"])[["month", "league"]]
        meta = meta.reindex(wp.index)
        rows.append({"group": grp, "index": wp.index, "P": P, "Q": np.clip(wq.to_numpy(float), 1e-6, 1), "y": y,
                     "month": meta["month"].to_numpy(), "league": meta["league"].to_numpy(), "sides": sides})
    return {r["group"]: r for r in rows}


# ------------------------------------------------------------------------------------------------ pooling
def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pool_fit(logs: list[np.ndarray], y: np.ndarray, x0=None, ridge: float = 1e-3):
    """softmax(a + Σ_j b_j log X_j) by maximum likelihood (v6.2 ``hybrids.pool_fit`` with a tiny ridge on b
    towards the starting point for numerical stability). Returns theta = [a_2..a_k, b_1..b_m]."""
    k, m, n = logs[0].shape[1], len(logs), len(y)
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1.0
    L = np.stack(logs, 0)
    x0 = np.r_[np.zeros(k - 1), np.ones(m) / m] if x0 is None else np.asarray(x0, float)

    def f(theta):
        a = np.r_[0.0, theta[:k - 1]]
        z = a + np.tensordot(theta[k - 1:], L, axes=1)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        P = np.exp(z - lse[:, None])
        nll = -(z[np.arange(n), y] - lse).sum()
        R = P - Y
        g = np.r_[R[:, 1:].sum(0), (L * R[None]).sum((1, 2))]
        return nll + ridge * n * ((theta - x0) ** 2).sum(), g + 2 * ridge * n * (theta - x0)
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B")
    return res.x


def pool_predict(theta, logs):
    k = logs[0].shape[1]
    a = np.r_[0.0, theta[:k - 1]]
    return _softmax(a + sum(theta[k - 1 + j] * logs[j] for j in range(len(logs))))


def logs(P):
    return np.log(np.clip(P, 1e-6, 1.0))


def ll_rows(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


MIN_TRAIN = 300


def walk_forward_pool(tab: dict, kind: str = "blend", min_train: int = MIN_TRAIN) -> dict:
    """Monthly expanding window inside 2026/27: for the matches of month M the pool is fitted on rows of the
    earlier months with a known outcome. ``kind``: blend (log q, log p), market (log q only, i.e. a recalibrated
    market), model (log p only). Months without enough training rows keep the model probability (production)
    for ``blend`` / ``model`` and the raw market probability for ``market``. Returns group -> (n, k) array and the
    fitted coefficients by month."""
    out, coefs = {}, {}
    for grp, r in tab.items():
        P, Q, y, month = r["P"], r["Q"], r["y"], r["month"]
        res = (P if kind != "market" else Q).copy()
        coefs[grp] = {}
        for mth in MONTHS:
            test = month == mth
            train = (month < mth) & (y >= 0)
            if not test.any():
                continue
            if train.sum() < min_train:
                coefs[grp][mth] = None
                continue
            X = {"blend": [logs(Q[train]), logs(P[train])], "market": [logs(Q[train])],
                 "model": [logs(P[train])]}[kind]
            th = pool_fit(X, y[train])
            Xt = {"blend": [logs(Q[test]), logs(P[test])], "market": [logs(Q[test])], "model": [logs(P[test])]}[kind]
            res[test] = pool_predict(th, Xt)
            coefs[grp][mth] = {"theta": th.tolist(), "train_rows": int(train.sum())}
        out[grp] = res
    return out, coefs


def fixed_pool(tab: dict, weights: dict) -> dict:
    """Pool with fixed exponents: group -> (b_market, b_model) (intercepts 0). Groups absent keep the model."""
    out = {}
    for grp, r in tab.items():
        if grp in weights:
            bq, bp = weights[grp]
            out[grp] = _softmax(bq * logs(r["Q"]) + bp * logs(r["P"]))
        else:
            out[grp] = r["P"].copy()
    return out


def back_to_candidates(c: pd.DataFrame, tab: dict, probs: dict, col: str) -> pd.DataFrame:
    """Write pooled side probabilities back to the candidate rows (handicap back to the win scale with the model's
    refund probability; double chance from the pooled 1X2); rows without a pooled value keep the model's."""
    parts = []
    for grp, r in tab.items():
        idx, sides, P = r["index"], r["sides"], probs[grp]
        for j, s in enumerate(sides):
            parts.append(pd.DataFrame({"match_id": idx.get_level_values(0), "market": idx.get_level_values(1),
                                       "selection": s, "_v": P[:, j]}))
    v = pd.concat(parts, ignore_index=True)
    x12 = v.loc[v["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="_v")
    if len(x12):
        dc = pd.concat([pd.DataFrame({"match_id": x12.index, "market": "DOUBLE_CHANCE", "selection": s,
                                      "_v": (x12[a] + x12[b]).to_numpy()})
                        for s, (a, b) in {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}.items()])
        v = pd.concat([v, dc], ignore_index=True)
    out = c.drop(columns=[col], errors="ignore").merge(v, on=["match_id", "market", "selection"], how="left")
    hc = out["group"] == "HANDICAP"
    out.loc[hc, "_v"] = out.loc[hc, "_v"] * (1 - out.loc[hc, "push_probability"])
    out[col] = out["_v"].fillna(out["p_model"])
    out[f"{col}_has"] = out["_v"].notna()
    return out.drop(columns="_v")


# ------------------------------------------------------------------------------------------------ Main rule
def odds_pref_vec(odds: np.ndarray, preferred, softness) -> np.ndarray:
    low, high = preferred
    x = np.log(np.clip(odds, 1e-9, None))
    d = np.where(odds < low, math.log(low) - x, np.where(odds > high, x - math.log(high), 0.0))
    out = np.exp(-0.5 * (d / softness) ** 2) if softness > 0 else (d == 0).astype(float)
    return np.where(odds > 1.0, out, 0.0)


def rule_scores(c: pd.DataFrame, prob: np.ndarray, params: RuleParams) -> tuple[np.ndarray, np.ndarray]:
    odds = c["odds"].to_numpy(float)
    push = c["push_probability"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    ev = np.where(priced, prob * odds - (1 - push), np.nan)
    eff = np.where(priced, odds, np.maximum(1 - push, 1e-9) / np.maximum(prob, 1e-9))
    rel = c["group"].map(config.MARKET_RELIABILITY).fillna(0.8).to_numpy(float)
    base = rel * prob ** params.probability_power * odds_pref_vec(eff, params.preferred, params.softness)
    score = np.where(priced, base * (1 + params.ev_weight * np.clip(np.nan_to_num(ev), -EV_CLIP, EV_CLIP)),
                     base * params.unpriced_factor)
    return score, ev


def main_vec(c: pd.DataFrame, prob_col: str = "probability", params: RuleParams | None = None,
             extra_eligible: np.ndarray | None = None, rank_score: np.ndarray | None = None,
             method: str = "rule") -> pd.DataFrame:
    """Production's Main rule on the candidate table with any probability column (EV recomputed with it).
    ``extra_eligible`` removes candidates (e.g. a disagreement filter); ``rank_score`` replaces the score used to
    rank (eligibility unchanged). If no candidate is eligible the 1X2 selection with the highest probability is
    taken, as in production."""
    params = params or main_params()
    prob = c[prob_col].to_numpy(float)
    score, ev = rule_scores(c, prob, params)
    if rank_score is not None:
        score = np.asarray(rank_score, float)
    odds = c["odds"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    grp = c["group"].to_numpy()
    elig = ~np.isin(grp, ["EHANDICAP", "EXACT_SCORE"]) & (prob >= params.min_probability) & (
        prob <= params.max_probability)
    if extra_eligible is not None:
        elig &= np.asarray(extra_eligible, bool)
    d = c.assign(_s=score, _priced=priced.astype(int), _p=prob, _ev=ev, _e=elig)
    keys = ["match_id", "_s", "_priced", "_p"]
    order = [True, False, False, False]
    chosen = []
    if params.value_first_min_ev is not None:
        v = d.loc[d["_e"] & d["_priced"].astype(bool) & (d["_ev"] >= params.value_first_min_ev)]
        chosen.append(v.sort_values(keys, ascending=order).drop_duplicates("match_id"))
    e = d.loc[d["_e"]]
    chosen.append(e.sort_values(keys, ascending=order).drop_duplicates("match_id"))
    x = d.loc[d["market"] == "1X2"].sort_values(["match_id", "_p"], ascending=[True, False]).drop_duplicates(
        "match_id")
    chosen.append(x)
    out = pd.concat(chosen, ignore_index=True).drop_duplicates("match_id", keep="first")
    out = out.assign(probability=out["_p"], ev=out["_ev"], method=method)
    return out.drop(columns=["_s", "_priced", "_p", "_ev", "_e"]).reset_index(drop=True)


def self_check(universe: str) -> dict:
    """main_vec with production parameters must reproduce Baseline B (production engine) for every match."""
    c = load_candidates(universe)
    mine = main_vec(c)
    b = baseline(f"baseline_b_{universe}")
    j = mine.merge(b[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_b"))
    same = (j["market"] == j["market_b"]) & (j["selection"] == j["selection_b"])
    return {"matches": int(len(j)), "same_main_share": float(same.mean())}


# ------------------------------------------------------------------------------------------------ evaluation
def played(main: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    return main.loc[main["match_id"].isin(res.index)].reset_index(drop=True)


def evaluate_variant(main: pd.DataFrame, base: pd.DataFrame, res: pd.DataFrame) -> dict:
    p = played(main, res)
    return {"summary": EV.summary(p), "roi_ci": EV.roi_with_ci(p), "vs_b": EV.compare(p, played(base, res))}


def stability(main: pd.DataFrame, base: pd.DataFrame, res: pd.DataFrame) -> dict:
    """Months, leagues, top wins, odds sensitivity of a variant next to Baseline B."""
    p, b = played(main, res), played(base, res)
    p = p.assign(month=pd.to_datetime(p["date"]).dt.strftime("%Y-%m"))
    b = b.assign(month=pd.to_datetime(b["date"]).dt.strftime("%Y-%m"))
    mon = EV.breakdown(p, "month").merge(EV.breakdown(b, "month"), on="month", suffixes=("", "_b"))
    lg = EV.breakdown(p, "league").merge(EV.breakdown(b, "league"), on="league", suffixes=("", "_b"))
    lg["small"] = lg["bets"] < 50
    bets = p.loc[p["bet"] == 1].sort_values("profit", ascending=False)
    n = len(bets)
    out = {"months": mon.to_dict("records"), "leagues": lg.to_dict("records"),
           "leagues_better": int((lg["roi"] > lg["roi_b"]).sum()), "leagues_total": int(len(lg)),
           "months_better": int((mon["roi"] > mon["roi_b"]).sum()), "months_total": int(len(mon))}
    if n:
        for k in (5, 10):
            rest = bets.iloc[k:]
            out[f"roi_without_top{k}"] = float(rest["profit"].sum() / len(rest)) if len(rest) else None
        for h in (0.97, 0.95):           # every price 3 % / 5 % worse
            prof = np.where(bets["status"] == "WON", bets["odds"] * h - 1, -1.0)
            out[f"roi_odds_x{h}"] = float(prof.sum() / n)
    return out


def preregister(path, entries: list[dict]) -> dict:
    """Freeze B candidates for a forward check: an entry keeps its first ``frozen_at`` when the run is repeated."""
    old = C.read_json(path) if path.exists() else {"entries": []}
    known = {e["id"]: e for e in old.get("entries", [])}
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    out = []
    for e in entries:
        prev = known.get(e["id"])
        out.append({**e, "frozen_at": prev["frozen_at"] if prev else now})
    doc = {"entries": out, "note": "B candidates frozen for a forward check on matches played after frozen_at; "
                                   "production is not changed."}
    C.write_json(path, doc)
    return doc
