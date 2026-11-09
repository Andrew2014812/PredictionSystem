"""Production Main / Risk selection, vectorised over a long candidate table, and its bet statistics.

The rule is src/markets/recommendation.py with the production parameters (config.MAIN_* / RISK_*): the same
candidate sets, rule score, preferred-odds window, EV bonus at a real price, unpriced factor, Main fallback to the
most likely 1X2 outcome, and the Risk exclusion of selections that win / lose on exactly the same scores as Main
or exactly against it (footprints of src.markets.recommendation). ``self_check`` compares the vectorised picks
with the production functions on a sample of matches.

Candidate table columns: match_id, market, selection, group, p, p_push, odds (NaN without a real price), status
(1 won, 0 lost, 2 refund, -1 unknown).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import config
from src.markets.markets import handicap_market
from src.markets.odds import PricedSelection
from src.markets.recommendation import (EV_CLIP, main_params, main_prediction, opposite_outcome, risk_params,
                                        risk_prediction, same_outcome)

from . import common as C
from . import probs as PR

MP, RP = main_params(), risk_params()


# ------------------------------------------------------------------------------------------------ candidates
def _status(market, sel, hg, ag, tc) -> np.ndarray:
    hg, ag = np.asarray(hg, float), np.asarray(ag, float)
    m, t = hg - ag, hg + ag
    if market == "1X2":
        o = np.where(m > 0, "H", np.where(m < 0, "A", "D"))
        return (o == sel).astype(np.int8)
    if market == "DOUBLE_CHANCE":
        return {"1X": m >= 0, "X2": m <= 0, "12": m != 0}[sel].astype(np.int8)
    if market.startswith("TOTAL_"):
        line = float(market.split("_")[1])
        return ((t > line) == (sel == "OVER")).astype(np.int8)
    if market == "BTTS":
        return (((hg > 0) & (ag > 0)) == (sel == "YES")).astype(np.int8)
    if market == "EXACT_SCORE":
        h, a = (int(x) for x in sel.split("-"))
        return ((hg == h) & (ag == a)).astype(np.int8)
    if market.startswith("HANDICAP_"):
        line = float(market.split("_")[1])
        mm = m + line
        won = (mm > 0) if sel == "H" else (mm < 0)
        return np.where(mm == 0, 2, won.astype(int)).astype(np.int8)
    if market.startswith("CORNERS_"):
        line = float(market.split("_")[1])
        c = np.asarray(tc, float)
        return np.where(np.isnan(c), -1, ((c > line) == (sel == "OVER")).astype(int)).astype(np.int8)
    raise ValueError(market)


def candidates(meta: pd.DataFrame, book: dict, es_top: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Long table of every Main / Risk selection of production from a book.

    ``es_top``: five (score label array, probability array) pairs (production keeps the 5 most likely scores).
    """
    n = len(meta)
    z = np.zeros(n)
    cols: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for k, s in enumerate("HDA"):
        cols.append(("1X2", s, "1X2", book["1X2"][:, k], z))
    for s in ("1X", "X2", "12"):
        cols.append(("DOUBLE_CHANCE", s, "DOUBLE_CHANCE", book[f"DC_{s}"], z))
    for l in config.TOTAL_GOAL_LINES:
        o = book[PR.key_total(l)]
        cols += [(f"TOTAL_{l}", "OVER", "TOTAL", o, z), (f"TOTAL_{l}", "UNDER", "TOTAL", 1 - o, z)]
    cols += [("BTTS", "YES", "BTTS", book["BTTS"], z), ("BTTS", "NO", "BTTS", 1 - book["BTTS"], z)]
    for l in config.HANDICAP_LINES:
        mk = handicap_market(l)
        if l in PR.AH_HALF:
            h = book[PR.key_ah(l)]
            cols += [(mk, "H", "HANDICAP", h, z), (mk, "A", "HANDICAP", 1 - h, z)]
        else:
            w = book[PR.key_ahw(l)]
            cols += [(mk, "H", "HANDICAP", w[:, 0], w[:, 1]), (mk, "A", "HANDICAP", w[:, 2], w[:, 1])]
    hg, ag = meta["home_goals"].to_numpy(), meta["away_goals"].to_numpy()
    tc = meta["total_corners"].to_numpy(float)
    frames = []
    for mk, s, grp, p, push in cols:
        frames.append(pd.DataFrame({"mi": np.arange(n), "market": mk, "selection": s, "group": grp,
                                    "p": np.asarray(p, float), "p_push": np.asarray(push, float),
                                    "status": _status(mk, s, hg, ag, tc)}))
    for labels, p in es_top:
        st = np.array([int(f"{h}-{a}" == lab) for lab, h, a in zip(labels, hg.astype(int), ag.astype(int))],
                      dtype=np.int8)
        frames.append(pd.DataFrame({"mi": np.arange(n), "market": "EXACT_SCORE", "selection": labels,
                                    "group": "EXACT_SCORE", "p": np.asarray(p, float), "p_push": 0.0, "status": st}))
    for l in config.CORNER_LINES:                   # production order: goal markets, exact scores, corners
        o = book.get(f"CORNERS_{l}")
        if o is not None:
            for s, pr in (("OVER", o), ("UNDER", 1 - o)):
                frames.append(pd.DataFrame({"mi": np.arange(n), "market": f"CORNERS_{l}", "selection": s,
                                            "group": "CORNERS", "p": np.asarray(pr, float), "p_push": 0.0,
                                            "status": _status(f"CORNERS_{l}", s, hg, ag, tc)}))
    long = pd.concat(frames, ignore_index=True)
    long["order"] = np.arange(len(long))           # ties: the first selection in production order wins
    long = long.loc[np.isfinite(long["p"])]
    long.insert(0, "match_id", meta["match_id"].to_numpy()[long["mi"].to_numpy()])
    return long.reset_index(drop=True)


def top_scores(M: np.ndarray, k: int = config.EXACT_SCORE_TOP_N) -> list[tuple[np.ndarray, np.ndarray]]:
    flat = M.reshape(len(M), -1)
    idx = np.argsort(-flat, axis=1, kind="stable")[:, :k]
    g = M.shape[1]
    out = []
    for r in range(k):
        h, a = idx[:, r] // g, idx[:, r] % g
        out.append((np.array([f"{x}-{y}" for x, y in zip(h, a)]), flat[np.arange(len(M)), idx[:, r]]))
    return out


def attach_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    pr = prices.drop_duplicates(["match_id", "market", "selection"]).set_index(["match_id", "market", "selection"])
    key = pd.MultiIndex.from_frame(long[["match_id", "market", "selection"]])
    pos = pr.index.get_indexer(key)
    o = np.where(pos >= 0, pr["odds"].to_numpy(float)[np.maximum(pos, 0)], np.nan)
    long = long.copy()
    long["odds"] = np.where(o > 1, o, np.nan)
    return long


# ------------------------------------------------------------------------------------------------ the rule
def _pref(odds, preferred, softness) -> np.ndarray:
    lo, hi = preferred
    x = np.log(np.clip(odds, 1.0 + 1e-12, None))
    d = np.where(odds < lo, math.log(lo) - x, np.where(odds > hi, x - math.log(hi), 0.0))
    return np.where(odds > 1.0, np.exp(-0.5 * (d / softness) ** 2), 0.0)


def rule_score(long: pd.DataFrame, params) -> np.ndarray:
    p = long["p"].to_numpy(float)
    odds = long["odds"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1)
    eff = np.where(priced, odds, 1.0 / np.clip(p, 1e-9, None))
    rel = long["group"].map(lambda g: config.MARKET_RELIABILITY.get(g, 0.8)).to_numpy(float)
    base = rel * np.clip(p, 0, None) ** params.probability_power * _pref(eff, params.preferred, params.softness)
    ev = p * np.nan_to_num(odds) - (1 - long["p_push"].to_numpy(float))
    return np.where(priced, base * (1 + params.ev_weight * np.clip(ev, -EV_CLIP, EV_CLIP)),
                    base * params.unpriced_factor)


def _pick(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Row with the highest (score, has_odds, p) per match (production's max key)."""
    d = df.assign(_s=score, _h=np.isfinite(df["odds"].to_numpy(float)).astype(int), _o=-df["order"].to_numpy())
    d = d.sort_values(["match_id", "_s", "_h", "p", "_o"], kind="mergesort")
    return d.groupby("match_id", sort=False).tail(1).drop(columns=["_s", "_h", "_o"])


_REL: dict = {}


def _relation(a: tuple, b: tuple) -> int:
    """0 independent, 1 same outcome, 2 opposite outcome (production footprints)."""
    if (a, b) not in _REL:
        sa, sb = PricedSelection(a[0], a[1], 0.5), PricedSelection(b[0], b[1], 0.5)
        _REL[(a, b)] = 1 if same_outcome(sa, sb) else (2 if opposite_outcome(sa, sb) else 0)
    return _REL[(a, b)]


def main_risk(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = long.reset_index(drop=True)
    p = long["p"].to_numpy(float)
    grp = long["group"].to_numpy()
    s_main = rule_score(long, MP)
    cand = (~np.isin(grp, ["EHANDICAP", "EXACT_SCORE"])) & (p >= MP.min_probability) & (p <= MP.max_probability)
    main = _pick(long.loc[cand], s_main[cand])
    missing = sorted(set(long["match_id"]) - set(main["match_id"]))
    if missing:                                              # fallback: the most likely 1X2 outcome
        fb = long.loc[long["match_id"].isin(missing) & (long["market"] == "1X2")]
        fb = fb.sort_values(["match_id", "p"]).groupby("match_id").tail(1)
        main = pd.concat([main, fb])
    main = main.set_index("match_id")
    mk = main.reindex(long["match_id"])[["market", "selection"]]
    rel = np.array([_relation((a, b), (c, d)) for a, b, c, d in
                    zip(long["market"], long["selection"], mk["market"], mk["selection"])])
    elig = (grp != "EHANDICAP") & (rel == 0)
    s_risk = rule_score(long, RP)
    good = elig & (p >= RP.min_probability)
    has_good = pd.Series(good).groupby(long["match_id"]).transform("any").to_numpy()
    pool = good | (elig & ~has_good)
    risk = _pick(long.loc[pool], s_risk[pool]).set_index("match_id")
    return main.reset_index(), risk.reset_index()


def self_check(long: pd.DataFrame, n: int = 400, seed: int = 7) -> dict:
    """Share of sampled matches where the vectorised Main / Risk equal the production functions."""
    ids = pd.Series(long["match_id"].unique()).sample(min(n, long["match_id"].nunique()), random_state=seed)
    sub = long.loc[long["match_id"].isin(ids)]
    main, risk = main_risk(sub)
    mv = main.set_index("match_id")
    rv = risk.set_index("match_id")
    ok_m = ok_r = 0
    for mid, g in sub.groupby("match_id"):
        sels = [PricedSelection(r.market, r.selection, r.p, r.odds if r.odds == r.odds else None,
                                push_probability=r.p_push) for r in g.itertuples()]
        m = main_prediction(sels)
        r = risk_prediction(sels, m.pick if m else None)
        ok_m += int(m is not None and (m.pick.market, m.pick.selection) == (mv.at[mid, "market"], mv.at[mid, "selection"]))
        ok_r += int(r is not None and (r.pick.market, r.pick.selection) == (rv.at[mid, "market"], rv.at[mid, "selection"]))
    k = len(ids)
    return {"matches": int(k), "main_equal": ok_m / k, "risk_equal": ok_r / k}


# ------------------------------------------------------------------------------------------------ statistics
def profit(picks: pd.DataFrame) -> np.ndarray:
    """Per pick: odds - 1 / -1 for a priced decided bet; NaN for a refund, no price or unknown result."""
    o, st = picks["odds"].to_numpy(float), picks["status"].to_numpy(int)
    return np.where(np.isfinite(o) & (st == 1), o - 1, np.where(np.isfinite(o) & (st == 0), -1.0, np.nan))


def summary(picks: pd.DataFrame) -> dict:
    st = picks["status"].to_numpy(int)
    dec = (st == 0) | (st == 1)
    pr = profit(picks)
    ok = np.isfinite(pr)
    out = {"matches": int(len(picks)), "decided": int(dec.sum()),
           "hit_rate": float((st[dec] == 1).mean()) if dec.any() else np.nan,
           "mean_p": float(picks["p"].to_numpy(float)[dec].mean()) if dec.any() else np.nan,
           "priced_bets": int(ok.sum()), "priced_share": float(ok.sum() / max(dec.sum(), 1)),
           "avg_odds": float(picks["odds"].to_numpy(float)[ok].mean()) if ok.any() else np.nan,
           "roi": float(pr[ok].mean()) if ok.any() else np.nan}
    out["gap"] = out["mean_p"] - out["hit_rate"] if dec.any() else np.nan
    if ok.sum() >= 10:
        out["roi_ci95"] = C.ci(C.boot_stat(lambda a: np.nanmean(a, 1), pr[ok]))
    out["groups"] = picks["group"].value_counts().to_dict()
    return out


def paired_roi(ref: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Bootstrap over matches of ROI(new) - ROI(ref) (profit NaN = no priced decided bet)."""
    a = pd.Series(profit(ref), index=ref["match_id"].to_numpy())
    b = pd.Series(profit(new), index=new["match_id"].to_numpy())
    ids = a.index.union(b.index)
    a, b = a.reindex(ids).to_numpy(), b.reindex(ids).to_numpy()
    if np.isfinite(a).sum() < 10 or np.isfinite(b).sum() < 10:
        return {"diff": np.nan, "ci95": [np.nan, np.nan], "p": np.nan}
    dist = C.boot_stat(lambda x, y: np.nanmean(y, 1) - np.nanmean(x, 1), a, b)
    return {"diff": float(np.nanmean(b) - np.nanmean(a)), "ci95": C.ci(dist), "p": C.p_two_sided(dist),
            "changed_picks": int((ref.set_index("match_id")[["market", "selection"]].reindex(ids)
                                  != new.set_index("match_id")[["market", "selection"]].reindex(ids)).any(axis=1).sum())}


def paired_hit(ref: pd.DataFrame, new: pd.DataFrame) -> dict:
    def hits(d):
        st = d["status"].to_numpy(int)
        return pd.Series(np.where(st == 1, 1.0, np.where(st == 0, 0.0, np.nan)), index=d["match_id"].to_numpy())
    a, b = hits(ref), hits(new)
    ids = a.index.union(b.index)
    a, b = a.reindex(ids).to_numpy(), b.reindex(ids).to_numpy()
    dist = C.boot_stat(lambda x, y: np.nanmean(y, 1) - np.nanmean(x, 1), a, b, n_boot=1000)
    return {"diff": float(np.nanmean(b) - np.nanmean(a)), "ci95": C.ci(dist)}
