"""Does the market over-react to recent form?  A form correction of the de-margined Bet365 opening 1X2.

Model (multinomial logit with the market as offset, home = reference):

    z_H = log q_H,   z_k = log q_k + a_k + Σ_j β_kj x_j   (k = D, A),   p_k = exp z_k / Σ exp z

x = standardised form differences (home − away; standardisation and missing -> 0 from the training rows), fitted by
maximum likelihood with a small ridge (0.5 Σβ²) on seasons 2022/23 ... s-1 and applied to season s (walk-forward).

    MF0  intercepts only (re-calibration of the market; control)
    MF1  x = EW-8 points difference (diff_ew8_ppg)
    MF2  x = EW-8 points, Elo change over 10 matches, venue form (10), last-3 points differences

Evaluation on the same matches: Δ log loss vs the de-margined market (negative = better than the market);
Holm across the three variants on the pooled selection; value bets at the same Bet365 opening price where
p_k · odds_k > 1 (one bet per match: the largest EV), ROI with bootstrap interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.v6_2.leagues import wf

from . import common as C
from . import variants as V

SETS = {"MF0": [], "MF1": ["diff_ew8_ppg"], "MF2": ["diff_ew8_ppg", "diff_elo_d10", "diff_vf10_ppg", "diff_f3_ppg"]}
NAMES = {"MF0": "перекалибровка рынка (контроль)", "MF1": "рынок + разница EW-8 очков",
         "MF2": "рынок + 4 признака формы"}
RIDGE = 0.5


def rows(s: int) -> pd.DataFrame:
    fr = wf.frame(s)
    fr = fr.loc[fr["in_eval"]].reset_index(drop=True)
    fd = wf.fd_table(s).set_index("match_id")
    odds = fd.reindex(fr["match_id"])[["x_H_b365", "x_D_b365", "x_A_b365"]].to_numpy(float)
    out = fr[["match_id", "league", "season", "y", "qH", "qD", "qA"]].copy()
    out[["oH", "oD", "oA"]] = odds
    return out.merge(V.temporal_features(), on="match_id", how="left")


def _design(d: pd.DataFrame, cols, mu=None, sd=None):
    X = d[cols].to_numpy(float) if cols else np.zeros((len(d), 0))
    if mu is None:
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    X = np.nan_to_num((X - mu) / sd, nan=0.0)
    return np.column_stack([np.ones(len(d)), X]), mu, sd


def _probs(theta, X, lq):
    k = X.shape[1]
    bD, bA = theta[:k], theta[k:]
    z = np.column_stack([lq[:, 0], lq[:, 1] + X @ bD, lq[:, 2] + X @ bA])
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit(X, lq, y):
    k = X.shape[1]
    onehot = np.eye(3)[y]
    pen = np.r_[0.0, np.ones(k - 1)]

    def obj(th):
        p = _probs(th, X, lq)
        v = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum() + RIDGE * ((th[:k] ** 2 * pen).sum() + (th[k:] ** 2 * pen).sum())
        r = p - onehot
        g = np.r_[X.T @ r[:, 1] + 2 * RIDGE * th[:k] * pen, X.T @ r[:, 2] + 2 * RIDGE * th[k:] * pen]
        return v, g
    return minimize(obj, np.zeros(2 * k), jac=True, method="L-BFGS-B").x


def value_bets(p: np.ndarray, odds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Profit per match (NaN = no bet) of the largest-EV outcome with EV > 0 at the given prices."""
    ev = p * odds - 1
    k = np.nanargmax(np.where(np.isfinite(ev), ev, -9), 1)
    best = ev[np.arange(len(p)), k]
    won = (k == y)
    return np.where(best > 0, np.where(won, odds[np.arange(len(p)), k] - 1, -1.0), np.nan)


def run() -> dict:
    data = {s: rows(s) for s in range(C.FIRST_OOS, C.CURRENT + 1)}
    res = {"names": NAMES, "sets": SETS, "seasons": {}, "coefs": {}}
    per = {v: [] for v in SETS}
    for s in C.EVAL:
        tr = pd.concat([data[t] for t in range(C.FIRST_OOS, s)], ignore_index=True)
        te = data[s]
        lq_tr = np.log(tr[["qH", "qD", "qA"]].to_numpy(float))
        lq_te = np.log(te[["qH", "qD", "qA"]].to_numpy(float))
        y_te = te["y"].to_numpy(int)
        ll_mkt = C.logloss_rows(te[["qH", "qD", "qA"]].to_numpy(float), y_te)
        res["seasons"][str(s)] = {}
        for v, cols in SETS.items():
            Xtr, mu, sd = _design(tr, cols)
            th = fit(Xtr, lq_tr, tr["y"].to_numpy(int))
            Xte, _, _ = _design(te, cols, mu, sd)
            p = _probs(th, Xte, lq_te)
            ll = C.logloss_rows(p, y_te)
            prof = value_bets(p, te[["oH", "oD", "oA"]].to_numpy(float), y_te)
            per[v].append(pd.DataFrame({"season": s, "ll": ll, "ll_mkt": ll_mkt, "profit": prof}))
            res["seasons"][str(s)][v] = {"n": int(len(te)), "d_ll": C.paired_diff_ci(ll_mkt, ll),
                                         "roi": C.mean_ci(prof)}
            res["coefs"].setdefault(v, {})[str(s)] = {"theta": th.tolist(), "train_rows": int(len(tr))}
    pv = {}
    res["select"] = {}
    for v in SETS:
        d = pd.concat(per[v], ignore_index=True)
        sel = d.loc[d["season"].isin(C.SELECT)]
        res["select"][v] = {"d_ll": C.paired_diff_ci(sel["ll_mkt"], sel["ll"]), "roi": C.mean_ci(sel["profit"])}
        pv[v] = res["select"][v]["d_ll"]["p_two_sided"]
    adj = C.holm(pv)
    for v in SETS:
        res["select"][v]["p_holm"] = adj.get(v)
    C.save_json("market_form.json", res)
    return res
