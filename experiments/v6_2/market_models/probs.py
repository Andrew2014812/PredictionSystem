"""Market probabilities ("books") from a score matrix, from the direct models, and their outcomes / losses.

A book is a dict key -> array over matches:

    1X2            (n, 3)  P(home win), P(draw), P(away win)
    DC_1X, DC_X2, DC_12     P(selection wins)
    TOTAL_<l>      P(over l), l in TOTAL_LINES
    BTTS           P(both teams score)
    AH_<l>         Asian half line l (home perspective, l in +-0.5, +-1.5, +-2.5): P(home side wins)
    AHW_<l>        Asian whole line l (+-1, +-2): (n, 3) P(home side wins), P(refund), P(away side wins)
    ES17           (n, 17) exact score h, a <= 3 (index 4h + a) and "other"
    TOT7, GD7      (n, 7) total goals 0..5, 6+ / goal difference <=-3 .. >=3
    SCORE          P(the actual score) — full matrices only (log score of the whole distribution)
    CORNERS_<l>    P(over l corners), l in 8.5, 9.5, 10.5

Lines and selections are the production ones (src/config.py: TOTAL_GOAL_LINES, HANDICAP_LINES, CORNER_LINES);
the goal-distribution study adds totals 0.5, 4.5, 5.5.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from experiments.v6_research import scores as SC

from . import common as C

G = SC.G
I, J = SC._I, SC._J
TOTAL_LINES = (1.5, 2.5, 3.5)
ALL_TOTAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
AH_HALF = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
AH_WHOLE = (-2.0, -1.0, 1.0, 2.0)
CORNER_LINES = (8.5, 9.5, 10.5)


def key_total(l):
    return f"TOTAL_{l}"


def key_ah(l):
    return f"AH_{l:+g}"


def key_ahw(l):
    return f"AHW_{l:+g}"


# ------------------------------------------------------------------------------------------------ from a matrix
def from_matrix(M: np.ndarray, hg=None, ag=None, total_lines=TOTAL_LINES) -> dict:
    """Book of every goal market from (n, G, G) score matrices."""
    M = np.asarray(M, float)
    tot = I + J
    mar = I - J
    ph, pd_, pa = (M * (I > J)).sum((1, 2)), (M * (I == J)).sum((1, 2)), (M * (I < J)).sum((1, 2))
    b = {"1X2": np.stack([ph, pd_, pa], 1), "DC_1X": ph + pd_, "DC_X2": pd_ + pa, "DC_12": ph + pa}
    for l in total_lines:
        b[key_total(l)] = (M * (tot > l)).sum((1, 2))
    b["BTTS"] = (M * ((I > 0) & (J > 0))).sum((1, 2))
    for l in AH_HALF:
        b[key_ah(l)] = (M * (mar + l > 0)).sum((1, 2))
    for l in AH_WHOLE:
        b[key_ahw(l)] = np.stack([(M * (mar + l > 0)).sum((1, 2)), (M * (mar + l == 0)).sum((1, 2)),
                                  (M * (mar + l < 0)).sum((1, 2))], 1)
    es = np.zeros((len(M), 17))
    for h in range(4):
        for a in range(4):
            es[:, 4 * h + a] = M[:, h, a]
    es[:, 16] = 1 - es[:, :16].sum(1)
    b["ES17"] = np.clip(es, 0, 1)
    b["TOT7"] = np.stack([(M * (tot == k)).sum((1, 2)) for k in range(6)] + [(M * (tot >= 6)).sum((1, 2))], 1)
    b["GD7"] = np.stack([(M * (mar <= -3)).sum((1, 2))] + [(M * (mar == k)).sum((1, 2)) for k in (-2, -1, 0, 1, 2)]
                        + [(M * (mar >= 3)).sum((1, 2))], 1)
    if hg is not None:
        h = np.minimum(np.asarray(hg, int), G - 1)
        a = np.minimum(np.asarray(ag, int), G - 1)
        b["SCORE"] = M[np.arange(len(M)), h, a]
    return b


def from_gd(P: np.ndarray) -> dict:
    """1X2, double chance and every Asian line from a 7-class goal-difference distribution."""
    P = np.asarray(P, float)
    m = np.array([-3, -2, -1, 0, 1, 2, 3])
    ph, pd_, pa = P[:, m > 0].sum(1), P[:, m == 0].sum(1), P[:, m < 0].sum(1)
    b = {"1X2": np.stack([ph, pd_, pa], 1), "DC_1X": ph + pd_, "DC_X2": pd_ + pa, "DC_12": ph + pa, "GD7": P}
    for l in AH_HALF:
        b[key_ah(l)] = P[:, m + l > 0].sum(1)
    for l in AH_WHOLE:
        b[key_ahw(l)] = np.stack([P[:, m + l > 0].sum(1), P[:, m + l == 0].sum(1), P[:, m + l < 0].sum(1)], 1)
    return b


def from_tot(P: np.ndarray, total_lines=TOTAL_LINES) -> dict:
    P = np.asarray(P, float)
    k = np.arange(7)
    b = {"TOT7": P}
    for l in total_lines:
        b[key_total(l)] = P[:, k > l].sum(1)
    return b


def corners_from_nb(mu: np.ndarray, nb_size) -> dict:
    """Production corner markets: NB(mu, r) (Poisson without over-dispersion), 41 cells, tail folded."""
    mu = np.asarray(mu, float)
    k = np.arange(41)
    out = {}
    ok = np.isfinite(mu)
    pm = np.full((len(mu), 41), np.nan)
    if ok.any():
        if nb_size:
            p = nb_size / (nb_size + mu[ok])
            pmf = stats.nbinom.pmf(k[None, :], nb_size, p[:, None])
        else:
            pmf = stats.poisson.pmf(k[None, :], mu[ok][:, None])
        pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
        pm[ok] = pmf
    for l in CORNER_LINES:
        v = (pm * (k > l)).sum(1)
        v[~ok] = np.nan
        out[f"CORNERS_{l}"] = v
    return out


def production_matrices(base) -> np.ndarray:
    """Exactly the production score matrices: independent Poisson (rho = 0), reconciled with the 1X2 model."""
    return SC.reconcile(SC.matrices(base["lh"].to_numpy(), base["la"].to_numpy()),
                        base[["pH", "pD", "pA"]].to_numpy())


# ------------------------------------------------------------------------------------------------ outcomes
def outcomes(hg, ag, tc=None, total_lines=TOTAL_LINES) -> dict:
    hg, ag = np.asarray(hg, int), np.asarray(ag, int)
    t, m = hg + ag, hg - ag
    o = {"1X2": np.where(m > 0, 0, np.where(m == 0, 1, 2)), "DC_1X": (m >= 0).astype(float),
         "DC_X2": (m <= 0).astype(float), "DC_12": (m != 0).astype(float), "BTTS": ((hg > 0) & (ag > 0)).astype(float)}
    for l in total_lines:
        o[key_total(l)] = (t > l).astype(float)
    for l in AH_HALF:
        o[key_ah(l)] = (m + l > 0).astype(float)
    for l in AH_WHOLE:
        o[key_ahw(l)] = np.where(m + l > 0, 0, np.where(m + l == 0, 1, 2))
    es = np.full(len(hg), 16)
    small = (hg <= 3) & (ag <= 3)
    es[small] = hg[small] * 4 + ag[small]
    o["ES17"], o["TOT7"], o["GD7"] = es, np.minimum(t, 6), np.clip(m, -3, 3) + 3
    if tc is not None:
        tc = np.asarray(tc, float)
        for l in CORNER_LINES:
            o[f"CORNERS_{l}"] = np.where(np.isnan(tc), np.nan, (tc > l).astype(float))
    return o


MULTI = {"1X2", "ES17", "TOT7", "GD7"}


def key_loss(key: str, book: dict, out: dict) -> np.ndarray:
    """Per-match log loss of one market key (NaN where undefined)."""
    if key == "SCORE":
        return -np.log(np.clip(book["SCORE"], 1e-12, 1))
    p, o = book[key], out[key]
    if key in MULTI or key.startswith("AHW_"):
        return C.multi_ll(o, p)
    ll = C.bin_ll(np.nan_to_num(o), p)
    return np.where(np.isfinite(o) & np.isfinite(np.asarray(p, float)), ll, np.nan)


def key_brier(key: str, book: dict, out: dict) -> np.ndarray:
    p, o = book[key], out[key]
    if key in MULTI or key.startswith("AHW_"):
        return C.multi_brier(o, p)
    return np.where(np.isfinite(o), C.bin_brier(np.nan_to_num(o), p), np.nan)


# market groups of the study and their per-match primary loss (mean log loss over the lines of the group)
GROUP_KEYS = {
    "1X2": ["1X2"],
    "DOUBLE_CHANCE": ["DC_1X", "DC_X2", "DC_12"],
    "TOTAL": [key_total(l) for l in TOTAL_LINES],
    "BTTS": ["BTTS"],
    "HANDICAP": [key_ah(l) for l in AH_HALF] + [key_ahw(l) for l in AH_WHOLE],
    "EXACT_SCORE": ["ES17"],
    "CORNERS": [f"CORNERS_{l}" for l in CORNER_LINES],
}


def group_loss(group: str, book: dict, out: dict) -> np.ndarray:
    return np.nanmean(np.stack([key_loss(k, book, out) for k in GROUP_KEYS[group]], 1), 1)


# ------------------------------------------------------------------------------------------------ pooling
def log_pool(A: np.ndarray, B: np.ndarray, a: float, b: float) -> np.ndarray:
    """Normalised log-linear pool p ∝ A^a B^b over classes (last axis)."""
    la, lb = np.log(np.clip(A, 1e-12, 1)), np.log(np.clip(B, 1e-12, 1))
    z = a * la + b * lb
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_log_pool(A, B, y) -> tuple[float, float]:
    """Weights of the log-linear pool by maximum likelihood (Nelder-Mead from (1, 0))."""
    from scipy.optimize import minimize
    y = np.asarray(y, int)

    def nll(w):
        return C.multi_ll(y, log_pool(A, B, w[0], w[1])).mean()

    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return float(r.x[0]), float(r.x[1])


def fit_stack_binary(p_der, p_dir, o) -> tuple[float, float, float]:
    """Logistic regression on (logit derived, logit direct) with an intercept (practically unpenalised)."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([C.logit(p_der), C.logit(p_dir)])
    ok = np.isfinite(X).all(1) & np.isfinite(o)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(X[ok], np.asarray(o)[ok].astype(int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0]), float(lr.coef_[0, 1])


def apply_stack_binary(p_der, p_dir, w) -> np.ndarray:
    return C.sigmoid(w[0] + w[1] * C.logit(p_der) + w[2] * C.logit(p_dir))
