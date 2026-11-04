"""Count distributions of total corners and their scores.

Production: NB(mu, r) with one size r per model, r by the method of moments on validation
residuals (``src.modelling.training.negative_binomial_size``), lines read from the pmf exactly as
``src.markets.distributions.corners_distribution`` / ``prob_over`` do (checked in run.py).

Variants compared here (all fitted on the INNER season only):
    poisson      r = inf
    nb_mom       production: one r, method of moments
    nb_mle       one r, maximum likelihood
    nb_league    one r per league (maximum likelihood, shrunk to the global r)
    nb1          Var = phi * mu  (NB with r_i = mu_i / (phi - 1))
    conv_indep   home ~ NB(mu_h, r_h), away ~ NB(mu_a, r_a) independent, total = convolution
    nb_joint     NB of the total with Var = Var_h + Var_a + 2 rho sd_h sd_a (rho = residual
                 correlation of home and away corners on the inner season)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.modelling.training import negative_binomial_size

from .models import nb_loglik

MAXC = 40
K = np.arange(MAXC + 1)


def pmf_nb(mu, r) -> np.ndarray:
    """(n, MAXC+1) pmf, tail folded into the last cell (as corners_distribution)."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    if r is None:
        pmf = stats.poisson.pmf(K[None, :], mu)
    else:
        r = np.broadcast_to(np.asarray(r, float), (mu.shape[0],))[:, None]
        pmf = stats.nbinom.pmf(K[None, :], r, r / (r + mu))
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
    return pmf


def convolve(ph: np.ndarray, pa: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ph)
    for i in range(MAXC + 1):
        out[:, i:] += ph[:, [i]] * pa[:, :MAXC + 1 - i]
    out[:, -1] += np.clip(1 - out.sum(1), 0, None)
    return out


# ---------------------------------------------------------------------------------------------- dispersion fits
def r_mom(y, mu):
    return negative_binomial_size(np.asarray(y, float), np.asarray(mu, float))


def r_mle(y, mu) -> float | None:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    f = lambda lr: -nb_loglik(y, mu, np.exp(lr)).mean()  # noqa: E731
    res = optimize.minimize_scalar(f, bounds=(np.log(1.0), np.log(5000.0)), method="bounded")
    pois = -stats.poisson.logpmf(y, np.clip(mu, 1e-9, None)).mean()
    return None if pois <= res.fun + 1e-9 else float(np.exp(res.x))


def r_by_league(y, mu, league, shrink: float = 400.0) -> dict:
    """Per-league ML size, shrunk on the 1/r scale towards the global value with ``shrink`` matches."""
    y, mu, league = np.asarray(y, float), np.asarray(mu, float), np.asarray(league)
    g = r_mle(y, mu)
    inv_g = 0.0 if g is None else 1.0 / g
    out = {"__global__": g}
    for lg in np.unique(league):
        m = league == lg
        rl = r_mle(y[m], mu[m])
        inv_l = 0.0 if rl is None else 1.0 / rl
        w = m.sum() / (m.sum() + shrink)
        inv = w * inv_l + (1 - w) * inv_g
        out[str(lg)] = None if inv <= 1e-6 else float(1.0 / inv)
    return out


def phi_nb1(y, mu) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(np.mean((y - mu) ** 2) / np.mean(mu))


class Dispersion:
    """One fitted dispersion rule (from inner-season y / mu) that turns means into pmfs."""

    def __init__(self, kind: str):
        self.kind = kind
        self.info: dict = {}

    def fit(self, y, mu, league=None, yh=None, mh=None, ya=None, ma=None):
        k = self.kind
        if k == "nb_mom":
            self.info["r"] = r_mom(y, mu)
        elif k == "nb_mle":
            self.info["r"] = r_mle(y, mu)
        elif k == "nb_league":
            self.info["r_league"] = r_by_league(y, mu, league)
        elif k == "nb1":
            self.info["phi"] = phi_nb1(y, mu)
        elif k in ("conv_indep", "nb_joint"):
            self.info["r_home"], self.info["r_away"] = r_mle(yh, mh), r_mle(ya, ma)
            vh = mh + (0 if self.info["r_home"] is None else mh ** 2 / self.info["r_home"])
            va = ma + (0 if self.info["r_away"] is None else ma ** 2 / self.info["r_away"])
            zh, za = (yh - mh) / np.sqrt(vh), (ya - ma) / np.sqrt(va)
            self.info["rho"] = float(np.mean(zh * za))
        return self

    def pmf(self, mu, league=None, mh=None, ma=None) -> np.ndarray:
        k = self.kind
        if k == "poisson":
            return pmf_nb(mu, None)
        if k in ("nb_mom", "nb_mle"):
            return pmf_nb(mu, self.info["r"])
        if k == "nb_league":
            rl = self.info["r_league"]
            g = rl["__global__"]
            r = np.array([rl.get(str(x), g) or 1e6 for x in league], float)
            return pmf_nb(mu, r)
        if k == "nb1":
            phi = self.info["phi"]
            if phi <= 1.0001:
                return pmf_nb(mu, None)
            return pmf_nb(mu, np.clip(np.asarray(mu, float), 1e-6, None) / (phi - 1))
        rh, ra = self.info["r_home"], self.info["r_away"]
        if k == "conv_indep":
            return convolve(pmf_nb(mh, rh), pmf_nb(ma, ra))
        if k == "nb_joint":
            vh = mh + (0 if rh is None else mh ** 2 / rh)
            va = ma + (0 if ra is None else ma ** 2 / ra)
            m = mh + ma
            var = vh + va + 2 * self.info["rho"] * np.sqrt(vh * va)
            excess = var - m
            r = np.where(excess > 1e-6, m ** 2 / np.maximum(excess, 1e-6), 1e6)
            return pmf_nb(m, r)
        raise ValueError(k)


# ---------------------------------------------------------------------------------------------- scores
def log_score(pmf, y) -> np.ndarray:
    y = np.clip(np.asarray(y, int), 0, MAXC)
    return -np.log(np.clip(pmf[np.arange(len(y)), y], 1e-12, None))


def crps(pmf, y) -> np.ndarray:
    """Discrete CRPS (= ranked probability score over 0..MAXC)."""
    y = np.asarray(y, int)
    F = np.cumsum(pmf, 1)
    ind = (K[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2).sum(1)


def pit(pmf, y, seed: int = 5) -> np.ndarray:
    """Randomised PIT values (uniform if the forecast distribution is right)."""
    y = np.clip(np.asarray(y, int), 0, MAXC)
    F = np.cumsum(pmf, 1)
    lo = np.where(y > 0, F[np.arange(len(y)), np.maximum(y - 1, 0)], 0.0)
    hi = F[np.arange(len(y)), y]
    u = np.random.default_rng(seed).random(len(y))
    return lo + u * (hi - lo)


def pit_summary(u, bins: int = 10) -> dict:
    h = np.histogram(u, bins=bins, range=(0, 1))[0] / len(u)
    chi2 = float(len(u) * np.sum((h - 1 / bins) ** 2 / (1 / bins)))
    return {"hist": h.tolist(), "max_abs_dev": float(np.max(np.abs(h - 1 / bins))), "chi2": chi2,
            "chi2_p": float(stats.chi2.sf(chi2, bins - 1)), "mean": float(np.mean(u)), "var": float(np.var(u))}


def p_over(pmf, line: float) -> np.ndarray:
    return pmf[:, K > line].sum(1)


def ece(o, p, bins: int = 10) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(o[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(o, p, bins: int = 10, min_n: int = 20) -> list[dict]:
    o, p = np.asarray(o, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"predicted": float(p[idx == b].mean()), "observed": float(o[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).sum() >= min_n]


def binary_ll(o, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    o = np.asarray(o, float)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def platt(o_in, p_in):
    """Logistic recalibration a + b * logit(p) fitted on the inner season; returns a function."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p_in, 1e-6, 1 - 1e-6) / np.clip(1 - p_in, 1e-6, 1))[:, None]
    lr = LogisticRegression(C=1e6).fit(z, np.asarray(o_in, int))

    def f(p):
        zz = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
        return lr.predict_proba(zz)[:, 1]
    f.coef = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return f


def bootstrap_mean(d, n: int = 2000, seed: int = 7) -> dict:
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def season_phase(dates: pd.Series) -> np.ndarray:
    m = pd.to_datetime(dates).dt.month.to_numpy()
    return np.select([np.isin(m, [7, 8, 9]), np.isin(m, [10, 11, 12]), np.isin(m, [1, 2, 3])],
                     ["1_Jul-Sep", "2_Oct-Dec", "3_Jan-Mar"], "4_Apr-Jun")
