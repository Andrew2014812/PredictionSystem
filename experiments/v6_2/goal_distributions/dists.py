"""Score-matrix families with given goal means (vectorised, (n, 11, 11), MAX_GOALS = 10 as production).

All families take the XGBoost goal means lh (home) and la (away) and add at most two shape parameters, so
the comparison is between distributions, not means:

    poisson    independent Poisson (production, rho = 0; identical to src/markets/distributions.score_matrix)
    dc         Dixon-Coles tau on 0-0, 0-1, 1-0, 1-1 (rho)
    bp         bivariate Poisson (Karlis & Ntzoufras 2003): X = Y1 + Y3, Y = Y2 + Y3, Y3 ~ Pois(l3), the
               marginal means stay lh, la (Y1 ~ Pois(lh - l3), Y2 ~ Pois(la - l3)); only positive covariance
    nb         independent negative binomials NB(lh, r_h), NB(la, r_a) (over-dispersion; r -> inf = Poisson)
    cmp        independent Conway-Maxwell-Poisson with mean lh / la and one dispersion nu
               (nu > 1 under-dispersed, nu < 1 over-dispersed, nu = 1 Poisson)
    frank      Frank copula (theta) on Poisson marginals (negative and positive dependence)
    frank_cmp  Frank copula on CMP marginals (dispersion + dependence)
    zi         0-0 inflation: (1 - w) P + w * 1{0-0}  (only if the excess-zeros test justifies it)
    diag       diagonal inflation (Karlis & Ntzoufras 2003): (1 - w) P + w * draw distribution Pois-shaped

Marginal tails beyond 10 goals are folded into the 10-goal cell (as production); joint families are renormalised.
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats

from experiments.v6_research import scores as SC

G = SC.G
K = np.arange(G)


def _fold(pmf: np.ndarray) -> np.ndarray:
    pmf = np.clip(pmf, 0, None)
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
    return pmf


def _norm(m: np.ndarray) -> np.ndarray:
    return m / m.sum((1, 2), keepdims=True)


# ------------------------------------------------------------------------------------------------ marginals
def pois_marg(lam) -> np.ndarray:
    return SC.poisson_vectors(lam)


def nb_marg(mu, r) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    if not np.isfinite(r) or r > 1e6:
        return pois_marg(mu)
    p = r / (r + mu)
    return _fold(stats.nbinom.pmf(K[None, :], r, p[:, None]))


class CMPTable:
    """Mean-matched Conway-Maxwell-Poisson: for a dispersion nu, log(lambda) as a function of the mean."""
    SUPPORT = 40

    def __init__(self, nu: float):
        self.nu = float(nu)
        x = np.arange(self.SUPPORT + 1)
        self.loglam = np.linspace(-9.0, 5.0, 2400)
        lw = self.loglam[:, None] * x[None, :] - self.nu * special.gammaln(x + 1)[None, :]
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        self.means = (w * x).sum(1) / w.sum(1)
        order = np.argsort(self.means)
        self.means, self.loglam = self.means[order], self.loglam[order]

    def pmf(self, mu) -> np.ndarray:
        mu = np.clip(np.asarray(mu, float), 0.03, 6.0)
        ll = np.interp(mu, self.means, self.loglam)
        x = np.arange(self.SUPPORT + 1)
        lw = ll[:, None] * x[None, :] - self.nu * special.gammaln(x + 1)[None, :]
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(1, keepdims=True)
        out = w[:, :G].copy()
        out[:, -1] += w[:, G:].sum(1)
        return out


_CMP_CACHE: dict = {}


def cmp_marg(mu, nu: float) -> np.ndarray:
    if abs(nu - 1) < 1e-9:
        return pois_marg(mu)
    key = round(float(nu), 6)
    if key not in _CMP_CACHE:
        if len(_CMP_CACHE) > 64:
            _CMP_CACHE.clear()
        _CMP_CACHE[key] = CMPTable(key)
    return _CMP_CACHE[key].pmf(mu)


# ------------------------------------------------------------------------------------------------ joint families
def poisson(lh, la) -> np.ndarray:
    return SC.matrices(lh, la)


def dc(lh, la, rho: float) -> np.ndarray:
    return SC.matrices(lh, la, rho=rho)


def independent(ph: np.ndarray, pa: np.ndarray) -> np.ndarray:
    return _norm(ph[:, :, None] * pa[:, None, :])


def bp(lh, la, l3: float) -> np.ndarray:
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    if l3 <= 1e-9:
        return poisson(lh, la)
    l3v = np.minimum(l3, 0.95 * np.minimum(lh, la))
    p1 = stats.poisson.pmf(K[None, :], np.clip(lh - l3v, 1e-6, None)[:, None])
    p2 = stats.poisson.pmf(K[None, :], np.clip(la - l3v, 1e-6, None)[:, None])
    p3 = stats.poisson.pmf(K[None, :], np.clip(l3v, 1e-12, None)[:, None])
    m = np.zeros((len(lh), G, G))
    for k in range(G):
        # P(X = x, Y = y) += P(Y1 = x - k) P(Y2 = y - k) P(Y3 = k)
        a = np.zeros((len(lh), G))
        b = np.zeros((len(lh), G))
        a[:, k:] = p1[:, :G - k]
        b[:, k:] = p2[:, :G - k]
        m += p3[:, k][:, None, None] * a[:, :, None] * b[:, None, :]
    return _norm(m)


def _frank_C(u, v, theta):
    if abs(theta) < 1e-8:
        return u * v
    num = np.expm1(-theta * u) * np.expm1(-theta * v)
    return -np.log1p(num / np.expm1(-theta)) / theta


def copula_frank(ph: np.ndarray, pa: np.ndarray, theta: float) -> np.ndarray:
    """Joint pmf from marginal pmfs (n, G) by the Frank copula on the CDFs (rectangle differences)."""
    Fh = np.concatenate([np.zeros((len(ph), 1)), np.cumsum(ph, 1)], 1)
    Fa = np.concatenate([np.zeros((len(pa), 1)), np.cumsum(pa, 1)], 1)
    Fh[:, -1] = 1.0
    Fa[:, -1] = 1.0
    Cm = _frank_C(Fh[:, :, None], Fa[:, None, :], theta)          # (n, G+1, G+1)
    m = Cm[:, 1:, 1:] - Cm[:, :-1, 1:] - Cm[:, 1:, :-1] + Cm[:, :-1, :-1]
    return _norm(np.clip(m, 1e-15, None))


def frank(lh, la, theta: float) -> np.ndarray:
    return copula_frank(pois_marg(lh), pois_marg(la), theta)


def frank_cmp(lh, la, theta: float, nu: float) -> np.ndarray:
    return copula_frank(cmp_marg(lh, nu), cmp_marg(la, nu), theta)


def nb(lh, la, rh: float, ra: float) -> np.ndarray:
    return independent(nb_marg(lh, rh), nb_marg(la, ra))


def cmp(lh, la, nu: float) -> np.ndarray:
    return independent(cmp_marg(lh, nu), cmp_marg(la, nu))


def zero_inflate(m: np.ndarray, w: float) -> np.ndarray:
    out = (1 - w) * m
    out[:, 0, 0] += w
    return out


def diag_inflate(m: np.ndarray, w: float, lam_d: float = 1.0) -> np.ndarray:
    """(1 - w) P + w D, D on the diagonal with Poisson(lam_d) weights over the draw score."""
    d = stats.poisson.pmf(K, lam_d)
    d /= d.sum()
    out = (1 - w) * m
    out[:, K, K] += w * d[None, :]
    return out


def score_loglik(m: np.ndarray, hg, ag) -> np.ndarray:
    h = np.minimum(np.asarray(hg, int), G - 1)
    a = np.minimum(np.asarray(ag, int), G - 1)
    return np.log(np.clip(m[np.arange(len(m)), h, a], 1e-15, 1))
