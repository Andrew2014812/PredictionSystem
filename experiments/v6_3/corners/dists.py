"""Count distributions of total corners, their fitting and the probabilistic metrics.

Every distribution returns a pmf matrix (n matches x 41 values 0..40, the tail folded into the last cell, as
production's ``corners_distribution`` does), so the over / under probabilities are computed the same way.

    poisson     Pois(μ)
    nb          NB2(μ, r): Var = μ + μ²/r with one global size r (production: r by moments on validation)
    nb_league   NB2 with a size per league (shrunk to the global r)
    nbp         NB-P: Var = μ + α μ^p, i.e. a size per match r = μ^{2−p} / α
    cmp         Conway–Maxwell–Poisson with a global ν (ν > 1 under-, ν < 1 over-dispersed), mean = μ
    conv        home and away corners NB(μ_h, r_h) and NB(μ_a, r_a) joined by a Frank copula with parameter θ
                (θ = 0: independent); total = sum of the bivariate pmf along the anti-diagonals
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

K = 40
KS = np.arange(K + 1)


def _fold(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0, None)
    p[:, -1] += np.clip(1.0 - p.sum(1), 0, None)
    return p / p.sum(1, keepdims=True)


def pmf_poisson(mu) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 1e-3, None)
    return _fold(stats.poisson.pmf(KS[None, :], mu[:, None]))


def pmf_nb(mu, r) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 1e-3, None)
    r = np.broadcast_to(np.asarray(r, float), mu.shape)
    return _fold(stats.nbinom.pmf(KS[None, :], r[:, None], (r / (r + mu))[:, None]))


def pmf_nbp(mu, alpha, p) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 1e-3, None)
    return pmf_nb(mu, np.clip(mu ** (2 - p) / alpha, 0.5, 1e5))


# --------------------------------------------------------------------------------------- COM-Poisson
def _cmp_table(nu: float):
    """log λ grid -> mean of CMP(λ, ν) (on 0..120 for the normaliser)."""
    k = np.arange(121)
    loglam = np.linspace(-2, 12, 1400) if nu >= 1 else np.linspace(-3, 3.2, 1400)
    lp = loglam[:, None] * k[None, :] - nu * gammaln(k + 1)[None, :]
    lp -= lp.max(1, keepdims=True)
    w = np.exp(lp)
    mean = (w * k).sum(1) / w.sum(1)
    return loglam, mean


def pmf_cmp(mu, nu: float) -> np.ndarray:
    mu = np.clip(np.asarray(mu, float), 0.5, 30)
    loglam, mean = _cmp_table(nu)
    ll = np.interp(mu, mean, loglam)
    lp = ll[:, None] * KS[None, :] - nu * gammaln(KS + 1)[None, :]
    k = np.arange(121)
    lz = ll[:, None] * k[None, :] - nu * gammaln(k + 1)[None, :]
    m = lz.max(1, keepdims=True)
    logz = m[:, 0] + np.log(np.exp(lz - m).sum(1))
    return _fold(np.exp(lp - logz[:, None]))


# --------------------------------------------------------------------------------------- Frank copula
def _frank_c(u, v, theta):
    if abs(theta) < 1e-6:
        return u * v
    num = np.expm1(-theta * u) * np.expm1(-theta * v)
    return -np.log1p(num / np.expm1(-theta)) / theta


def pmf_conv(mu_h, mu_a, r_h, r_a, theta, chunk: int = 600) -> np.ndarray:
    """Total corners from a Frank-copula bivariate NB of (home, away)."""
    mu_h, mu_a = np.clip(np.asarray(mu_h, float), 0.05, None), np.clip(np.asarray(mu_a, float), 0.05, None)
    n = len(mu_h)
    out = np.zeros((n, K + 1))
    side = np.arange(K + 1)
    for s in range(0, n, chunk):
        mh, ma = mu_h[s:s + chunk], mu_a[s:s + chunk]
        F = stats.nbinom.cdf(side[None, :], r_h, (r_h / (r_h + mh))[:, None])
        G = stats.nbinom.cdf(side[None, :], r_a, (r_a / (r_a + ma))[:, None])
        F = np.concatenate([np.zeros((len(mh), 1)), F], 1)          # F(-1) = 0
        G = np.concatenate([np.zeros((len(ma), 1)), G], 1)
        Cm = _frank_c(F[:, :, None], G[:, None, :], theta)            # (m, K+2, K+2)
        joint = Cm[:, 1:, 1:] - Cm[:, :-1, 1:] - Cm[:, 1:, :-1] + Cm[:, :-1, :-1]
        joint = np.clip(joint, 0, None)
        tot = np.zeros((len(mh), 2 * K + 1))
        for i in range(K + 1):
            tot[:, i:i + K + 1] += joint[:, i, :]
        t = tot[:, :K + 1].copy()
        t[:, -1] += tot[:, K + 1:].sum(1)
        out[s:s + chunk] = t
    return _fold(out)


# --------------------------------------------------------------------------------------- fitting
def log_score(pmf: np.ndarray, y) -> np.ndarray:
    y = np.clip(np.asarray(y, int), 0, K)
    return -np.log(np.clip(pmf[np.arange(len(y)), y], 1e-12, None))


def nb_size_mom(y, mu) -> float | None:
    """Production's method of moments (src/modelling/training.negative_binomial_size)."""
    excess = np.mean((y - mu) ** 2 - mu)
    return None if excess <= 0 else float(np.mean(mu ** 2) / excess)


def fit_nb(y, mu) -> float:
    f = lambda lr: log_score(pmf_nb(mu, np.exp(lr)), y).mean()
    return float(np.exp(optimize.minimize_scalar(f, bounds=(0, 9), method="bounded").x))


def fit_nb_league(y, mu, league, r_global: float, shrink: float = 300.0) -> dict:
    """Per-league size by ML, shrunk: log r_L = (n log r̂_L + s log r) / (n + s)."""
    out = {}
    for L in np.unique(league):
        ok = league == L
        f = lambda lr: log_score(pmf_nb(mu[ok], np.exp(lr)), y[ok]).mean()
        r_l = np.exp(optimize.minimize_scalar(f, bounds=(0, 9), method="bounded").x)
        n = ok.sum()
        out[str(L)] = float(np.exp((n * np.log(r_l) + shrink * np.log(r_global)) / (n + shrink)))
    return out


def fit_nbp(y, mu) -> tuple[float, float]:
    def f(x):
        return log_score(pmf_nbp(mu, np.exp(x[0]), x[1]), y).mean()
    r = optimize.minimize(f, x0=[np.log(0.1), 1.0], method="Nelder-Mead",
                          options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 200})
    return float(np.exp(r.x[0])), float(r.x[1])


def fit_cmp(y, mu) -> float:
    f = lambda nu: log_score(pmf_cmp(mu, nu), y).mean()
    return float(optimize.minimize_scalar(f, bounds=(0.6, 1.4), method="bounded",
                                          options={"xatol": 1e-3}).x)


def fit_conv(y, mu_h, mu_a, independent: bool = False) -> dict:
    """r_h, r_a (and θ) maximising the log score of the TOTAL on the fitting rows."""
    def f(x):
        th = 0.0 if independent else x[2]
        return log_score(pmf_conv(mu_h, mu_a, np.exp(x[0]), np.exp(x[1]), th), y).mean()
    x0 = [np.log(20.0), np.log(20.0)] + ([] if independent else [-1.0])
    r = optimize.minimize(f, x0=x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 150})
    return {"r_h": float(np.exp(r.x[0])), "r_a": float(np.exp(r.x[1])),
            "theta": 0.0 if independent else float(r.x[2])}


def kendall_tau_frank(theta: float) -> float:
    """Kendall's τ of the Frank copula (Debye function), for reporting the dependence."""
    if abs(theta) < 1e-6:
        return 0.0
    from scipy import integrate
    d1 = integrate.quad(lambda t: t / np.expm1(t), 0, abs(theta))[0] / abs(theta)
    tau = 1 - 4 / abs(theta) * (1 - d1)
    return float(np.sign(theta) * tau)


# --------------------------------------------------------------------------------------- metrics
def p_over(pmf: np.ndarray, line: float) -> np.ndarray:
    return pmf[:, KS > line].sum(1)


def binary_ll(o, p) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def ece(o, p, bins: int = 10) -> float:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(o[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def reliability(o, p, bins: int = 10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return [{"p": float(p[idx == b].mean()), "freq": float(o[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).sum() > 0]


def rps(pmf: np.ndarray, y) -> np.ndarray:
    """Ranked probability score of the count (sum over k of (F(k) − 1{y ≤ k})²)."""
    y = np.asarray(y, int)
    F = np.cumsum(pmf, 1)
    ind = (KS[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2).sum(1)


def pit(pmf: np.ndarray, y, seed: int = 5) -> np.ndarray:
    """Randomised PIT values (uniform if the distribution is right)."""
    y = np.clip(np.asarray(y, int), 0, K)
    F = np.cumsum(pmf, 1)
    hi = F[np.arange(len(y)), y]
    lo = np.where(y > 0, F[np.arange(len(y)), np.maximum(y - 1, 0)], 0.0)
    u = np.random.default_rng(seed).uniform(size=len(y))
    return lo + u * (hi - lo)
