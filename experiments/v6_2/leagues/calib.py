"""Post-hoc 1X2 calibration maps: global, per league, per group, with empirical-Bayes shrinkage.

Map (multinomial "vector scaling" on log-probabilities, home as reference class):

    z_k = b · log p_k + a_k,   a_H = 0,   p'_k = exp(z_k) / Σ_j exp(z_j)          θ = (b, a_D, a_A)

Fitted by maximum likelihood on walk-forward predictions of earlier seasons only (2022/23 ... s-1 for season s).

Partial pooling (empirical Bayes). For a unit u (league or group) with MLE θ̂_u and covariance V_u (inverse
observed information), the between-unit variance of each parameter is estimated by moments,

    τ²_j = max(0, mean_u (θ̂_uj − θ_gj)² − mean_u V_u,jj),

and the unit's map is the posterior mode with the Gaussian prior θ_u ~ N(θ_g, diag τ²):

    θ_u = argmin  NLL_u(θ) + ½ Σ_j (θ_j − θ_gj)² / τ²_j.

τ² = 0 gives the global map, τ² → ∞ the unpooled per-unit MLE. The "random-effects" variant pools only the
two intercepts (league-specific draw / away offsets) and keeps the global slope b.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

TAU_FLOOR = 1e-6


def apply(p: np.ndarray, theta) -> np.ndarray:
    b, ad, aa = theta
    lp = np.log(np.clip(np.asarray(p, float), 1e-12, 1.0))
    z = b * lp + np.array([0.0, ad, aa])
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(theta, p, y) -> float:
    q = apply(p, theta)
    return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).sum())


def _grad(theta, p, y):
    """Analytic gradient of the NLL."""
    q = apply(p, theta)
    lp = np.log(np.clip(p, 1e-12, 1.0))
    onehot = np.zeros_like(q)
    onehot[np.arange(len(y)), y] = 1
    r = q - onehot                                    # dNLL/dz
    gb = float((r * lp).sum())
    return np.array([gb, float(r[:, 1].sum()), float(r[:, 2].sum())])


def fit(p, y, prior_mean=None, prior_var=None, free=(True, True, True)) -> np.ndarray:
    """MLE (prior None) or posterior mode under N(prior_mean, diag prior_var); ``free`` False fixes a parameter
    at prior_mean."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    m0 = np.array([1.0, 0.0, 0.0]) if prior_mean is None else np.asarray(prior_mean, float)
    free = np.asarray(free, bool)

    def full(x):
        t = m0.copy()
        t[free] = x
        return t

    def obj(x):
        t = full(x)
        v = nll(t, p, y)
        g = _grad(t, p, y)
        if prior_var is not None:
            pv = np.maximum(np.asarray(prior_var, float), TAU_FLOOR)
            v += 0.5 * float((((t - m0) ** 2) / pv)[free].sum())
            g = g + (t - m0) / pv
        return v, g[free]

    res = minimize(obj, m0[free], jac=True, method="L-BFGS-B")
    return full(res.x)


def covariance(theta, p, y) -> np.ndarray:
    """Inverse observed information (numerical Hessian of the analytic gradient)."""
    h = np.zeros((3, 3))
    eps = 1e-4
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        h[:, j] = (_grad(theta + d, p, y) - _grad(theta - d, p, y)) / (2 * eps)
    h = (h + h.T) / 2
    try:
        return np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan)


def eb_tau2(thetas: dict, covs: dict, center) -> np.ndarray:
    """Method-of-moments between-unit variance of each parameter around ``center`` (dict unit -> center or one
    vector)."""
    units = list(thetas)
    dev = np.array([thetas[u] - (center[u] if isinstance(center, dict) else center) for u in units])
    v = np.array([np.diag(covs[u]) for u in units])
    return np.maximum(np.nanmean(dev ** 2, 0) - np.nanmean(v, 0), 0.0)


class Calibrator:
    """Fits every map variant on (p, y, unit labels) of earlier seasons and applies it to new predictions."""

    def __init__(self, p, y, league, group):
        self.p, self.y = np.asarray(p, float), np.asarray(y, int)
        self.league, self.group = np.asarray(league), np.asarray(group)
        self.g = fit(self.p, self.y)
        self.info: dict = {"global": self.g.tolist()}

    def _unit_mle(self, labels):
        th, cv, n = {}, {}, {}
        for u in np.unique(labels):
            m = labels == u
            th[u] = fit(self.p[m], self.y[m], prior_mean=self.g, prior_var=np.full(3, 25.0))   # near-flat ridge
            cv[u] = covariance(th[u], self.p[m], self.y[m])
            n[u] = int(m.sum())
        return th, cv, n

    def fit_all(self) -> "Calibrator":
        self.lg_mle, self.lg_cov, self.lg_n = self._unit_mle(self.league)
        self.gr_mle, self.gr_cov, self.gr_n = self._unit_mle(self.group)
        # league <- global
        self.tau_lg = eb_tau2(self.lg_mle, self.lg_cov, self.g)
        self.lg_eb = {u: fit(self.p[self.league == u], self.y[self.league == u], self.g, self.tau_lg)
                      for u in self.lg_mle}
        # random effects: intercepts only
        tau_i = self.tau_lg.copy()
        tau_i[0] = TAU_FLOOR
        self.tau_re = tau_i
        self.lg_re = {u: fit(self.p[self.league == u], self.y[self.league == u], self.g, tau_i,
                             free=(False, True, True)) for u in self.lg_mle}
        # league <- its group (two-level): group maps themselves shrunk to global first
        self.tau_gr = eb_tau2(self.gr_mle, self.gr_cov, self.g)
        self.gr_eb = {u: fit(self.p[self.group == u], self.y[self.group == u], self.g, self.tau_gr)
                      for u in self.gr_mle}
        lg_group = {u: self.group[self.league == u][0] for u in self.lg_mle}
        centers = {u: self.gr_eb[lg_group[u]] for u in self.lg_mle}
        self.tau_lg_in_gr = eb_tau2(self.lg_mle, self.lg_cov, centers)
        self.lg_in_gr = {u: fit(self.p[self.league == u], self.y[self.league == u], centers[u], self.tau_lg_in_gr)
                         for u in self.lg_mle}
        self.info.update({
            "tau2_league": self.tau_lg.tolist(), "tau2_group": self.tau_gr.tolist(),
            "tau2_league_in_group": self.tau_lg_in_gr.tolist(),
            "league_mle": {u: v.tolist() for u, v in self.lg_mle.items()},
            "league_se": {u: np.sqrt(np.clip(np.diag(c), 0, None)).tolist() for u, c in self.lg_cov.items()},
            "league_eb": {u: v.tolist() for u, v in self.lg_eb.items()},
            "league_re": {u: v.tolist() for u, v in self.lg_re.items()},
            "group_mle": {str(u): v.tolist() for u, v in self.gr_mle.items()},
            "league_n": self.lg_n, "group_n": {str(k): v for k, v in self.gr_n.items()},
        })
        return self

    def transform(self, p, league, group, variant: str) -> np.ndarray:
        p, league, group = np.asarray(p, float), np.asarray(league), np.asarray(group)
        if variant == "C1":
            return apply(p, self.g)
        table = {"C2": self.lg_mle, "C3": self.lg_eb, "C3i": self.lg_re, "C5": self.lg_in_gr}.get(variant)
        labels = league
        if variant == "C4":
            table, labels = self.gr_mle, group
        out = apply(p, self.g)                       # unseen unit -> global map
        for u in np.unique(labels):
            if u in table:
                m = labels == u
                out[m] = apply(p[m], table[u])
        return out


VARIANTS = {
    "C1": "глобальная перекалибровка (контроль)",
    "C2": "по лиге, без сжатия",
    "C3": "по лиге, EB-сжатие к глобальной",
    "C3i": "случайные эффекты лиги (только сдвиги D/A)",
    "C4": "по группе лиг",
    "C5": "лига → группа → глобальная (иерархия)",
}
