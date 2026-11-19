"""Score-matrix constructions compared in the Goal-Model study (vectorised over matches, 11 x 11 matrices).

prod      production: independent Poisson(λh) x Poisson(λa), H / D / A regions rescaled to the 1X2 classifier
raw       independent Poisson without reconciliation
partial   regions moved only part of the way: factor (p_k / mass_k)^α, α chosen on 2023–2025
dc        Dixon–Coles τ with ρ, THEN reconciled (ρ fitted on the reconciled matrices of 2023–2025 — the fix of
          the latent production defect found by v6.2: production's pipeline fits ρ on the raw means)
nb        negative-binomial marginals (size r fitted on 2023–2025), reconciled
ipf       iterative proportional fitting: the matrix keeps the 1X2 of the classifier AND the total-goals
          distribution of the regressors' Poisson (production's region scaling distorts the totals)
sup       supremacy refit: λh + λa kept from the regressors, the split chosen so that the Poisson P(H) − P(A)
          equals the classifier's, then reconciled
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from experiments.v6_3.recent_weighting import lib as L
from src.markets import distributions as DIST

MAXG = L.MAXG
_I, _J = L._I, L._J
MASKS = DIST.outcome_masks(MAXG + 1)


def _norm(m):
    return m / m.sum((1, 2), keepdims=True)


def poisson(lh, la):
    return L.poisson_matrices(lh, la)


def nb_matrices(lh, la, size: float):
    k = np.arange(MAXG + 1)
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    ph = stats.nbinom.pmf(k[None, :], size, (size / (size + lh))[:, None])
    pa = stats.nbinom.pmf(k[None, :], size, (size / (size + la))[:, None])
    return _norm(ph[:, :, None] * pa[:, None, :])


def dc_matrices(lh, la, rho: float):
    m = poisson(lh, la)
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    tau = np.ones_like(m)
    tau[:, 0, 0] = 1 - lh * la * rho
    tau[:, 0, 1] = 1 + lh * rho
    tau[:, 1, 0] = 1 + la * rho
    tau[:, 1, 1] = 1 - rho
    return _norm(m * np.clip(tau, 1e-6, None))


def ipf(raw, p, iters: int = 30):
    """Match the 1X2 regions to p and the total-goal anti-diagonals to the raw matrix's totals; ends on 1X2."""
    tot = _I + _J
    target_t = np.stack([raw[:, tot == t].sum(1) for t in range(2 * MAXG + 1)], 1)
    m = raw.copy()
    for _ in range(iters):
        for t in range(2 * MAXG + 1):
            mask = tot == t
            mass = m[:, mask].sum(1)
            m[:, mask] *= np.where(mass > 0, target_t[:, t] / np.maximum(mass, 1e-15), 1.0)[:, None]
        m = L.reconcile(m, p)
    return m


def supremacy(lh, la, p):
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    s = lh + la
    target = p[:, 0] - p[:, 2]
    lo, hi = -0.98 * s, 0.98 * s
    for _ in range(40):
        d = (lo + hi) / 2
        m = poisson((s + d) / 2, (s - d) / 2)
        g = m[:, MASKS[0]].sum(1) - m[:, MASKS[2]].sum(1)
        up = g < target
        lo = np.where(up, d, lo)
        hi = np.where(up, hi, d)
    d = (lo + hi) / 2
    return L.reconcile(poisson((s + d) / 2, (s - d) / 2), p)


def build(kind: str, lh, la, p, par: dict | None = None):
    par = par or {}
    p = np.asarray(p, float)
    if kind == "prod":
        return L.reconcile(poisson(lh, la), p)
    if kind == "raw":
        return poisson(lh, la)
    if kind == "partial":
        return L.reconcile(poisson(lh, la), p, par["alpha"])
    if kind == "dc":
        return L.reconcile(dc_matrices(lh, la, par["rho"]), p)
    if kind == "dc_raw_rho":
        return L.reconcile(dc_matrices(lh, la, par["rho_raw"]), p)
    if kind == "nb":
        return L.reconcile(nb_matrices(lh, la, par["size"]), p)
    if kind == "ipf":
        return ipf(poisson(lh, la), p)
    if kind == "sup":
        return supremacy(lh, la, p)
    raise ValueError(kind)


def matrix_fn(kind: str, par: dict | None = None):
    """Single-match version for production's engine (lib.matrix_patch)."""
    def fn(lh, la, p):
        return build(kind, np.array([lh]), np.array([la]), np.asarray(p, float)[None, :], par)[0]
    return fn


def score_nll(m, hg, ag) -> float:
    return float(-np.log(np.clip(m[np.arange(len(hg)), np.clip(hg, 0, MAXG), np.clip(ag, 0, MAXG)], 1e-12, 1)).mean())


def fit_params(lh, la, p, hg, ag) -> dict:
    """ρ (after reconciliation), ρ on the raw matrices (production's pipeline), NB size and α on selection data."""
    hg, ag = np.asarray(hg, int), np.asarray(ag, int)
    out = {}
    r = minimize_scalar(lambda x: score_nll(build("dc", lh, la, p, {"rho": x}), hg, ag), bounds=(-0.3, 0.3),
                        method="bounded", options={"xatol": 1e-4})
    out["rho"] = float(r.x)
    out["rho_raw"] = float(DIST.fit_dixon_coles_rho(lh, la, hg, ag))
    r = minimize_scalar(lambda x: score_nll(build("nb", lh, la, p, {"size": np.exp(x)}), hg, ag),
                        bounds=(np.log(2), np.log(2000)), method="bounded", options={"xatol": 1e-3})
    out["size"] = float(np.exp(r.x))
    r = minimize_scalar(lambda x: score_nll(build("partial", lh, la, p, {"alpha": x}), hg, ag), bounds=(0, 1.5),
                        method="bounded", options={"xatol": 1e-3})
    out["alpha"] = float(r.x)
    return out
