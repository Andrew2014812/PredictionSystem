"""Vectorised score-matrix helpers (identical maths to src/markets/distributions.py)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from src import config
from src.markets.distributions import dixon_coles_tau, score_matrix

G = config.MAX_GOALS + 1
_I, _J = np.indices((G, G))


def poisson_vectors(lam: np.ndarray) -> np.ndarray:
    lam = np.clip(np.asarray(lam, float), 1e-6, None)
    pmf = stats.poisson.pmf(np.arange(G)[None, :], lam[:, None])
    pmf[:, -1] += np.clip(1.0 - pmf.sum(1), 0, None)
    return pmf


def matrices(lh: np.ndarray, la: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """(n, G, G) score matrices, normalised; Dixon-Coles tau on the four low cells."""
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    m = poisson_vectors(lh)[:, :, None] * poisson_vectors(la)[:, None, :]
    if rho:
        tau = np.stack([np.stack([1 - lh * la * rho, 1 + lh * rho], -1),
                        np.stack([1 + la * rho, np.full_like(lh, 1 - rho)], -1)], 1)
        m[:, :2, :2] *= np.clip(tau, 1e-6, None)
    return m / m.sum((1, 2), keepdims=True)


def outcome_1x2(m: np.ndarray) -> np.ndarray:
    return np.stack([(m * (_I > _J)).sum((1, 2)), (m * (_I == _J)).sum((1, 2)), (m * (_I < _J)).sum((1, 2))], 1)


def reconcile(m: np.ndarray, p1x2: np.ndarray) -> np.ndarray:
    """Vectorised reconcile_with_1x2: rescale H/D/A regions to the classifier probabilities."""
    cur = outcome_1x2(m)
    scale = np.where(cur > 0, p1x2 / np.clip(cur, 1e-300, None), 0.0)
    region = np.where(_I > _J, 0, np.where(_I == _J, 1, 2))
    out = m * scale[:, region]
    return out / out.sum((1, 2), keepdims=True)


def self_check() -> float:
    """Max abs difference against the production functions on a few rates."""
    lh, la = np.array([0.4, 1.3, 2.8]), np.array([1.9, 1.0, 0.3])
    v = matrices(lh, la, rho=-0.08)
    ref = np.stack([score_matrix(a, b, rho=-0.08) for a, b in zip(lh, la)])
    _ = dixon_coles_tau
    return float(np.abs(v - ref).max())
