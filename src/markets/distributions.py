"""Probability distributions behind every market.

Goals
-----
The two goal regressors output Poisson rates ``lambda_home`` and
``lambda_away``. Assuming the two counts are independent given the rates,

    P(home = i, away = j) = Pois(i; lambda_home) * Pois(j; lambda_away)

gives a full score matrix. Every goal market (totals, BTTS, exact score,
handicap) is a sum over cells of this one matrix, so all markets are
mutually consistent.

Independent Poisson is known to misprice draws, while the dedicated 1X2
classifier is trained on exactly that target. ``reconcile_with_1x2``
therefore rescales the three regions of the matrix (home win / draw / away
win) so that their masses equal the classifier probabilities, keeping the
relative shape of the scores inside each region.

Corners
-------
Total corners are over-dispersed counts. With the expected total ``mu`` and
a negative-binomial size ``r`` (estimated on validation residuals) the
distribution is NB(mu, r); without over-dispersion it falls back to Poisson.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .. import config

MAX_CORNERS = 40


def poisson_vector(lam: float, max_goals: int = config.MAX_GOALS) -> np.ndarray:
    pmf = stats.poisson.pmf(np.arange(max_goals + 1), max(lam, 1e-6))
    pmf[-1] += max(0.0, 1.0 - pmf.sum())       # fold the tail into the last cell
    return pmf


def score_matrix(lambda_home: float, lambda_away: float,
                 max_goals: int = config.MAX_GOALS) -> np.ndarray:
    m = np.outer(poisson_vector(lambda_home, max_goals), poisson_vector(lambda_away, max_goals))
    return m / m.sum()


def outcome_masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.indices((size, size))
    return i > j, i == j, i < j


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    home, draw, away = outcome_masks(matrix.shape[0])
    return np.array([matrix[home].sum(), matrix[draw].sum(), matrix[away].sum()])


def reconcile_with_1x2(matrix: np.ndarray, p1x2: np.ndarray) -> np.ndarray:
    """Rescale the H/D/A regions of ``matrix`` to the classifier probabilities."""
    current = outcome_probabilities(matrix)
    out = matrix.copy()
    for mask, target, mass in zip(outcome_masks(matrix.shape[0]), p1x2, current):
        if mass > 0:
            out[mask] *= target / mass
    return out / out.sum()


def empirical_score_matrix(home_goals: np.ndarray, away_goals: np.ndarray,
                           max_goals: int = config.MAX_GOALS, smoothing: float = 0.5) -> np.ndarray:
    """Historical score frequencies (additively smoothed)."""
    size = max_goals + 1
    m = np.full((size, size), smoothing / size ** 2)
    hg = np.clip(home_goals.astype(int), 0, max_goals)
    ag = np.clip(away_goals.astype(int), 0, max_goals)
    np.add.at(m, (hg, ag), 1.0)
    return m / m.sum()


def corners_distribution(mu: float, nb_size: float | None) -> np.ndarray:
    k = np.arange(MAX_CORNERS + 1)
    if nb_size:
        p = nb_size / (nb_size + mu)
        pmf = stats.nbinom.pmf(k, nb_size, p)
    else:
        pmf = stats.poisson.pmf(k, mu)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def empirical_corners_distribution(totals: np.ndarray, smoothing: float = 0.5) -> np.ndarray:
    pmf = np.full(MAX_CORNERS + 1, smoothing / (MAX_CORNERS + 1))
    np.add.at(pmf, np.clip(totals.astype(int), 0, MAX_CORNERS), 1.0)
    return pmf / pmf.sum()


def prob_over(pmf: np.ndarray, line: float) -> float:
    return float(pmf[np.arange(len(pmf)) > line].sum())
