"""Explaining the probability of a market through the chain of models that produces it.

No market probability is a direct model output. The chain is

    features x ──► XGBoost models ──► model outputs u ──► score matrix / corners distribution ──► P(market)

with the model outputs in their margin space

* goal markets (1X2, double chance, totals, BTTS, handicaps, exact score):
  u = (z_H, z_D, z_A, log λ_home, log λ_away) — the three class scores of the 1X2 classifier
  (p = softmax(z)) and the log-rates of the two goal regressors. The score matrix is
  ``reconcile_with_1x2(score_matrix(λ_home, λ_away, ρ), p)``, so EVERY goal market depends on the
  1X2 classifier as well; a total, for example, is P = Σ_r p_r · Q_r(λ_home, λ_away), where Q_r is
  the share of the market inside the home-win / draw / away-win region r of the Poisson matrix;
* corner totals: u = (log μ), the corners regressor, and P = P_NB(total > line | μ, r).

Attribution (the "path-integrated chain rule")
---------------------------------------------
TreeSHAP gives, for every model output k, additive feature contributions φ_jk in margin units with
Σ_j φ_jk = u_k − ū_k (ū = the explainers' expected outputs). The market probability is a smooth
function g(u), so along the straight path ū → u (Aumann–Shapley / integrated gradients in the space
of model outputs)

    G_k = ∫₀¹ ∂g/∂u_k (ū + t (u − ū)) dt,         P(x) − g(ū) = Σ_k G_k (u_k − ū_k)

and the contribution of feature j to the market probability is

    c_j = Σ_k G_k · φ_jk,        Σ_j c_j = P(x) − g(ū)        (exact, up to quadrature error).

c_j is in probability units, has the sign of the feature's real effect on THIS market (a feature
that raises the expected goals but also the draw probability is weighed by both routes) and is
zero for models that do not drive the market (λ for "Home win", p for nothing in corners).
The integral is evaluated with Gauss–Legendre quadrature on central differences of g.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..markets.distributions import MAX_CORNERS
from ..markets.markets import market_group, market_line
from ..modelling.tasks import RESULT_CLASSES

GOAL_OUTPUTS = ("z_H", "z_D", "z_A", "log_lambda_home", "log_lambda_away")
CORNER_OUTPUTS = ("log_mu",)
OUTPUT_MODEL = {"z_H": "result", "z_D": "result", "z_A": "result", "log_lambda_home": "home_goals",
                "log_lambda_away": "away_goals", "log_mu": "corners"}
N_NODES = 12            # Gauss–Legendre nodes of the path integral
STEP = 1e-5             # central-difference step in margin units


def outputs_for(market: str) -> tuple[str, ...]:
    return CORNER_OUTPUTS if market_group(market) == "CORNERS" else GOAL_OUTPUTS


# ----------------------------------------------------------------------------- the outer function g
def _poisson_rows(lam: np.ndarray, size: int) -> np.ndarray:
    k = np.arange(size)
    pmf = stats.poisson.pmf(k[None, :], np.maximum(lam, 1e-6)[:, None])
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
    return pmf


def score_matrices(U: np.ndarray, rho: float = 0.0, max_goals: int = config.MAX_GOALS) -> np.ndarray:
    """Reconciled score matrices for rows of goal outputs U (n, 5) — the same computation as
    ``reconcile_with_1x2(score_matrix(λh, λa, ρ), p)`` in the prediction engine, vectorised."""
    U = np.atleast_2d(np.asarray(U, float))
    z = U[:, :3] - U[:, :3].max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    lh, la = np.exp(U[:, 3]), np.exp(U[:, 4])
    size = max_goals + 1
    M = _poisson_rows(lh, size)[:, :, None] * _poisson_rows(la, size)[:, None, :]
    if rho:
        tau = np.stack([np.stack([1 - lh * la * rho, 1 + lh * rho], -1),
                        np.stack([1 + la * rho, np.full_like(lh, 1 - rho)], -1)], 1)
        M[:, :2, :2] *= np.clip(tau, 1e-6, None)
    M /= M.sum(axis=(1, 2), keepdims=True)
    i, j = np.indices((size, size))
    for r, mask in enumerate((i > j, i == j, i < j)):
        mass = M[:, mask].sum(axis=1)
        scale = np.where(mass > 0, p[:, r] / np.where(mass > 0, mass, 1.0), 1.0)
        M[:, mask] *= scale[:, None]
    return M / M.sum(axis=(1, 2), keepdims=True)


def selection_mask(market: str, selection: str, size: int = config.MAX_GOALS + 1) -> np.ndarray:
    """Cells of the score matrix (home goals i, away goals j) where a goal-market selection WINS."""
    i, j = np.indices((size, size))
    group, line = market_group(market), market_line(market)
    if group == "1X2":
        return {"H": i > j, "D": i == j, "A": i < j}[selection]
    if group == "DOUBLE_CHANCE":
        return {"1X": i >= j, "X2": i <= j, "12": i != j}[selection]
    if group == "TOTAL":
        return (i + j > line) if selection == "OVER" else (i + j < line)
    if group == "BTTS":
        both = (i > 0) & (j > 0)
        return both if selection == "YES" else ~both
    if group in ("HANDICAP", "EHANDICAP"):
        margin = i + line - j
        return {"H": margin > 0, "D": margin == 0, "A": margin < 0}[selection]
    if group == "EXACT_SCORE":
        h, a = (int(x) for x in selection.split("-"))
        return (i == h) & (j == a)
    raise ValueError(f"{market} is not a goal market")


def corner_probability(log_mu: np.ndarray, line: float, selection: str, nb_size: float | None) -> np.ndarray:
    mu = np.exp(np.asarray(log_mu, float))
    k = np.arange(MAX_CORNERS + 1)
    if nb_size:
        pmf = stats.nbinom.pmf(k[None, :], nb_size, (nb_size / (nb_size + mu))[:, None])
    else:
        pmf = stats.poisson.pmf(k[None, :], mu[:, None])
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
    over = pmf[:, k > line].sum(axis=1)
    return over if selection == "OVER" else 1.0 - over


def market_probability(market: str, selection: str, U, rho: float = 0.0,
                       nb_size: float | None = None) -> np.ndarray:
    """g(u): the probability of one selection for rows of model outputs U (n, len(outputs_for(market)))."""
    U = np.atleast_2d(np.asarray(U, float))
    if market_group(market) == "CORNERS":
        return corner_probability(U[:, 0], market_line(market), selection, nb_size)
    M = score_matrices(U, rho)
    return M[:, selection_mask(market, selection, M.shape[1])].sum(axis=1)


def gradient(market: str, selection: str, U, rho: float = 0.0, nb_size: float | None = None,
             step: float = STEP) -> np.ndarray:
    """∂g/∂u_k at rows of U by central differences (n, k)."""
    U = np.atleast_2d(np.asarray(U, float))
    n, k = U.shape
    shifts = np.concatenate([U[:, None, :] + step * np.eye(k)[None], U[:, None, :] - step * np.eye(k)[None]], 1)
    vals = market_probability(market, selection, shifts.reshape(-1, k), rho, nb_size).reshape(n, 2 * k)
    return (vals[:, :k] - vals[:, k:]) / (2 * step)


def path_weights(market: str, selection: str, u, base, rho: float = 0.0, nb_size: float | None = None,
                 nodes: int = N_NODES) -> np.ndarray:
    """G_k = ∫₀¹ ∂g/∂u_k(ū + t(u − ū)) dt (Gauss–Legendre)."""
    u, base = np.asarray(u, float), np.asarray(base, float)
    t, w = np.polynomial.legendre.leggauss(nodes)
    t, w = (t + 1) / 2, w / 2
    points = base[None, :] + t[:, None] * (u - base)[None, :]
    return (w[:, None] * gradient(market, selection, points, rho, nb_size)).sum(axis=0)


# ----------------------------------------------------------------------------- model outputs and SHAP
@dataclass
class ChainInputs:
    """Model outputs u, their SHAP base values ū and the TreeSHAP matrix (features × outputs)."""
    names: tuple[str, ...]
    u: np.ndarray
    base: np.ndarray
    phi: pd.DataFrame                    # index = feature (one-hot league collapsed), columns = names
    values: pd.Series                    # the feature values of this match
    means: dict = field(default_factory=dict)
    rho: float = 0.0
    nb_size: float | None = None


def chain_inputs(row: pd.DataFrame, bundles: dict, names: tuple[str, ...]) -> ChainInputs | None:
    """TreeSHAP of every model output in ``names`` for one match (``row``: one feature row)."""
    from .shap_explainer import _collapse_one_hot, _explainer
    needed = {OUTPUT_MODEL[n] for n in names}
    if any(bundles.get(m) is None for m in needed):
        return None
    cols, u, base, values, means = {}, [], [], None, {}
    cache = {}
    for name in names:
        task = OUTPUT_MODEL[name]
        if task not in cache:
            bundle = bundles[task]
            X = bundle.matrix(row)
            ex = _explainer(bundle)
            cache[task] = (X, np.asarray(ex.shap_values(X)), np.atleast_1d(np.asarray(ex.expected_value, float)),
                           bundle.model.predict(X, output_margin=True))
            means.update(bundle.extra.get("feature_means", {}))
        X, sv, ev, margin = cache[task]
        if task == "result":
            c = RESULT_CLASSES.index(name[-1])
            contrib, m, b = sv[0, :, c], float(np.atleast_2d(margin)[0, c]), float(ev[c])
        else:
            contrib, m, b = sv[0], float(np.ravel(margin)[0]), float(ev[0])
        s, v = _collapse_one_hot(pd.Series(contrib, index=X.columns), X.iloc[0])
        cols[name] = s
        values = v if values is None else values.combine_first(v)
        u.append(m)
        base.append(b)
    phi = pd.DataFrame(cols).fillna(0.0)[list(names)]
    rho = bundles["home_goals"].extra.get("dc_rho", 0.0) if "home_goals" in needed else 0.0
    nb = bundles["corners"].extra.get("nb_size") if "corners" in needed else None
    return ChainInputs(tuple(names), np.array(u), np.array(base), phi, values.reindex(phi.index), means,
                       float(rho or 0.0), nb)


@dataclass
class ChainExplanation:
    market: str
    selection: str
    probability: float                   # g(u): the model's probability of the selection
    base_probability: float              # g(ū): the same selection for an "average" set of model outputs
    weights: pd.Series                   # G_k per model output
    output_effect: pd.Series             # G_k (u_k − ū_k): probability moved through each model output
    contributions: pd.DataFrame          # feature, contribution, value, mean, + one column per output

    @property
    def total(self) -> float:
        return float(self.contributions["contribution"].sum())

    def model_effect(self) -> pd.Series:
        """Probability moved through each MODEL (the classifier's three outputs added up)."""
        return self.output_effect.groupby(lambda n: OUTPUT_MODEL[n]).sum()


def attribute(inputs: ChainInputs, market: str, selection: str, nodes: int = N_NODES) -> ChainExplanation:
    """Path-integrated chain rule: feature contributions to the market probability itself."""
    g = lambda U: market_probability(market, selection, U, inputs.rho, inputs.nb_size)   # noqa: E731
    G = path_weights(market, selection, inputs.u, inputs.base, inputs.rho, inputs.nb_size, nodes)
    via = inputs.phi.to_numpy() * G[None, :]
    out = pd.DataFrame(via, index=inputs.phi.index, columns=[f"via_{n}" for n in inputs.names])
    out.insert(0, "contribution", via.sum(axis=1))
    out["value"] = inputs.values
    out["mean"] = [inputs.means.get(f) for f in out.index]
    out = out.rename_axis("feature").reset_index()
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return ChainExplanation(market, selection, float(g(inputs.u)[0]), float(g(inputs.base)[0]),
                            pd.Series(G, index=inputs.names), pd.Series(G * (inputs.u - inputs.base), index=inputs.names),
                            out)


_CACHE: "OrderedDict[tuple, ChainInputs | None]" = OrderedDict()
_CACHE_SIZE = 32


def explain_market(row: pd.DataFrame, market: str, selection: str, bundles: dict,
                   key: str | None = None) -> ChainExplanation | None:
    """Chain explanation of one selection. With ``key`` (e.g. "<model version>|<match id>") the
    TreeSHAP of the models is computed once per match and reused for every market of the page."""
    names = outputs_for(market)
    if key is None:
        inputs = chain_inputs(row, bundles, names)
    else:
        ck = (key, names, tuple(id(bundles.get(OUTPUT_MODEL[n])) for n in names))
        if ck not in _CACHE:
            _CACHE[ck] = chain_inputs(row, bundles, names)
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        inputs = _CACHE[ck]
    return attribute(inputs, market, selection) if inputs is not None else None


# ----------------------------------------------------------------------------- chain values for the card
@dataclass
class ChainValues:
    """The model outputs a market is computed from, taken from the stored prediction."""
    group: str
    xg_home: float | None = None
    xg_away: float | None = None
    p1x2: tuple[float, float, float] | None = None
    exp_corners: float | None = None
    probability: float | None = None          # the selection, recomputed from the stored outputs
    push: float | None = None                 # whole-line Asian handicap: refund probability
    p_home_scores: float | None = None
    p_away_scores: float | None = None
    alternatives: list[tuple[str, float]] = field(default_factory=list)   # exact score: next likeliest

    @property
    def xg_total(self) -> float | None:
        return None if self.xg_home is None else self.xg_home + self.xg_away


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def chain_values(market: str, selection: str, snap, rho: float = 0.0) -> ChainValues | None:
    """Recompute the selection's probability from the stored model outputs of the prediction
    (p_home/p_draw/p_away, xg_home/xg_away, exp_corners/corners_nb_size)."""
    if snap is None:
        return None
    get = snap.get if hasattr(snap, "get") else (lambda k, d=None: d)
    group = market_group(market)
    if group == "CORNERS":
        mu, size = _num(get("exp_corners")), _num(get("corners_nb_size"))
        if mu is None:
            return None
        from ..markets.distributions import corners_distribution, prob_over
        over = prob_over(corners_distribution(mu, size), market_line(market))
        return ChainValues(group, exp_corners=mu, probability=over if selection == "OVER" else 1 - over)
    p = tuple(_num(get(c)) for c in ("p_home", "p_draw", "p_away"))
    lh, la = _num(get("xg_home")), _num(get("xg_away"))
    if None in p or lh is None or la is None:
        return None
    U = np.array([[*np.log(np.clip(p, 1e-12, None)), np.log(max(lh, 1e-6)), np.log(max(la, 1e-6))]])
    M = score_matrices(U, rho)[0]
    out = ChainValues(group, xg_home=lh, xg_away=la, p1x2=p,
                      probability=float(M[selection_mask(market, selection, M.shape[0])].sum()))
    if group in ("HANDICAP",):
        out.push = float(M[selection_mask(market.replace("HANDICAP", "EHANDICAP"), "D", M.shape[0])].sum())
    out.p_home_scores, out.p_away_scores = float(M[1:, :].sum()), float(M[:, 1:].sum())
    if group == "EXACT_SCORE":
        order = np.argsort(M, axis=None)[::-1]
        for idx in order[:4]:
            h, a = np.unravel_index(idx, M.shape)
            if f"{h}-{a}" != selection and len(out.alternatives) < 2:
                out.alternatives.append((f"{h}-{a}", float(M[h, a])))
    return out
