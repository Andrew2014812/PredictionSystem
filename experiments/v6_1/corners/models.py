"""Mean models of total corners (and of home / away corners) compared in the v6.1 study.

Every model has the same interface: ``fit(rows, y)`` then ``predict(rows) -> expected count``.
Hyperparameters, tree counts, regularisation strengths and blend weights are chosen on the INNER
split of a fold only (train < s-1, validation s-1); the OUTER fit then uses all seasons < s.

    XGB            production-faithful XGBoost (count:poisson, recency weights, league one-hot)
    tune_xgb       Hyperopt TPE with the production search space, on the inner split
    LGBM / CatB    LightGBM / CatBoost with a Poisson objective and early stopping (small grids)
    PoissonGLM     sklearn PoissonRegressor on a standardised compact feature set (+ missing flags)
    NBGLM          negative-binomial (NB2) GLM, coefficients and size r by penalised maximum likelihood
    TeamStrength   hierarchical / partial-pooling team model of SIDE corners:
                       log E[corners of team i vs j] = b(league, venue) + att_i + def_j
                   att / def are ridge-penalised (MAP of Gaussian random effects, shrunk to the league
                   mean), recent matches weigh more (exponential decay); refitted every half month
                   on the matches before the refit date only.
"""
from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

from src import config
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.training import SEARCH_SPACE, _clean_params, recency_weights

from .data import N_JOBS

log = logging.getLogger("corners")


def pdev(y, mu) -> float:
    return float(mean_poisson_deviance(np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)))


# ---------------------------------------------------------------------------------------------- XGBoost
class XGB:
    def __init__(self, numeric: list[str], params: dict, n_estimators: int, objective: str = "count:poisson"):
        self.numeric, self.params, self.n = list(numeric), dict(params), int(n_estimators)
        self.objective = objective
        self.seconds = 0.0

    def _est(self, n, early=False):
        kw = dict(n_estimators=n, objective=self.objective, eval_metric="poisson-nloglik", tree_method="hist",
                  random_state=config.RANDOM_STATE, n_jobs=N_JOBS,
                  **{k: v for k, v in self.params.items() if k != "half_life_years"})
        if early:
            kw["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS
        return XGBRegressor(**kw)

    def _w(self, rows):
        return recency_weights(rows["date"], self.params.get("half_life_years", 0))

    def early_trees(self, train, y_tr, valid, y_va) -> int:
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        m = self._est(config.MAX_ESTIMATORS, early=True)
        m.fit(prep.transform(train), np.asarray(y_tr), sample_weight=self._w(train),
              eval_set=[(prep.transform(valid), np.asarray(y_va))], verbose=False)
        self.n = int(m.best_iteration) + 1
        return self.n

    def fit(self, rows, y):
        t0 = time.time()
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), np.asarray(y), sample_weight=self._w(rows))
        self.seconds = time.time() - t0
        return self

    def predict(self, rows):
        return self.model.predict(self.prep.transform(rows)).astype(float)


class XGBOffset(XGB):
    """XGBoost boosted FROM an offset: log mu = log(offset) + trees(x) (``base_margin``).

    The offset is the hierarchical team-strength expectation, so the trees only learn corrections."""

    def early_trees(self, train, y_tr, valid, y_va, off_tr=None, off_va=None) -> int:
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        m = self._est(config.MAX_ESTIMATORS, early=True)
        m.fit(prep.transform(train), np.asarray(y_tr), sample_weight=self._w(train), base_margin=np.log(off_tr),
              eval_set=[(prep.transform(valid), np.asarray(y_va))], base_margin_eval_set=[np.log(off_va)],
              verbose=False)
        self.n = int(m.best_iteration) + 1
        return self.n

    def fit(self, rows, y, offset=None):
        t0 = time.time()
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), np.asarray(y), sample_weight=self._w(rows),
                                           base_margin=np.log(offset))
        self.seconds = time.time() - t0
        return self

    def predict(self, rows, offset=None):
        return self.model.predict(self.prep.transform(rows), base_margin=np.log(offset)).astype(float)


def tune_xgb(train, y_tr, valid, y_va, numeric, evals: int) -> tuple[dict, int, list]:
    """Hyperopt TPE over the production search space (src/modelling/training.py), Poisson deviance."""
    from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
    prep = FeaturePreprocessor(numeric, ["league"]).fit(train)
    X_tr, X_va = prep.transform(train), prep.transform(valid)
    trials_log = []

    def objective(p):
        p = _clean_params(p)
        m = XGB(numeric, p, config.MAX_ESTIMATORS)._est(config.MAX_ESTIMATORS, early=True)
        m.fit(X_tr, np.asarray(y_tr), sample_weight=recency_weights(train["date"], p["half_life_years"]),
              eval_set=[(X_va, np.asarray(y_va))], verbose=False)
        score = pdev(y_va, m.predict(X_va))
        n = int(m.best_iteration) + 1
        trials_log.append({"score": score, "n_estimators": n, **p})
        return {"loss": score, "status": STATUS_OK, "n_estimators": n}

    trials = Trials()
    best = fmin(objective, SEARCH_SPACE, algo=tpe.suggest, max_evals=evals, trials=trials,
                rstate=np.random.default_rng(config.RANDOM_STATE), show_progressbar=False)
    params = _clean_params(space_eval(SEARCH_SPACE, best))
    return params, int(trials.best_trial["result"]["n_estimators"]), trials_log


# ---------------------------------------------------------------------------------------------- other GBMs
class LGBM:
    grid = {"num_leaves": [7, 15, 31], "min_child_samples": [100, 300]}

    def __init__(self, numeric):
        self.numeric = list(numeric)
        self.params: dict = {}

    def _est(self, s, n):
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="poisson", n_estimators=n, learning_rate=0.03,
                                 num_leaves=s["num_leaves"], min_child_samples=s["min_child_samples"],
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                                 random_state=config.RANDOM_STATE, verbose=-1, n_jobs=N_JOBS)

    def tune(self, train, y_tr, valid, y_va):
        import lightgbm as lgb
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        X_tr, X_va = prep.transform(train), prep.transform(valid)
        best = None
        for vals in itertools.product(*self.grid.values()):
            s = dict(zip(self.grid, vals))
            m = self._est(s, 4000)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            n = int(m.best_iteration_ or 4000)
            score = pdev(y_va, m.predict(X_va, num_iteration=n))
            if best is None or score < best[0]:
                best = (score, s, n)
        self.params = {**best[1], "n_estimators": best[2], "inner_deviance": best[0]}
        return self

    def fit(self, rows, y):
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        s = {k: v for k, v in self.params.items() if k in self.grid}
        self.model = self._est(s, self.params["n_estimators"]).fit(self.prep.transform(rows), y)
        return self

    def predict(self, rows):
        return self.model.predict(self.prep.transform(rows)).astype(float)


class CatB:
    grid = {"depth": [4, 6], "l2_leaf_reg": [3, 10]}

    def __init__(self, numeric):
        self.numeric = list(numeric)
        self.params: dict = {}

    def _est(self, s, n, early=False):
        from catboost import CatBoostRegressor
        kw = dict(iterations=n, depth=s["depth"], learning_rate=0.05, l2_leaf_reg=s["l2_leaf_reg"],
                  loss_function="Poisson", random_seed=config.RANDOM_STATE, verbose=False, thread_count=N_JOBS,
                  allow_writing_files=False)
        if early:
            kw.update(od_type="Iter", od_wait=100)
        return CatBoostRegressor(**kw)

    def tune(self, train, y_tr, valid, y_va):
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        X_tr, X_va = prep.transform(train), prep.transform(valid)
        best = None
        for vals in itertools.product(*self.grid.values()):
            s = dict(zip(self.grid, vals))
            m = self._est(s, 3000, early=True)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            n = int(m.get_best_iteration() or 3000) + 1
            score = pdev(y_va, m.predict(X_va, prediction_type="Exponent"))
            if best is None or score < best[0]:
                best = (score, s, n)
        self.params = {**best[1], "n_estimators": best[2], "inner_deviance": best[0]}
        return self

    def fit(self, rows, y):
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        s = {k: v for k, v in self.params.items() if k in self.grid}
        self.model = self._est(s, self.params["n_estimators"]).fit(self.prep.transform(rows), y)
        return self

    def predict(self, rows):
        return np.asarray(self.model.predict(self.prep.transform(rows), prediction_type="Exponent"), float)


# ---------------------------------------------------------------------------------------------- GLMs
class LinearMatrix:
    """Median-imputed, standardised numeric features + missing flags + league one-hot (train rows only)."""

    def __init__(self, numeric):
        self.numeric = list(numeric)

    def fit(self, rows):
        X = rows[self.numeric].astype(float)
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1.0)
        self.flags = [c for c in self.numeric if X[c].isna().mean() > 0.005]
        self.leagues = sorted(rows["league"].unique())
        return self

    def transform(self, rows):
        X = rows[self.numeric].astype(float)
        Z = ((X.fillna(self.med) - self.mu) / self.sd).clip(-6, 6)
        parts = [Z.to_numpy(), X[self.flags].isna().to_numpy(float),
                 (rows["league"].to_numpy()[:, None] == np.array(self.leagues)[None, :]).astype(float)]
        return np.hstack(parts)


class PoissonGLM:
    grid = (1e-3, 1e-2, 1e-1, 0.3, 1.0)

    def __init__(self, numeric):
        self.numeric = list(numeric)
        self.params: dict = {}

    def tune(self, train, y_tr, valid, y_va):
        mat = LinearMatrix(self.numeric).fit(train)
        X_tr, X_va = mat.transform(train), mat.transform(valid)
        scores = {a: pdev(y_va, PoissonRegressor(alpha=a, max_iter=3000).fit(X_tr, y_tr).predict(X_va))
                  for a in self.grid}
        a = min(scores, key=scores.get)
        self.params = {"alpha": a, "inner_deviance": scores[a]}
        return self

    def fit(self, rows, y):
        self.mat = LinearMatrix(self.numeric).fit(rows)
        self.model = PoissonRegressor(alpha=self.params["alpha"], max_iter=3000).fit(self.mat.transform(rows), y)
        return self

    def predict(self, rows):
        return self.model.predict(self.mat.transform(rows)).astype(float)


def nb_loglik(y, mu, r) -> np.ndarray:
    """log NB2 pmf with mean mu and size r (vectorised)."""
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-9, None)
    r = np.broadcast_to(np.asarray(r, float), mu.shape)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


class NBGLM:
    """NB2 regression: log mu = a + X b, size r; penalised ML (L2 on b) by L-BFGS with exact gradients."""
    grid = (1e-4, 1e-3, 1e-2)

    def __init__(self, numeric):
        self.numeric = list(numeric)
        self.params: dict = {}

    @staticmethod
    def _fit(X, y, lam):
        n, p = X.shape
        y = np.asarray(y, float)

        def f(theta):
            a, b, logr = theta[0], theta[1:p + 1], theta[-1]
            r = np.exp(logr)
            eta = np.clip(a + X @ b, -10, 10)
            mu = np.exp(eta)
            ll = nb_loglik(y, mu, r)
            from scipy.special import digamma
            # d ll / d eta = r (y - mu) / (r + mu)
            g_eta = r * (y - mu) / (r + mu)
            g_r = digamma(y + r) - digamma(r) + np.log(r / (r + mu)) + (mu - y) / (r + mu)
            obj = -ll.mean() + 0.5 * lam * b @ b
            grad = np.concatenate([[-g_eta.mean()], -(X.T @ g_eta) / n + lam * b, [-(g_r.mean()) * r]])
            return obj, grad

        theta0 = np.concatenate([[np.log(max(y.mean(), 1e-3))], np.zeros(p), [np.log(30.0)]])
        res = optimize.minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        return res.x

    def tune(self, train, y_tr, valid, y_va):
        mat = LinearMatrix(self.numeric).fit(train)
        X_tr, X_va = mat.transform(train), mat.transform(valid)
        scores = {}
        for lam in self.grid:
            th = self._fit(X_tr, y_tr, lam)
            mu = np.exp(np.clip(th[0] + X_va @ th[1:-1], -10, 10))
            scores[lam] = float(-nb_loglik(y_va, mu, np.exp(th[-1])).mean())
        lam = min(scores, key=scores.get)
        self.params = {"l2": lam, "inner_nll": scores[lam]}
        return self

    def fit(self, rows, y):
        self.mat = LinearMatrix(self.numeric).fit(rows)
        self.theta = self._fit(self.mat.transform(rows), y, self.params["l2"])
        self.size = float(np.exp(self.theta[-1]))
        return self

    def predict(self, rows):
        X = self.mat.transform(rows)
        return np.exp(np.clip(self.theta[0] + X @ self.theta[1:-1], -10, 10))


# ---------------------------------------------------------------------------------------------- hierarchical
def _side_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: the corners of each team, with the opponent and the venue."""
    h = pd.DataFrame({"match_id": df["match_id"].to_numpy(), "date": df["date"].to_numpy(),
                      "league": df["league"].to_numpy(), "venue": "H",
                      "att": (df["country"] + "|" + df["home_team"]).to_numpy(),
                      "dfn": (df["country"] + "|" + df["away_team"]).to_numpy(),
                      "y": df["home_corners"].to_numpy(float)})
    a = pd.DataFrame({"match_id": df["match_id"].to_numpy(), "date": df["date"].to_numpy(),
                      "league": df["league"].to_numpy(), "venue": "A",
                      "att": (df["country"] + "|" + df["away_team"]).to_numpy(),
                      "dfn": (df["country"] + "|" + df["home_team"]).to_numpy(),
                      "y": df["away_corners"].to_numpy(float)})
    return pd.concat([h, a], ignore_index=True)


class TeamStrength:
    """Partial-pooling Poisson model of side corners (see module docstring).

    Objective (weights w = 0.5 ** (age_days / half_life)):
        sum w (mu - y log mu) / sum w  +  alpha / 2 * (||att||^2 + ||def||^2)
    The league x venue baselines are not penalised. Solved by L-BFGS with the exact gradient,
    warm-started from the previous refit (same optimum, fewer iterations).
    """

    def __init__(self, alpha: float, half_life_days: float, warm: "TeamStrength | None" = None):
        self.alpha, self.hl = float(alpha), float(half_life_days)
        self.warm = warm

    def fit(self, sides: pd.DataFrame, at: pd.Timestamp):
        rows = sides.loc[(sides["date"] < at) & sides["y"].notna()]
        age = (at - rows["date"]).dt.days.to_numpy(float)
        w = np.power(0.5, age / self.hl)
        keep = w > 1e-3                                   # older matches carry no weight
        rows, w = rows.loc[keep], w[keep]
        self.teams = {t: i for i, t in enumerate(pd.unique(np.concatenate([rows["att"], rows["dfn"]])))}
        self.bases = {b: i for i, b in enumerate(sorted(pd.unique(rows["league"] + "|" + rows["venue"])))}
        X = self._design(rows)
        y = rows["y"].to_numpy(float)
        w = w / w.sum()
        T, B = len(self.teams), len(self.bases)
        pen = np.r_[np.ones(2 * T), np.zeros(B)]
        theta0 = np.zeros(2 * T + B)
        base_rate = rows.groupby(rows["league"] + "|" + rows["venue"])["y"].mean()
        for b, i in self.bases.items():
            theta0[2 * T + i] = np.log(max(base_rate.get(b, 5.0), 0.1))
        if self.warm is not None:
            for t, i in self.teams.items():
                j = self.warm.teams.get(t)
                if j is not None:
                    theta0[i] = self.warm.coef[j]
                    theta0[T + i] = self.warm.coef[len(self.warm.teams) + j]
        alpha = self.alpha
        Xt = X.T.tocsr()

        def f(th):
            eta = np.clip(X @ th, -10, 10)
            mu = np.exp(eta)
            obj = float(np.sum(w * (mu - y * eta))) + 0.5 * alpha * float(np.sum(pen * th * th))
            grad = Xt @ (w * (mu - y)) + alpha * pen * th
            return obj, grad

        res = optimize.minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
        self.coef, self.n_iter = res.x, int(res.nit)
        self.warm = None
        return self

    def _design(self, rows):
        n, T, B = len(rows), len(self.teams), len(self.bases)
        r = np.arange(n)
        ai = rows["att"].map(self.teams)
        di = rows["dfn"].map(self.teams)
        bi = (rows["league"] + "|" + rows["venue"]).map(self.bases)
        cols, rr = [], []
        for idx, off in ((ai, 0), (di, T), (bi, 2 * T)):
            ok = idx.notna().to_numpy()
            rr.append(r[ok]); cols.append(idx.to_numpy()[ok].astype(int) + off)
        rr, cols = np.concatenate(rr), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(rr)), (rr, cols)), shape=(n, 2 * T + B))

    def predict(self, sides):
        return np.exp(np.clip(self._design(sides) @ self.coef, -10, 10))

    def effects(self) -> pd.DataFrame:
        T = len(self.teams)
        return pd.DataFrame({"team": list(self.teams), "attack": self.coef[:T], "defence": self.coef[T:2 * T]})


def refit_dates(dates: pd.Series) -> list[pd.Timestamp]:
    """1st and 16th of every month covering ``dates``."""
    lo, hi = dates.min().normalize(), dates.max().normalize()
    out = []
    for m in pd.date_range(lo.replace(day=1), hi, freq="MS"):
        out += [m, m + pd.Timedelta(days=15)]
    return [d for d in out if d <= hi] or [lo]


def rolling_team_strength(df_all: pd.DataFrame, target: pd.DataFrame, alpha: float, hl: float,
                          min_history: int = 2000) -> pd.DataFrame:
    """Predicted home / away corners of every match in ``target`` from a model refitted every half
    month on the matches strictly before the refit date (never on the predicted matches)."""
    sides_all = _side_rows(df_all)
    tsides = _side_rows(target)
    dates = refit_dates(target["date"])
    pred = np.full(len(tsides), np.nan)
    prev = None
    for k, d in enumerate(dates):
        nxt = dates[k + 1] if k + 1 < len(dates) else pd.Timestamp.max
        sel = ((tsides["date"] >= d) & (tsides["date"] < nxt)).to_numpy()
        if k == 0:
            sel |= (tsides["date"] < d).to_numpy()
        if not sel.any() or (sides_all["date"] < d).sum() < min_history:
            continue
        prev = TeamStrength(alpha, hl, warm=prev).fit(sides_all, d)
        pred[sel] = prev.predict(tsides.loc[sel])
    tsides["mu"] = pred
    wide = tsides.pivot_table(index="match_id", columns="venue", values="mu", aggfunc="first")
    out = pd.DataFrame({"match_id": target["match_id"].to_numpy()})
    out["ts_home"] = out["match_id"].map(wide["H"])
    out["ts_away"] = out["match_id"].map(wide["A"])
    return out


# ---------------------------------------------------------------------------------------------- ensembles
def convex_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimising the Poisson deviance of the weighted mean."""
    k = preds.shape[1]

    def f(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return pdev(y, preds @ w)

    res = optimize.minimize(f, np.zeros(k), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4})
    w = np.exp(res.x - res.x.max())
    return w / w.sum()
