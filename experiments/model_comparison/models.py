"""Candidate algorithms with one interface: tune on (train, valid) -> fit on all rows before
the evaluated season -> predict. Every candidate of a task gets the same rows, features
and fold; only algorithm-specific preprocessing differs (documented per class).

Hyperparameter budgets are small and comparable (a handful of settings per algorithm,
early stopping on the fold's validation season); XGBoost uses the production Hyperopt
search with the same number of evaluations as the others get settings.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_poisson_deviance

from src import config
from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer

from .data import Matrix

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def _score(kind: str, y, pred) -> float:
    if kind == "classifier":
        return float(log_loss(y, pred, labels=sorted(set(range(pred.shape[1])))))
    if kind == "binary":
        return float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return float(mean_poisson_deviance(y, np.clip(pred, 1e-6, None)))


class Candidate:
    name = "base"
    family = "base"

    def __init__(self, kind: str, numeric: list[str]):
        self.kind = kind                     # classifier (3 classes) | binary | count
        self.numeric = numeric
        self.params: dict = {}
        self.valid_pred = None

    # tune(train, valid, y_tr, y_va) -> sets self.params and self.valid_pred
    # fit(rows, y); predict(rows) -> probabilities (n x 3 / n) or expected counts


# -- gradient boosting ----------------------------------------------------------------
class XGBoostCandidate(Candidate):
    """Production pipeline (TaskTrainer: Hyperopt TPE, early stopping, recency weights)."""
    name, family = "XGBoost", "XGBoost"

    def __init__(self, kind, numeric, task: str, evals: int = 12, label: str | None = None):
        super().__init__(kind, numeric)
        self.task, self.evals = task, evals
        if label:
            self.name = label

    def tune(self, train, valid, y_tr, y_va):
        trainer = TaskTrainer(TASKS[self.task], list(train.columns))
        trainer.numeric = self.numeric
        params, n, _ = trainer.tune(train, valid, self.evals)
        self.params = {**params, "n_estimators": n}
        self.trainer = trainer
        bundle = trainer.fit(train, params, n)
        self.valid_pred = bundle.predict(trainer._rows(valid))

    def fit(self, rows, y):
        p = {k: v for k, v in self.params.items() if k != "n_estimators"}
        self.bundle = self.trainer.fit(rows, p, self.params["n_estimators"])
        return self

    def predict(self, rows):
        return self.bundle.predict(rows)


class _GBM(Candidate):
    grid: dict = {}

    def _settings(self):
        keys = list(self.grid)
        return [dict(zip(keys, v)) for v in itertools.product(*self.grid.values())]

    def tune(self, train, valid, y_tr, y_va):
        self.matrix = Matrix(self.numeric).fit(train)
        X_tr, X_va = self.matrix.tree(train), self.matrix.tree(valid)
        best = None
        for s in self._settings():
            model, n = self._fit_early(X_tr, y_tr, X_va, y_va, s)
            pred = self._predict(model, X_va)
            score = _score(self.kind, y_va, pred)
            if best is None or score < best[0]:
                best = (score, s, n, pred)
        _, self.params, n, self.valid_pred = best
        self.params = {**self.params, "n_estimators": n}

    def fit(self, rows, y):
        self.matrix = Matrix(self.numeric).fit(rows)
        self.model = self._fit_fixed(self.matrix.tree(rows), y, self.params)
        return self

    def predict(self, rows):
        return self._predict(self.model, self.matrix.tree(rows))

    def _predict(self, model, X):
        if self.kind == "classifier":
            return model.predict_proba(X)
        if self.kind == "binary":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)


class LightGBMCandidate(_GBM):
    """LightGBM: NaN handled natively; league one-hot as for XGBoost."""
    name, family = "LightGBM", "LightGBM"
    grid = {"num_leaves": [15, 31], "learning_rate": [0.05], "min_child_samples": [60, 200]}

    def _estimator(self, s, n):
        import lightgbm as lgb
        common = dict(n_estimators=n, num_leaves=s["num_leaves"], learning_rate=s["learning_rate"],
                      min_child_samples=s["min_child_samples"], subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0, random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1)
        if self.kind == "count":
            return lgb.LGBMRegressor(objective="poisson", **common)
        return lgb.LGBMClassifier(**common)

    def _fit_early(self, X_tr, y_tr, X_va, y_va, s):
        import lightgbm as lgb
        model = self._estimator(s, 3000)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return model, int(model.best_iteration_ or 3000)

    def _fit_fixed(self, X, y, params):
        s = {k: v for k, v in params.items() if k != "n_estimators"}
        return self._estimator(s, params["n_estimators"]).fit(X, y)


class CatBoostCandidate(_GBM):
    """CatBoost: NaN handled natively (Min), league one-hot as for the other trees."""
    name, family = "CatBoost", "CatBoost"
    grid = {"depth": [4, 6], "learning_rate": [0.06], "l2_leaf_reg": [3, 10]}

    def _estimator(self, s, n, early=False):
        from catboost import CatBoostClassifier, CatBoostRegressor
        common = dict(iterations=n, depth=s["depth"], learning_rate=s["learning_rate"], l2_leaf_reg=s["l2_leaf_reg"],
                      random_seed=config.RANDOM_STATE, verbose=False, thread_count=-1, allow_writing_files=False)
        if early:
            common["od_type"], common["od_wait"] = "Iter", 100
        if self.kind == "count":
            return CatBoostRegressor(loss_function="Poisson", **common)
        if self.kind == "binary":
            return CatBoostClassifier(loss_function="Logloss", **common)
        return CatBoostClassifier(loss_function="MultiClass", **common)

    def _fit_early(self, X_tr, y_tr, X_va, y_va, s):
        model = self._estimator(s, 2000, early=True)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        return model, int(model.get_best_iteration() or 2000) + 1

    def _fit_fixed(self, X, y, params):
        s = {k: v for k, v in params.items() if k != "n_estimators"}
        return self._estimator(s, params["n_estimators"]).fit(X, y)

    def _predict(self, model, X):
        if self.kind == "count":
            return model.predict(X, prediction_type="Exponent")
        return super()._predict(model, X)


# -- linear / statistical -----------------------------------------------------------------
class _Linear(Candidate):
    """Median-imputed, standardised features + missing indicators (fitted on train rows)."""
    grid: list = []

    def tune(self, train, valid, y_tr, y_va):
        self.matrix = Matrix(self.numeric).fit(train)
        X_tr, X_va = self.matrix.linear(train), self.matrix.linear(valid)
        best = None
        for c in self.grid:
            model = self._estimator(c).fit(X_tr, y_tr)
            pred = self._predict(model, X_va)
            score = _score(self.kind, y_va, pred)
            if best is None or score < best[0]:
                best = (score, c, pred)
        _, c, self.valid_pred = best
        self.params = {"regularisation": c}

    def fit(self, rows, y):
        self.matrix = Matrix(self.numeric).fit(rows)
        self.model = self._estimator(self.params["regularisation"]).fit(self.matrix.linear(rows), y)
        return self

    def predict(self, rows):
        return self._predict(self.model, self.matrix.linear(rows))


class LogisticCandidate(_Linear):
    name, family = "Logistic regression", "GLM"
    grid = [0.003, 0.01, 0.03, 0.1]

    def _estimator(self, c):
        return LogisticRegression(C=c, max_iter=2000)

    def _predict(self, model, X):
        p = model.predict_proba(X)
        return p if self.kind == "classifier" else p[:, 1]


class PoissonGLMCandidate(_Linear):
    name, family = "Poisson regression", "GLM"
    grid = [1e-3, 1e-2, 1e-1, 1.0]

    def _estimator(self, alpha):
        return PoissonRegressor(alpha=alpha, max_iter=1000)

    def _predict(self, model, X):
        return model.predict(X)


class NegBinGLMCandidate(_Linear):
    """Negative-binomial GLM (statsmodels, log link). The dispersion alpha is estimated from
    the train-row Pearson residuals of a Poisson fit; the L2 strength is tuned on validation."""
    name, family = "Negative binomial regression", "GLM"
    grid = [1e-3, 1e-2, 1e-1]

    def _estimator(self, alpha_l2):
        return _NB(alpha_l2)

    def _predict(self, model, X):
        return model.predict(X)


class _NB:
    def __init__(self, l2: float):
        self.l2 = l2

    def fit(self, X, y):
        import statsmodels.api as sm
        Xc = sm.add_constant(X, has_constant="add")
        mu = PoissonRegressor(alpha=self.l2, max_iter=1000).fit(X, y).predict(X)
        disp = max(float(np.mean(((y - mu) ** 2 - mu) / np.clip(mu ** 2, 1e-9, None))), 1e-4)
        self.dispersion = disp
        model = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=disp))
        self.result = model.fit_regularized(alpha=self.l2, L1_wt=0.0, maxiter=200)
        return self

    def predict(self, X):
        import statsmodels.api as sm
        return np.asarray(self.result.predict(sm.add_constant(X, has_constant="add")))


# -- ensemble ---------------------------------------------------------------------------
class Ensemble(Candidate):
    """Weighted average of already-tuned candidates; weights chosen on the fold's validation
    predictions (grid of convex combinations), probabilities / rates are averaged."""
    family = "Ensemble"

    def __init__(self, kind, members: list[Candidate], y_va):
        super().__init__(kind, [])
        self.members = members
        self.name = "Ensemble (" + " + ".join(m.name for m in members) + ")"
        steps = np.linspace(0, 1, 11)
        best = None
        for w in itertools.product(steps, repeat=len(members)):
            if abs(sum(w) - 1) > 1e-9:
                continue
            pred = sum(wi * m.valid_pred for wi, m in zip(w, members))
            score = _score(kind, y_va, pred)
            if best is None or score < best[0]:
                best = (score, w, pred)
        _, self.weights, self.valid_pred = best
        self.params = {"weights": dict(zip([m.name for m in members], map(float, self.weights)))}

    def predict(self, rows):
        return sum(w * m.predict(rows) for w, m in zip(self.weights, self.members))
