"""Candidate algorithms of the alternative-ML study and the shared data / fitting helpers.

Every algorithm sees exactly the production inputs: the task's production feature set (``TaskTrainer``), the
production ``FeaturePreprocessor`` (one-hot league + numeric pass-through), production's recency weights of the
task (half-life of the production parameters) and, for corners, production's corner leagues. Only the estimator
changes. Linear models get a median imputer with missing-value indicators and standardisation (they cannot read
NaN); tree models read NaN natively.

Compute budget (declared before the run, see run.py): at most 2 threads per model, small fixed grids (2–4
configurations per algorithm and task), no Bayesian search.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_poisson_deviance

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.tasks import TASKS
from src.modelling.training import TaskTrainer, recency_weights

TASK_NAMES = ["result", "home_goals", "away_goals", "corners"]
OUT_COL = {"home_goals": "lam_home", "away_goals": "lam_away", "corners": "corner_mu"}
SEED = 42

# ------------------------------------------------------------------------------------------------ configurations
# name -> list of configs. "iters" = boosting rounds upper bound (early stopping on the validation season).
GRID: dict[str, list[dict]] = {
    "xgb": [{"cfg": "prod"}, {"cfg": "prod_es"}],
    "lgbm": [{"cfg": "l15_m100", "num_leaves": 15, "min_child_samples": 100},
             {"cfg": "l31_m200", "num_leaves": 31, "min_child_samples": 200},
             {"cfg": "l7_m300", "num_leaves": 7, "min_child_samples": 300}],
    "catboost": [{"cfg": "d4_l3", "depth": 4, "l2": 3.0}, {"cfg": "d6_l10", "depth": 6, "l2": 10.0}],
    "hgb": [{"cfg": "l15_m100", "max_leaf_nodes": 15, "min_samples_leaf": 100},
            {"cfg": "l31_m300", "max_leaf_nodes": 31, "min_samples_leaf": 300}],
    "rf": [{"cfg": "leaf50_f033", "min_samples_leaf": 50, "max_features": 0.33},
           {"cfg": "leaf150_fsqrt", "min_samples_leaf": 150, "max_features": "sqrt"}],
    "et": [{"cfg": "leaf50_f05", "min_samples_leaf": 50, "max_features": 0.5},
           {"cfg": "leaf150_fsqrt", "min_samples_leaf": 150, "max_features": "sqrt"}],
    "glm": [{"cfg": "c0.01", "C": 0.01}, {"cfg": "c0.1", "C": 0.1}, {"cfg": "c1", "C": 1.0}],
}
ALGO_TITLE = {"xgb": "XGBoost (production)", "lgbm": "LightGBM", "catboost": "CatBoost", "hgb": "HistGradientBoosting",
              "rf": "Random Forest", "et": "Extra Trees", "glm": "Regularised GLM (logistic / Poisson)"}
BOOSTING = {"lgbm", "catboost", "hgb", "xgb"}
MAX_ITERS = {"lgbm": 3000, "catboost": 3000, "hgb": 2000, "xgb": 3000}


class WeightedPipe:
    """Imputer + scaler + linear model with ``fit(X, y, sample_weight=)`` and predict(_proba) (ModelBundle API)."""

    def __init__(self, kind: str, strength: float):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, PoissonRegressor
        from sklearn.preprocessing import StandardScaler
        self.kind = kind
        self.imp = SimpleImputer(strategy="median", add_indicator=True)
        self.sc = StandardScaler()
        # C for the logistic model; for Poisson alpha = 1 / (C · n) -> same strength scale, chosen in the grid
        self.model = (LogisticRegression(C=strength, max_iter=3000) if kind == "classifier"
                      else PoissonRegressor(alpha=strength, max_iter=3000))

    def _x(self, X, fit=False):
        X = np.asarray(X, float)
        if fit:
            return self.sc.fit_transform(self.imp.fit_transform(X))
        return self.sc.transform(self.imp.transform(X))

    def fit(self, X, y, sample_weight=None):
        Z = self._x(X, fit=True)
        self.model.fit(Z, np.asarray(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._x(X))

    def predict(self, X):
        return self.model.predict(self._x(X))


def glm_strength(kind: str, C_: float) -> float:
    """Logistic: C as is. Poisson: alpha = 1e-3 / C (C = 0.01 -> alpha 0.1, strongest)."""
    return C_ if kind == "classifier" else 1e-3 / C_


def make(algo: str, task_name: str, cfg: dict, n_iter: int | None = None, base=None):
    """An unfitted estimator. ``n_iter`` fixes the boosting rounds (None -> the upper bound for early stopping)."""
    task = TASKS[task_name]
    clf = task.kind == "classifier"
    n = n_iter or MAX_ITERS.get(algo)
    if algo == "xgb":
        base = base or EN.production_bundles()[task_name]
        return EN._xgb(task, base.params, n_iter or base.n_estimators)
    if algo == "lgbm":
        import lightgbm as lgb
        kw = dict(n_estimators=n, learning_rate=0.03, num_leaves=cfg["num_leaves"],
                  min_child_samples=cfg["min_child_samples"], subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                  reg_lambda=5.0, n_jobs=C.N_JOBS, random_state=SEED, verbose=-1)
        return lgb.LGBMClassifier(objective="multiclass", **kw) if clf else lgb.LGBMRegressor(objective="poisson", **kw)
    if algo == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        kw = dict(iterations=n, learning_rate=0.05, depth=cfg["depth"], l2_leaf_reg=cfg["l2"], thread_count=C.N_JOBS,
                  random_seed=SEED, verbose=0, allow_writing_files=False)
        return CatBoostClassifier(loss_function="MultiClass", **kw) if clf else CatBoostRegressor(loss_function="Poisson", **kw)
    if algo == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        kw = dict(learning_rate=0.02, max_iter=n, max_leaf_nodes=cfg["max_leaf_nodes"],
                  min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=1.0, early_stopping=False,
                  random_state=SEED)
        return HistGradientBoostingClassifier(**kw) if clf else HistGradientBoostingRegressor(loss="poisson", **kw)
    if algo in ("rf", "et"):
        from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                                      RandomForestRegressor)
        kw = dict(n_estimators=300, min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                  n_jobs=C.N_JOBS, random_state=SEED)
        if algo == "rf":
            kw["max_samples"] = 0.5
            return RandomForestClassifier(**kw) if clf else RandomForestRegressor(**kw)
        return ExtraTreesClassifier(**kw) if clf else ExtraTreesRegressor(**kw)
    if algo == "glm":
        return WeightedPipe(task.kind, glm_strength(task.kind, cfg["C"]))
    raise ValueError(algo)


# ------------------------------------------------------------------------------------------------ data
def load_features() -> pd.DataFrame:
    from experiments.v6_3.replay import data as D
    return D.load_features()


def task_rows(task_name: str, df: pd.DataFrame) -> pd.DataFrame:
    task = TASKS[task_name]
    rows = df.loc[task.trainable(df)]
    if task_name == "corners":
        leagues = EN.production_bundles()["corners"].extra.get("leagues")
        if leagues:
            rows = rows.loc[rows["league"].isin(leagues)]
    return rows


def weights(task_name: str, rows: pd.DataFrame):
    hl = EN.production_bundles()[task_name].params.get("half_life_years", 0)
    return recency_weights(rows["date"], hl)


def score(task_name: str, y, pred) -> float:
    if TASKS[task_name].kind == "classifier":
        return float(log_loss(y, np.clip(pred, 1e-9, 1), labels=[0, 1, 2]))
    return float(mean_poisson_deviance(y, np.clip(pred, 1e-6, None)))


def _pred(task_name, model, X):
    return model.predict_proba(X) if TASKS[task_name].kind == "classifier" else model.predict(X)


def fit_with_validation(algo: str, task_name: str, cfg: dict, X_tr, y_tr, w, X_va, y_va) -> dict:
    """Fit one configuration on the training seasons; boosting rounds chosen by early stopping on the validation
    season. Returns validation predictions, score, rounds and seconds."""
    t0 = time.time()
    n_iter = None
    if algo == "xgb" and cfg["cfg"] == "prod":
        model = make("xgb", task_name, cfg)
        model.fit(X_tr, y_tr, sample_weight=w)
        n_iter = model.n_estimators
    elif algo == "xgb":
        model = make("xgb", task_name, cfg, n_iter=MAX_ITERS["xgb"])
        model.set_params(early_stopping_rounds=100)
        model.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)
        n_iter = int(model.best_iteration) + 1
    elif algo == "lgbm":
        import lightgbm as lgb
        model = make(algo, task_name, cfg)
        model.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        n_iter = int(model.best_iteration_ or model.n_estimators)
    elif algo == "catboost":
        model = make(algo, task_name, cfg)
        model.fit(X_tr, y_tr, sample_weight=w, eval_set=(X_va, y_va), early_stopping_rounds=100)
        n_iter = int(model.get_best_iteration()) + 1
        model.shrink(n_iter)
    elif algo == "hgb":
        model = make(algo, task_name, cfg, n_iter=25)
        model.set_params(warm_start=True)
        best, best_n, best_pred, bad = np.inf, 0, None, 0
        for n in range(25, MAX_ITERS["hgb"] + 1, 25):
            model.set_params(max_iter=n)
            model.fit(X_tr, y_tr, sample_weight=w)
            p = _pred(task_name, model, X_va)
            s = score(task_name, y_va, p)
            if s < best - 1e-6:
                best, best_n, best_pred, bad = s, n, p, 0
            else:
                bad += 1
                if bad >= 4:
                    break
        return {"pred": best_pred, "score": best, "n_iter": best_n, "seconds": time.time() - t0}
    else:
        model = make(algo, task_name, cfg)
        model.fit(X_tr, y_tr, sample_weight=w)
    p = _pred(task_name, model, X_va)
    return {"pred": p, "score": score(task_name, y_va, p), "n_iter": n_iter, "seconds": time.time() - t0}


def fit_bundle(algo: str, task_name: str, cfg: dict, n_iter: int | None, train: pd.DataFrame):
    """A production-like ModelBundle of the algorithm fitted on ``train`` (engine.fit_task: same features,
    preprocessing, recency weights and corner leagues as production)."""
    if algo == "xgb":
        return EN.fit_task(task_name, train)
    return EN.fit_task(task_name, train, estimator=make(algo, task_name, cfg, n_iter=n_iter))


def outputs(bundles: dict, matches: pd.DataFrame) -> pd.DataFrame:
    return EN.outputs_from_bundles({**EN.production_bundles(), **bundles}, matches)


def prep_xy(task_name: str, train: pd.DataFrame, score_df: pd.DataFrame):
    trainer = TaskTrainer(TASKS[task_name], train.columns)
    rows = task_rows(task_name, train)
    prep = FeaturePreprocessor(trainer.numeric, trainer.categorical).fit(rows)
    return prep, rows
