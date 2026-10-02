"""XGBoost training with Hyperopt tuning on a temporal validation season.

Hyperopt objective per task (lower is better):

* result classifier  -> multiclass log loss. Every downstream market and the
  EV calculation consume *probabilities*, so the tuning target must reward
  well-calibrated probabilities, not just the arg-max class (accuracy and
  macro F1 ignore how confident a forecast was).
* goal / corner regressors -> mean Poisson deviance, the proper loss for a
  count target whose prediction is used as a Poisson rate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.metrics import log_loss, mean_poisson_deviance
from xgboost import XGBClassifier, XGBRegressor

from .. import config
from ..features.sets import model_features
from .preprocessing import FeaturePreprocessor
from .tasks import Task

log = logging.getLogger(__name__)

SEARCH_SPACE = {
    "max_depth": hp.quniform("max_depth", 2, 6, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.15)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1), np.log(60)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1.0),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(0.5), np.log(30)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(5)),
    "gamma": hp.uniform("gamma", 0.0, 2.0),
    # Recency weighting: weight halves every `half_life_years` (0 = none).
    "half_life_years": hp.choice("half_life_years", [0, 1, 2, 4]),
}


@dataclass
class ModelBundle:
    """Everything needed to reproduce a prediction of one task."""
    task: Task
    preprocessor: FeaturePreprocessor
    model: object
    params: dict
    n_estimators: int
    trained_on: dict = field(default_factory=dict)    # seasons / dates / rows
    extra: dict = field(default_factory=dict)         # e.g. dispersion, leagues

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.feature_names

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Class probabilities (n x 3) or expected counts (n,)."""
        X = self.matrix(df)
        if self.task.kind == "classifier":
            return self.model.predict_proba(X)
        return self.model.predict(X)


def recency_weights(dates: pd.Series, half_life_years: float) -> np.ndarray | None:
    if not half_life_years:
        return None
    age_years = (dates.max() - dates).dt.days.to_numpy() / 365.25
    return np.power(0.5, age_years / half_life_years)


def _estimator(task: Task, params: dict, n_estimators: int, early_stopping: bool):
    cls = XGBClassifier if task.kind == "classifier" else XGBRegressor
    kwargs = dict(
        n_estimators=n_estimators,
        objective=task.objective,
        eval_metric=task.eval_metric,
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        **{k: v for k, v in params.items() if k != "half_life_years"},
    )
    if early_stopping:
        kwargs["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS
    return cls(**kwargs)


def _clean_params(params: dict) -> dict:
    params = dict(params)
    params["max_depth"] = int(params["max_depth"])
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in params.items()}


class TaskTrainer:
    def __init__(self, task: Task, columns: list[str]):
        self.task = task
        self.numeric, self.categorical = model_features(task.name, columns)

    def _rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.task.trainable(df)]

    def _score(self, y: np.ndarray, pred: np.ndarray) -> float:
        if self.task.kind == "classifier":
            return float(log_loss(y, pred, labels=[0, 1, 2]))
        return float(mean_poisson_deviance(y, np.clip(pred, 1e-6, None)))

    def _predict(self, model, X) -> np.ndarray:
        return model.predict_proba(X) if self.task.kind == "classifier" else model.predict(X)

    def tune(self, train: pd.DataFrame, valid: pd.DataFrame, max_evals: int) -> tuple[dict, int, list]:
        """Hyperopt TPE search; returns (params, n_estimators, trial log)."""
        train, valid = self._rows(train), self._rows(valid)
        prep = FeaturePreprocessor(self.numeric, self.categorical).fit(train)
        X_tr, X_va = prep.transform(train), prep.transform(valid)
        y_tr, y_va = self.task.target_values(train), self.task.target_values(valid)
        trial_log: list[dict] = []

        def objective(params):
            params = _clean_params(params)
            model = _estimator(self.task, params, config.MAX_ESTIMATORS, early_stopping=True)
            model.fit(X_tr, y_tr, sample_weight=recency_weights(train["date"], params["half_life_years"]),
                      eval_set=[(X_va, y_va)], verbose=False)
            score = self._score(y_va, self._predict(model, X_va))
            n_trees = int(model.best_iteration) + 1
            trial_log.append({"score": score, "n_estimators": n_trees, **params})
            return {"loss": score, "status": STATUS_OK, "n_estimators": n_trees}

        trials = Trials()
        started = time.time()
        best = fmin(objective, SEARCH_SPACE, algo=tpe.suggest, max_evals=max_evals, trials=trials,
                    rstate=np.random.default_rng(config.RANDOM_STATE), show_progressbar=False)
        params = _clean_params(space_eval(SEARCH_SPACE, best))
        n_estimators = trials.best_trial["result"]["n_estimators"]
        log.info("%s: best validation score %.4f after %d trials (%.0fs)", self.task.name,
                 trials.best_trial["result"]["loss"], max_evals, time.time() - started)
        return params, n_estimators, trial_log

    def fit(self, df: pd.DataFrame, params: dict, n_estimators: int) -> ModelBundle:
        """Fit the final model with fixed parameters on ``df``."""
        rows = self._rows(df)
        prep = FeaturePreprocessor(self.numeric, self.categorical).fit(rows)
        model = _estimator(self.task, params, n_estimators, early_stopping=False)
        model.fit(prep.transform(rows), self.task.target_values(rows),
                  sample_weight=recency_weights(rows["date"], params["half_life_years"]))
        means = rows[self.numeric].mean(numeric_only=True)
        return ModelBundle(
            task=self.task, preprocessor=prep, model=model, params=params, n_estimators=n_estimators,
            extra={"feature_means": {k: float(v) for k, v in means.items() if v == v}},
            trained_on={
                "seasons": sorted(int(s) for s in rows["season"].unique()),
                "date_from": str(rows["date"].min().date()),
                "date_to": str(rows["date"].max().date()),
                "rows": int(len(rows)),
            },
        )


def negative_binomial_size(y: np.ndarray, mu: np.ndarray) -> float | None:
    """Method-of-moments size ``r`` of NB(mu, r): Var = mu + mu^2 / r.

    Returns None when the data are not over-dispersed (Poisson is enough).
    """
    excess = np.mean((y - mu) ** 2 - mu)
    if excess <= 0:
        return None
    return float(np.mean(mu ** 2) / excess)


def corner_leagues(df: pd.DataFrame) -> list[str]:
    """Leagues with enough corner statistics to train and offer the market."""
    played = df.loc[df["played"].astype(bool)]
    coverage = played.groupby("league")["total_corners"].apply(lambda s: s.notna().mean())
    return sorted(coverage.loc[coverage >= config.MIN_STAT_COVERAGE].index)
