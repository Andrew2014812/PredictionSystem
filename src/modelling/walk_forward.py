"""Walk-forward evaluation with per-fold hyperparameter search.

For every evaluated season ``s`` (from the third available season on):

    1. tune   — Hyperopt on TRAIN = seasons < s-1, VALIDATION = season s-1;
    2. fit    — the tuned parameters on all seasons < s;
    3. score  — season s.

The hyperparameters of an early fold are therefore never chosen with data of
a later season (v4 reused the parameters tuned on 2024/25 for every fold,
which leaked information into the 2023/24 row).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .. import config
from .evaluation import classifier_metrics, regressor_metrics
from .pipeline import _subset
from .tasks import TASKS
from .training import TaskTrainer, corner_leagues

log = logging.getLogger(__name__)


def evaluated_seasons(features: pd.DataFrame) -> list[int]:
    seasons = sorted(int(s) for s in features.loc[features["played"].astype(bool), "season"].unique())
    return [s for s in seasons[2:] if s < config.current_season()]


def walk_forward(features: pd.DataFrame, tasks: tuple[str, ...] = ("result", "home_goals", "away_goals", "corners"),
                 max_evals: int = 15) -> list[dict]:
    rows = []
    cl = corner_leagues(features.loc[features["played"].astype(bool)])
    for s in evaluated_seasons(features):
        before = sorted(x for x in features["season"].unique() if x < s)
        for name in tasks:
            task = TASKS[name]
            tune_tr, tune_va = _subset(features, before[:-1]), _subset(features, [before[-1]])
            fit_rows, ev = _subset(features, before), _subset(features, [s])
            if name == "corners":
                tune_tr, tune_va, fit_rows, ev = (d.loc[d["league"].isin(cl)] for d in (tune_tr, tune_va, fit_rows, ev))
            trainer = TaskTrainer(task, features.columns)
            params, n_estimators, _ = trainer.tune(tune_tr, tune_va, max_evals)
            ev = ev.loc[task.trainable(ev)]
            pred = trainer.fit(fit_rows, params, n_estimators).predict(ev)
            y = task.target_values(ev).to_numpy()
            row = {"season": config.season_label(s), "task": name, "matches": int(len(ev)),
                   "tuned_on": config.season_label(before[-1])}
            if task.kind == "classifier":
                y_tr = task.target_values(fit_rows.loc[task.trainable(fit_rows)]).to_numpy()
                prior = np.bincount(y_tr, minlength=3) / len(y_tr)
                m = classifier_metrics(y, pred)
                row.update({k: m[k] for k in ("accuracy", "macro_f1", "log_loss", "brier")})
                row["draw_recall"] = m["recall"]["D"]
                row["baseline_log_loss"] = float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2]))
            else:
                m = regressor_metrics(y, pred)
                base = regressor_metrics(y, np.full(len(y), float(fit_rows[task.target].mean())))
                row.update({k: m[k] for k in ("mae", "rmse", "r2", "poisson_deviance")})
                row["baseline_poisson_deviance"] = base["poisson_deviance"]
            rows.append(row)
            log.info("walk-forward %s", row)
    return rows
