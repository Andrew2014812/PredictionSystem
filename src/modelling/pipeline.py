"""End-to-end (re)training with a temporal train / validation / test split.

    TRAIN       seasons FIRST_SEASON .. current-3   fit during tuning
    VALIDATION  season current-2                    Hyperopt objective only
    TEST        season current-1                    final, untouched evaluation
    CURRENT     season current                      live out-of-sample predictions

Two model versions are produced:

* ``backtest_<test season>`` - fitted on TRAIN + VALIDATION with the tuned
  parameters; evaluated on TEST and used to fill the prediction history of
  the test season (source = "backtest");
* ``prod_<date>`` - the same parameters refitted on every completed season
  (TRAIN + VALIDATION + TEST); becomes the active model for the current
  season.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config
from ..explain.shap_explainer import gain_importance, global_importance
from ..markets.distributions import (corners_distribution, prob_over, reconcile_with_1x2,
                                     score_matrix)
from ..prediction.engine import LeagueBaselines, predict_matches
from ..prediction.store import save_batch, settle
from .evaluation import (binary_metrics, calibration_table, classifier_metrics, market_benchmark,
                         prior_benchmark, regressor_metrics)
from .registry import save_version
from .tasks import RESULT_CLASSES, TASKS
from .training import ModelBundle, TaskTrainer, corner_leagues, negative_binomial_size

log = logging.getLogger(__name__)


def season_split(current: int | None = None) -> dict:
    current = current or config.current_season()
    validation = current - config.VALIDATION_OFFSET
    test = current - config.TEST_OFFSET
    return {
        "train": list(range(config.FIRST_SEASON, validation)),
        "validation": [validation],
        "test": [test],
        "current": current,
    }


def _subset(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    return df.loc[df["season"].isin(seasons) & df["played"].astype(bool)]


def _derived_market_metrics(bundles: dict[str, ModelBundle], test: pd.DataFrame) -> dict:
    """Quality of the markets read from the score matrix on the TEST season."""
    test = test.loc[test["home_goals"].notna()]
    p1x2 = bundles["result"].predict(test)
    lh, la = bundles["home_goals"].predict(test), bundles["away_goals"].predict(test)
    hg, ag = test["home_goals"].to_numpy(), test["away_goals"].to_numpy()
    out: dict = {}
    raw, rec = {k: [] for k in ("o15", "o25", "o35", "btts")}, {k: [] for k in ("o15", "o25", "o35", "btts")}
    top_hit = []
    for i in range(len(test)):
        base = score_matrix(lh[i], la[i])
        adj = reconcile_with_1x2(base, p1x2[i])
        ii, jj = np.indices(base.shape)
        for store, m in ((raw, base), (rec, adj)):
            store["o15"].append(m[ii + jj > 1.5].sum())
            store["o25"].append(m[ii + jj > 2.5].sum())
            store["o35"].append(m[ii + jj > 3.5].sum())
            store["btts"].append(m[(ii > 0) & (jj > 0)].sum())
        top = np.unravel_index(np.argmax(adj), adj.shape)
        top_hit.append(top[0] == hg[i] and top[1] == ag[i])
    outcomes = {"o15": hg + ag > 1.5, "o25": hg + ag > 2.5, "o35": hg + ag > 3.5,
                "btts": (hg > 0) & (ag > 0)}
    names = {"o15": "Over 1.5", "o25": "Over 2.5", "o35": "Over 3.5", "btts": "BTTS Yes"}
    for key, name in names.items():
        out[name] = {"poisson": binary_metrics(outcomes[key], np.array(raw[key])),
                     "reconciled": binary_metrics(outcomes[key], np.array(rec[key])),
                     "calibration": calibration_table(outcomes[key].astype(int), np.array(rec[key]))}
    out["Exact score (top 1)"] = {"hit_rate": float(np.mean(top_hit)), "n": len(top_hit)}

    corners = bundles.get("corners")
    if corners is not None:
        rows = test.loc[test["total_corners"].notna() & test["league"].isin(corners.extra["leagues"])]
        if len(rows):
            mu = corners.predict(rows)
            p = np.array([prob_over(corners_distribution(m, corners.extra.get("nb_size")),
                                    config.DEFAULT_CORNER_LINE) for m in mu])
            outcome = rows["total_corners"].to_numpy() > config.DEFAULT_CORNER_LINE
            out[f"Corners over {config.DEFAULT_CORNER_LINE:g}"] = {"reconciled": binary_metrics(outcome, p)}
    return out


def train_all(features: pd.DataFrame, max_evals: dict | None = None) -> dict:
    max_evals = max_evals or config.HYPEROPT_EVALS
    split = season_split()
    train = _subset(features, split["train"])
    valid = _subset(features, split["validation"])
    test = _subset(features, split["test"])
    full = pd.concat([train, valid, test])
    log.info("split: train %s (%d), validation %s (%d), test %s (%d)", split["train"], len(train),
             split["validation"], len(valid), split["test"], len(test))
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("not enough seasons for a train / validation / test split")

    leagues_with_corners = corner_leagues(full)
    backtest, production, task_meta = {}, {}, {}
    for name, task in TASKS.items():
        trainer = TaskTrainer(task, features.columns)
        tr, va, te, fu = train, valid, test, full
        if name == "corners":
            tr, va, te, fu = (d.loc[d["league"].isin(leagues_with_corners)] for d in (tr, va, te, fu))

        params, n_estimators, trials = trainer.tune(tr, va, max_evals[name])
        extra = {}
        if name == "corners":
            tuned = trainer.fit(tr, params, n_estimators)
            rows = va.loc[task.trainable(va)]
            extra = {"leagues": leagues_with_corners,
                     "nb_size": negative_binomial_size(rows[task.target].to_numpy(float),
                                                       tuned.predict(rows))}

        bt = trainer.fit(pd.concat([tr, va]), params, n_estimators)
        prod = trainer.fit(fu, params, n_estimators)
        for bundle in (bt, prod):
            bundle.extra.update(extra)
        backtest[name], production[name] = bt, prod

        rows = te.loc[task.trainable(te)]
        y_true = task.target_values(rows).to_numpy()
        pred = bt.predict(rows)
        if task.kind == "classifier":
            metrics = classifier_metrics(y_true, pred)
            benchmarks = {"prior": prior_benchmark(task.target_values(tr.loc[task.trainable(tr)]), y_true),
                          "bookmaker": market_benchmark(rows, y_true)}
            calibration = {c: calibration_table((y_true == i).astype(int), pred[:, i])
                           for i, c in enumerate(RESULT_CLASSES)}
        else:
            metrics = regressor_metrics(y_true, pred)
            benchmarks = {"mean": regressor_metrics(y_true, np.full(len(y_true), tr[task.target].mean()))}
            calibration = None

        best_trial = min(trials, key=lambda t: t["score"])
        task_meta[name] = {
            "title": task.title, "description": task.description, "kind": task.kind,
            "target": task.target, "objective": task.objective,
            "hyperopt_metric": "log_loss" if task.kind == "classifier" else "poisson_deviance",
            "hyperopt_evals": max_evals[name], "validation_score": best_trial["score"],
            "params": params, "n_estimators": n_estimators,
            "n_features": len(bt.feature_names), "features": bt.feature_names,
            "test_metrics": metrics, "benchmarks": benchmarks, "calibration": calibration,
            "trials": trials, "extra": {k: v for k, v in extra.items() if k != "feature_means"},
            "shap_global": global_importance(bt, rows),
            "gain_importance": gain_importance(bt),
            "trained_on": {"backtest": bt.trained_on, "production": prod.trained_on},
        }
        log.info("%s test metrics: %s", name,
                 {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    created = datetime.now().isoformat(timespec="seconds")
    periods = {k: [config.season_label(s) for s in v] for k, v in split.items() if k != "current"}
    periods["current"] = config.season_label(split["current"])
    common = {"created": created, "periods": periods, "tasks": task_meta,
              "derived_markets": _derived_market_metrics(backtest, test),
              "use_odds_features": config.USE_ODDS_FEATURES}

    bt_version = f"backtest_{config.season_code(split['test'][0])}"
    prod_version = f"prod_{datetime.now():%Y%m%d}"
    save_version(bt_version, backtest, {**common, "version": bt_version, "role": "backtest",
                                        "trained_to": str(pd.concat([train, valid])["date"].max().date())},
                 activate=False)
    save_version(prod_version, production, {**common, "version": prod_version, "role": "production",
                                            "trained_to": str(full["date"].max().date())},
                 activate=True)

    # Fill the prediction history of the test season with the backtest model.
    test_rows = features.loc[features["season"].isin(split["test"])]
    batch = predict_matches(test_rows, backtest, LeagueBaselines(features), bt_version, "backtest")
    save_batch(batch, replace_source="backtest")
    settle(features)
    log.info("backtest predictions: %d matches", len(batch.snapshots))
    return {"backtest": bt_version, "production": prod_version}
