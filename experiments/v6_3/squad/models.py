"""Fitting the production tasks with additional squad features, the squad-aware compact models, and the metrics.

* ``fit_plus`` — ``engine.fit_task`` with the production feature set PLUS extra columns (same parameters, same
  number of trees, same recency weights, same corner leagues). ``fit_plus(name, train, [])`` is identical to
  ``engine.fit_task(name, train)``.
* ``fit_compact`` — the squad-aware model: a compact feature set (Elo, league, rest, table, short form + squad
  features) with parameters chosen on an earlier season (``tune_compact``), never on the evaluated one.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.tasks import TASKS
from src.modelling.training import ModelBundle, TaskTrainer, recency_weights

TASK_NAMES = ["result", "home_goals", "away_goals", "corners"]


def squad_columns(columns, family: str) -> list[str]:
    """Squad feature columns of a family: 'sq' (dynamic) or 'pre' (pre-season); value_cov is coverage, not a feature."""
    pat = re.compile(rf"^(h|a|diff)_{family}_")
    return [c for c in columns if pat.search(c) and not c.endswith("value_cov")]


def corner_extra(columns) -> list[str]:
    """For corners: only the squad columns about wide / attacking players and attacking output."""
    keep = ("wide_value", "wide_n", "att_value", "xi_ga90", "xi_value", "squad_value")
    return [c for c in columns if c.endswith(keep)]


def _fit(name: str, rows: pd.DataFrame, numeric: list[str], categorical: list[str], params: dict, n_estimators: int,
         base: ModelBundle) -> ModelBundle:
    task = TASKS[name]
    prep = FeaturePreprocessor(numeric, categorical).fit(rows)
    X, y = prep.transform(rows), task.target_values(rows)
    w = recency_weights(rows["date"], params.get("half_life_years", 0))
    model = EN._xgb(task, params, n_estimators)
    model.fit(X, y, sample_weight=w)
    return ModelBundle(task=task, preprocessor=prep, model=model, params=params, n_estimators=n_estimators,
                       extra=copy.deepcopy(base.extra),
                       trained_on={"date_to": str(rows["date"].max().date()), "rows": int(len(rows)),
                                   "features": len(numeric)})


def task_rows(name: str, train: pd.DataFrame, base: ModelBundle) -> pd.DataFrame:
    task = TASKS[name]
    rows = train.loc[task.trainable(train)]
    if name == "corners" and base.extra.get("leagues"):
        rows = rows.loc[rows["league"].isin(base.extra["leagues"])]
    return rows


def fit_plus(name: str, train: pd.DataFrame, extra: list[str], base: ModelBundle | None = None) -> ModelBundle:
    base = base or EN.production_bundles()[name]
    trainer = TaskTrainer(TASKS[name], train.columns)
    rows = task_rows(name, train, base)
    numeric = list(trainer.numeric) + [c for c in extra if c not in trainer.numeric]
    return _fit(name, rows, numeric, trainer.categorical, dict(base.params), base.n_estimators, base)


# ------------------------------------------------------------------------------------------------ squad-aware
COMPACT = [r"^[ha]_elo$", r"^elo_(diff|exp_home)$", r"^[ha]_rest_days$", r"^diff_rest_days$",
           r"^[ha]_(season_ppg|position_norm|season_gd)$", r"^[ha]_form5_(ppg|gd|gf|ga)$",
           r"^league_(home_win_rate|draw_rate|avg_home_goals|avg_away_goals|over25_rate|btts_rate)$",
           r"^league_avg_(home_|away_)?corners$", r"^[ha]_form10_corners_(for|against)$"]
GRID = [{"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 40, "subsample": 0.7, "colsample_bytree": 0.6},
        {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8}]


def compact_columns(columns, squad: list[str]) -> list[str]:
    pats = [re.compile(p) for p in COMPACT]
    return [c for c in columns if any(p.search(c) for p in pats)] + list(squad)


def tune_compact(name: str, train: pd.DataFrame, valid: pd.DataFrame, squad: list[str]) -> dict:
    """Pick one of GRID and the number of trees (early stopping) on ``valid`` (an earlier season)."""
    from xgboost import XGBClassifier, XGBRegressor
    base = EN.production_bundles()[name]
    task = TASKS[name]
    tr, va = task_rows(name, train, base), task_rows(name, valid, base)
    numeric = compact_columns(train.columns, squad)
    prep = FeaturePreprocessor(numeric, ["league"]).fit(tr)
    Xtr, Xva = prep.transform(tr), prep.transform(va)
    ytr, yva = task.target_values(tr), task.target_values(va)
    best, log = None, []
    for g in GRID:
        cls = XGBClassifier if task.kind == "classifier" else XGBRegressor
        m = cls(n_estimators=2000, objective=task.objective, eval_metric=task.eval_metric, tree_method="hist",
                random_state=42, n_jobs=C.N_JOBS, early_stopping_rounds=60, **g)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        score = float(m.best_score)
        log.append({**g, "score": score, "n_estimators": int(m.best_iteration) + 1})
        if best is None or score < best["score"]:
            best = log[-1]
    return {"best": best, "grid": log, "features": len(numeric)}


def fit_compact(name: str, train: pd.DataFrame, squad: list[str], tuned: dict) -> ModelBundle:
    base = EN.production_bundles()[name]
    rows = task_rows(name, train, base)
    params = {k: v for k, v in tuned.items() if k not in ("score", "n_estimators")}
    return _fit(name, rows, compact_columns(train.columns, squad), ["league"], params, tuned["n_estimators"], base)


# ------------------------------------------------------------------------------------------------ metrics
def per_match_losses(out: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-match losses of an outputs table (match_id, p_home, p_draw, p_away, lam_home, lam_away, corner_mu).
    Totals and BTTS from independent Poisson goals (the same approximation for every method)."""
    t = truth.set_index("match_id")
    o = out.set_index("match_id").loc[t.index]
    res = t["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    P = np.clip(o[["p_home", "p_draw", "p_away"]].to_numpy(float), 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    hg, ag = t["home_goals"].to_numpy(float), t["away_goals"].to_numpy(float)
    lh, la = o["lam_home"].to_numpy(float), o["lam_away"].to_numpy(float)
    onehot = np.eye(3)[res]
    df = pd.DataFrame(index=t.index)
    df["ll_1x2"] = -np.log(P[np.arange(len(res)), res])
    df["brier_1x2"] = ((P - onehot) ** 2).sum(1)
    df["acc_1x2"] = (P.argmax(1) == res).astype(float)
    df["dev_home"] = 2 * (np.where(hg > 0, hg * np.log(np.maximum(hg, 1e-9) / lh), 0) - (hg - lh))
    df["dev_away"] = 2 * (np.where(ag > 0, ag * np.log(np.maximum(ag, 1e-9) / la), 0) - (ag - la))
    df["ae_goals"] = (np.abs(hg - lh) + np.abs(ag - la)) / 2
    p_over = 1 - poisson.cdf(2, lh + la)
    over = (hg + ag) > 2.5
    df["ll_ou25"] = -np.log(np.clip(np.where(over, p_over, 1 - p_over), 1e-9, 1))
    p_btts = (1 - np.exp(-lh)) * (1 - np.exp(-la))
    btts = (hg > 0) & (ag > 0)
    df["ll_btts"] = -np.log(np.clip(np.where(btts, p_btts, 1 - p_btts), 1e-9, 1))
    if "corner_mu" in o and "total_corners" in t:
        c, mu = t["total_corners"].to_numpy(float), o["corner_mu"].to_numpy(float)
        ok = np.isfinite(c) & np.isfinite(mu)
        df["dev_corners"] = np.where(ok, 2 * (np.where(c > 0, c * np.log(np.maximum(c, 1e-9) / np.maximum(mu, 1e-9)), 0)
                                              - (c - mu)), np.nan)
        df["ae_corners"] = np.where(ok, np.abs(c - mu), np.nan)
    return df


METRICS = ["ll_1x2", "brier_1x2", "acc_1x2", "dev_home", "dev_away", "ae_goals", "ll_ou25", "ll_btts",
           "dev_corners", "ae_corners"]


def compare_losses(new: pd.DataFrame, base: pd.DataFrame) -> dict:
    common = new.index.intersection(base.index)
    out = {}
    for m in METRICS:
        if m in new and m in base:
            a, b = new.loc[common, m].to_numpy(float), base.loc[common, m].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            sd, mean = float(d.std(ddof=1)), float(d.mean())
            n80 = int(np.ceil(((1.96 + 0.8416) * sd / abs(mean)) ** 2)) if mean != 0 else None
            out[m] = {"new": float(a[ok].mean()), "base": float(b[ok].mean()), **C.paired_mean_diff(a[ok], b[ok]),
                      "sd_paired": sd, "n_for_80pct_power": n80}
    return out


def r2(y, mu) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    ok = np.isfinite(y) & np.isfinite(mu)
    return float(1 - ((y[ok] - mu[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


__all__ = ["fit_plus", "fit_compact", "tune_compact", "squad_columns", "corner_extra", "per_match_losses",
           "compare_losses", "METRICS", "TASK_NAMES", "log_loss", "r2"]
