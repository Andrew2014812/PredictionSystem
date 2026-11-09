"""Dedicated ("direct") market models, walk-forward by season.

For every outer fold s (2022/23 ... 2026/27) each model is fitted on all played matches of seasons < s and
predicts the played matches of season s. Its tree count is chosen by early stopping in the previous fold
(fitted on seasons < s-1, stopped on season s-1; for s = 2022/23 on the last quarter of 2021/22 by date), so no
number of season s is used to build its own prediction.

Models (XGBoost, the preprocessing / estimator settings of production, src/modelling/training.py):

    btts, o15, o25, o35, dc_1x, dc_x2   binary:logistic   goal features (union of the production result,
                                                          home-goals and away-goals feature sets) + league
    tot                                 multi:softprob    total goals 0, 1, ..., 5, 6+          (7 classes)
    gd                                  multi:softprob    goal difference <=-3, -2, ..., 2, >=3  (7 classes)
    es                                  multi:softprob    exact score h, a <= 3 (16) + "other"  (17 classes)
    c85, c95, c105                      binary:logistic   production corner features, corner leagues only

Hyper-parameters are not tuned here: binary goal models use the tuned parameters of the production 1X2
classifier; the multi-class models the same with learning rate 0.03 (7 / 17 trees per round would otherwise
make them 7-17x slower than the 1X2 model); corner models the tuned parameters of the production corner model.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from experiments.v6_research import common as V6

from . import common as C

_orig_est = V6.XGBFit._est


def _est_limited(self, n, early=False):
    est = _orig_est(self, n, early)
    est.set_params(n_jobs=C.N_JOBS)
    return est


V6.XGBFit._est = _est_limited

ES_CLASSES = [(h, a) for h in range(4) for a in range(4)]          # + index 16 = other
GD_EDGES = (-3, -2, -1, 0, 1, 2, 3)                                 # class k <-> margin GD_EDGES[k] (clipped)
TOT_MAX = 6


def targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    hg, ag = df["home_goals"].to_numpy(int), df["away_goals"].to_numpy(int)
    t = hg + ag
    es = np.full(len(df), 16)
    small = (hg <= 3) & (ag <= 3)
    es[small] = hg[small] * 4 + ag[small]
    tc = df["total_corners"].to_numpy(float)
    return {"btts": ((hg > 0) & (ag > 0)).astype(int), "o15": (t > 1.5).astype(int), "o25": (t > 2.5).astype(int),
            "o35": (t > 3.5).astype(int), "dc_1x": (hg >= ag).astype(int), "dc_x2": (hg <= ag).astype(int),
            "tot": np.minimum(t, TOT_MAX), "gd": np.clip(hg - ag, -3, 3) + 3, "es": es,
            "c85": np.where(np.isnan(tc), -1, (tc > 8.5).astype(int)),
            "c95": np.where(np.isnan(tc), -1, (tc > 9.5).astype(int)),
            "c105": np.where(np.isnan(tc), -1, (tc > 10.5).astype(int))}


SPECS = {
    # name: (feature set, kind, n_classes)
    "btts": ("goals", "bin", 2), "o15": ("goals", "bin", 2), "o25": ("goals", "bin", 2), "o35": ("goals", "bin", 2),
    "dc_1x": ("goals", "bin", 2), "dc_x2": ("goals", "bin", 2),
    "tot": ("goals", "multi", TOT_MAX + 1), "gd": ("goals", "multi", 7), "es": ("goals", "multi", 17),
    "c85": ("corners", "bin", 2), "c95": ("corners", "bin", 2), "c105": ("corners", "bin", 2),
}


def feature_sets() -> dict[str, list[str]]:
    goals = []
    for task in ("result", "home_goals", "away_goals"):
        for c in V6.task_numeric(task):
            if c not in goals:
                goals.append(c)
    return {"goals": goals, "corners": V6.task_numeric("corners")}


def model_for(name: str, feats: dict, n: int = 100) -> V6.XGBFit:
    fset, kind, k = SPECS[name]
    if fset == "corners":
        params, _ = V6.prod_params("corners")
    else:
        params, _ = V6.prod_params("result")
        if kind == "multi":
            params = {**params, "learning_rate": 0.03}
    obj, metric = ("binary:logistic", "logloss") if kind == "bin" else ("multi:softprob", "mlogloss")
    return V6.XGBFit(feats[fset], params, n, obj, metric, classifier=True)


def _rows(df: pd.DataFrame, name: str, cl: list[str]) -> pd.DataFrame:
    if SPECS[name][0] == "corners":
        return df.loc[df["total_corners"].notna() & df["league"].isin(cl)]
    return df


def _ensure_classes(y: np.ndarray, k: int, rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """XGBoost needs every class 0..k-1 in the training labels (never binding here, checked)."""
    missing = sorted(set(range(k)) - set(np.unique(y)))
    if missing:
        raise ValueError(f"classes {missing} missing from the training labels")
    return rows, y


def tree_count(name: str, season: int, df: pd.DataFrame, feats: dict) -> int:
    """Early stopping on season - 1 with a model fitted on seasons < season - 1 (2021/22: last quarter)."""
    if season - 1 <= int(df["season"].min()):
        first = df.loc[df["season"] == season - 1].sort_values("date")
        cut = int(len(first) * 0.75)
        tr, va = first.iloc[:cut], first.iloc[cut:]
    else:
        tr, va = V6.before(df, season - 1), V6.seasons(df, [season - 1])
    cl = V6.fold_corner_leagues(tr)
    tr, va = _rows(tr, name, cl), _rows(va, name, cl)
    k = SPECS[name][2]
    y_tr, y_va = targets(tr)[name], targets(va)[name]
    _ensure_classes(y_tr, k, tr)
    m = model_for(name, feats)
    return m.early_stopped_trees(tr, y_tr, va, y_va)


def fold(season: int, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    path, ipath = C.CACHE / f"direct_{season}.parquet", C.CACHE / f"direct_{season}.json"
    if path.exists() and ipath.exists() and not refresh:
        return pd.read_parquet(path), json.loads(ipath.read_text())
    df = V6.features()
    feats = feature_sets()
    train, score = V6.before(df, season), V6.seasons(df, [season])
    cl = V6.fold_corner_leagues(train)
    out = pd.DataFrame({"match_id": score["match_id"].to_numpy()})
    info = {"models": {}, "corner_leagues": cl, "train_rows": int(len(train)),
            "n_features": {k: len(v) for k, v in feats.items()}}
    for name, (fset, kind, k) in SPECS.items():
        t0 = time.time()
        n = tree_count(name, season, df, feats)
        t_es = time.time() - t0
        tr = _rows(train, name, cl)
        y = targets(tr)[name]
        _ensure_classes(y, k, tr)
        m = model_for(name, feats, n).fit(tr, y)
        t1 = time.time()
        p = m.predict(score)
        lat = (time.time() - t1) / max(len(score), 1) * 1000
        if kind == "bin":
            col = np.asarray(p, float)
            if fset == "corners":
                col = np.where(score["league"].isin(cl).to_numpy(), col, np.nan)
            out[name] = col
        else:
            for j in range(k):
                out[f"{name}_{j}"] = p[:, j]
        size = len(pickle.dumps(m.model))
        info["models"][name] = {"trees": n, "fit_seconds": round(m.seconds, 2), "es_seconds": round(t_es, 2),
                                "predict_ms_per_match": round(lat, 4), "pickle_bytes": size,
                                "train_rows": int(len(tr)), "classes": k}
        C.log.info("fold %s %s: %d trees, fit %.1fs, es %.1fs", season, name, n, m.seconds, t_es)
    out.to_parquet(path)
    ipath.write_text(json.dumps(info))
    return out, info


def build_all(refresh: bool = False, seasons=C.FOLDS) -> dict:
    return {s: fold(s, refresh)[1] for s in seasons}


if __name__ == "__main__":
    C.setup_logging()
    which = [int(x) for x in sys.argv[1:]] or list(C.FOLDS)
    build_all(seasons=which)
    print("DONE")
