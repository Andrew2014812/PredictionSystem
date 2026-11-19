"""Model fitting of the corners study (heavy: run through experiments/v6_3/heavy.sh, 2 threads).

    python -m experiments.v6_3.corners.fit select            # fit < 2025/26, predict 2025/26 (+ tree counts)
    python -m experiments.v6_3.corners.fit eval              # fit < 2026/27, predict 2026/27
    python -m experiments.v6_3.corners.fit wf --models a,b   # monthly refits inside 2026/27

Mean models (expected corners), every one trained only on matches played before its cut-off:

    xgb_prod     production XGBoost: production features and hyper-parameters (2-year recency half-life)
    xgb_stats    XGBoost, production parameters, production + new statistics features (ratings, EWMA, shots…)
    xgb_mkt      as xgb_stats + market profile of the match (pre-match Football-Data 1X2 / total 2.5 prices)
    lgbm_mkt     LightGBM, Poisson objective, stats + market features
    cat_mkt      CatBoost, Poisson objective, stats + market features
    xgb_ha       two XGBoost Poisson models (home corners, away corners), stats + market features; total = sum

Tree counts are chosen by early stopping in the select stage only (train 2021/22–2023/24, validation 2024/25)
and reused unchanged afterwards.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.corners import data as D
from src.modelling.preprocessing import FeaturePreprocessor
from src.modelling.training import recency_weights

XGB_MODELS = {"xgb_prod": "prod", "xgb_stats": "stats", "xgb_mkt": "stats_mkt"}
ALL = ["xgb_prod", "xgb_stats", "xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha"]
LGBM = dict(objective="poisson", learning_rate=0.02, num_leaves=15, min_child_samples=100, feature_fraction=0.6,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=C.N_JOBS, random_state=42)
CAT = dict(loss_function="Poisson", learning_rate=0.03, depth=5, l2_leaf_reg=6.0, thread_count=C.N_JOBS,
           random_seed=42, verbose=False, allow_writing_files=False)
MAX_TREES = 3000


def prod_bundle():
    from experiments.v6_3.replay.engine import production_bundles
    return production_bundles()["corners"]


def _weights(rows):
    return recency_weights(rows["date"], 2)


def _target(rows, target):
    return rows[target].to_numpy(float)


class Model:
    def __init__(self, kind: str, numeric: list[str], target: str = "total_corners", n: int | None = None):
        self.kind, self.numeric, self.target, self.n = kind, numeric, target, n

    def _est(self, n, early=False):
        if self.kind == "xgb":
            from xgboost import XGBRegressor
            p = {k: v for k, v in prod_bundle().params.items() if k != "half_life_years"}
            kw = dict(n_estimators=n, objective="count:poisson", eval_metric="poisson-nloglik", tree_method="hist",
                      random_state=42, n_jobs=C.N_JOBS, **p)
            if early:
                kw["early_stopping_rounds"] = 100
            return XGBRegressor(**kw)
        if self.kind == "lgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=n, **LGBM)
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=n, **CAT)

    def early(self, train, valid) -> int:
        prep = FeaturePreprocessor(self.numeric, ["league"]).fit(train)
        Xt, Xv = prep.transform(train), prep.transform(valid)
        yt, yv = _target(train, self.target), _target(valid, self.target)
        m = self._est(MAX_TREES, early=True)
        if self.kind == "xgb":
            m.fit(Xt, yt, sample_weight=_weights(train), eval_set=[(Xv, yv)], verbose=False)
            self.n = int(m.best_iteration) + 1
        elif self.kind == "lgbm":
            from lightgbm import early_stopping
            m.fit(Xt, yt, sample_weight=_weights(train), eval_set=[(Xv, yv)], callbacks=[early_stopping(100, verbose=False)])
            self.n = int(m.best_iteration_)
        else:
            m.fit(Xt, yt, sample_weight=_weights(train), eval_set=(Xv, yv), early_stopping_rounds=100)
            self.n = int(m.get_best_iteration()) + 1
        return self.n

    def fit(self, rows):
        self.prep = FeaturePreprocessor(self.numeric, ["league"]).fit(rows)
        self.model = self._est(self.n).fit(self.prep.transform(rows), _target(rows, self.target),
                                           sample_weight=_weights(rows))
        return self

    def predict(self, rows) -> np.ndarray:
        return np.asarray(self.model.predict(self.prep.transform(rows)), float)


def specs(sets: dict) -> dict[str, list[tuple]]:
    """name -> list of (kind, feature set, target) (two entries for the home / away pair)."""
    out = {k: [("xgb", v, "total_corners")] for k, v in XGB_MODELS.items()}
    out["lgbm_mkt"] = [("lgbm", "stats_mkt", "total_corners")]
    out["cat_mkt"] = [("cat", "stats_mkt", "total_corners")]
    out["xgb_ha"] = [("xgb", "stats_mkt", "home_corners"), ("xgb", "stats_mkt", "away_corners")]
    return out


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    leagues = prod_bundle().extra["leagues"]
    return df.loc[df["played"].astype(bool) & df["total_corners"].notna() & df["home_corners"].notna()
                  & df["league"].isin(leagues)]


def run_stage(stage: str, models: list[str], log) -> None:
    df, sets = D.load_frame()
    sp = specs(sets)
    trees_path = D.CACHE / "trees.json"
    trees = C.read_json(trees_path) if trees_path.exists() else {}
    tr_all = trainable(df)
    if stage == "select":
        cuts = [(pd.Timestamp("2025-07-01"), df.loc[df["season"] == D.SELECT])]
    elif stage == "eval":
        cuts = [(pd.Timestamp("2026-07-01"), df.loc[df["season"] == D.EVAL])]
    else:
        m26 = df.loc[df["season"] == D.EVAL]
        starts = [pd.Timestamp(x) for x in ("2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01")]
        cuts = []
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp.max
            cuts.append((s, m26.loc[(m26["date"] >= s) & (m26["date"] < end)]))
    frames = []
    for cut, target in cuts:
        train = tr_all.loc[tr_all["date"] < cut]
        out = pd.DataFrame({"match_id": target["match_id"].to_numpy()})
        out["cutoff"] = cut
        for name in models:
            t0 = time.time()
            if name == "xgb_prod" and stage == "eval":
                out["mu_xgb_prod"] = prod_bundle().predict(target)       # the production model itself
                out["mu_xgb_prod_refit"] = _fit_one(name, sp[name][0], sets, trees, train, target, stage, log)
                continue
            parts = [_fit_one(f"{name}:{t[2]}", t, sets, trees, train, target, stage, log) for t in sp[name]]
            if name == "xgb_ha":
                out["mu_h_xgb_ha"], out["mu_a_xgb_ha"] = parts
                out["mu_xgb_ha"] = parts[0] + parts[1]
            else:
                out[f"mu_{name}"] = parts[0]
            log.info("%s cut %s %s: %d train rows, %.0fs", stage, cut.date(), name, len(train), time.time() - t0)
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    path = D.CACHE / f"preds_{stage}.parquet"
    if path.exists() and stage == "wf":
        old = pd.read_parquet(path)
        keep = [c for c in old.columns if c not in res.columns or c in ("match_id", "cutoff")]
        res = old[keep].merge(res.drop(columns="cutoff"), on="match_id", how="outer")
    res.to_parquet(path)
    C.write_json(trees_path, trees)
    log.info("wrote %s %s", path, res.shape)


def _fit_one(key, spec, sets, trees, train, target, stage, log) -> np.ndarray:
    kind, fset, tgt = spec
    m = Model(kind, sets[fset], tgt)
    if key.startswith("xgb_prod"):
        m.n = prod_bundle().n_estimators
    elif key in trees:
        m.n = trees[key]
    else:
        if stage != "select":
            raise RuntimeError(f"no tree count for {key}: run the select stage first")
        tr = train.loc[train["season"] < D.SELECT - 1]
        va = train.loc[train["season"] == D.SELECT - 1]
        trees[key] = m.early(tr, va)
        log.info("early stopping %s: %d trees", key, trees[key])
    m.fit(train)
    return m.predict(target)


def wf_models_from_select(log) -> list[str]:
    """Models to refit monthly: production (control) + the mean model chosen on 2025/26 (all ensemble parts when
    an ensemble is chosen). The choice uses 2025/26 only, exactly as run.py makes it."""
    from experiments.v6_3.corners import evaluate as E
    from experiments.v6_3.corners import models as Mo
    from experiments.v6_3.corners import run as RUN
    frame, _ = D.load_frame()
    sel = E.scored(E.stage_table("select", frame, RUN.SEL_CUT))
    y = sel["total_corners"].to_numpy(float)
    w = Mo.convex_weights(sel[[f"mu_{k}" for k in E.ENS_PARTS]].to_numpy(float), y)
    sel = E.add_ensembles(sel, dict(zip(E.ENS_PARTS, w)))
    sel["mu_ens_w"] = RUN.crossfit_ensemble(sel)
    cand = ["xgb_prod", "xgb_stats", "xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha", "rating", "glm", "ens_w", "ens_eq"]
    dev = {m: D.poisson_dev(y, sel[f"mu_{m}"]) for m in cand}
    best = min(dev, key=dev.get)
    log.info("2025/26 deviance: %s -> best %s", {k: round(v, 5) for k, v in dev.items()}, best)
    if best.startswith("ens"):
        extra = [m for m in E.ENS_PARTS if m in ALL]
    else:
        extra = [best] if best in ALL else []
    return list(dict.fromkeys(["xgb_prod"] + extra))


if __name__ == "__main__":
    stage = sys.argv[1]
    models = ALL
    if "--models" in sys.argv:
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
    lg = C.setup_logging("corners.fit")
    t = time.time()
    if stage == "all":                    # select -> eval -> monthly walk-forward in one heavy slot
        run_stage("select", ALL, lg)
        run_stage("eval", ALL, lg)
        wfm = wf_models_from_select(lg)
        lg.info("walk-forward models: %s", wfm)
        run_stage("wf", wfm, lg)
    else:
        run_stage(stage, models, lg)
    lg.info("done %s in %.0fs", stage, time.time() - t)
