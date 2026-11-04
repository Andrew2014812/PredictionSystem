"""Fit every mean model of one fold; inner (train < s-1 -> s-1) and outer (train < s -> s) predictions.

Each model's predictions are cached in cache/pred_<season>_<model>.parquet (columns: match_id, part
['inner' | 'outer'], mu and, for side models, mu_home / mu_away), with its parameters and fit time in
cache/pred_<season>_<model>.json, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.modelling.registry import version_metadata

from . import data as D
from . import models as M

log = D.log
PROD_VERSION = "prod_20261012"
TS_GRID = [(a, hl) for a in (1e-3, 3e-3, 1e-2, 3e-2) for hl in (180, 365, 730)]
TUNE_EVALS = 20

MODELS = ["xgb_current", "xgb_enriched", "xgb_tuned", "xgb_ts", "lgbm", "catboost", "poisson_glm", "nb_glm",
          "team_strength", "xgb_home_away", "naive_league", "xgb_ts_offset"]


def prod_params() -> tuple[dict, int]:
    info = version_metadata(PROD_VERSION)["tasks"]["corners"]
    return dict(info["params"]), int(info["n_estimators"])


# ---------------------------------------------------------------------------------------------- team strength
def ts_path(alpha, hl):
    return D.CACHE / f"ts_{alpha:g}_{hl}.parquet"


def team_strength_grid() -> dict:
    """Rolling half-monthly refits for every grid point over 2022/23 .. 2026/27 (cached)."""
    df = D.with_corners(D.frame())
    target = df                       # 2021/22 from its first half month with >= 1000 matches of history
    scores = {}
    for alpha, hl in TS_GRID:
        path = ts_path(alpha, hl)
        if not path.exists():
            t0 = time.time()
            M.rolling_team_strength(df, target, alpha, hl).to_parquet(path)
            log.info("team strength alpha=%g hl=%d done in %.0fs", alpha, hl, time.time() - t0)
        r = pd.read_parquet(path).merge(target[["match_id", "season", "total_corners"]], on="match_id")
        r["mu"] = r["ts_home"] + r["ts_away"]
        r = r.loc[r["season"] >= 2022]
        scores[f"{alpha:g}|{hl}"] = {str(s): M.pdev(g["total_corners"], g["mu"]) for s, g in r.groupby("season")}
    return scores


def ts_choice(scores: dict, season: int) -> tuple[float, int]:
    """Grid point with the lowest Poisson deviance on the INNER season s-1."""
    key = min(scores, key=lambda k: scores[k][str(season - 1)])
    a, hl = key.split("|")
    return float(a), int(hl)


def ts_features(alpha, hl) -> pd.DataFrame:
    r = pd.read_parquet(ts_path(alpha, hl))
    r["ts_total"] = r["ts_home"] + r["ts_away"]
    return r


# ---------------------------------------------------------------------------------------------- one fold
def _out(rows, mu, part, mh=None, ma=None):
    o = pd.DataFrame({"match_id": rows["match_id"].to_numpy(), "part": part, "mu": np.asarray(mu, float)})
    if mh is not None:
        o["mu_home"], o["mu_away"] = np.asarray(mh, float), np.asarray(ma, float)
    return o


def fit_model(name: str, season: int, sets: dict, ts_scores: dict) -> tuple[pd.DataFrame, dict]:
    f = D.fold_rows(season)
    tr_in, va_in, tr_out, te = f["inner_train"], f["inner_valid"], f["outer_train"], f["test"]
    y = "total_corners"
    params, n_prod = prod_params()
    info: dict = {}
    t0 = time.time()

    if name == "xgb_current":
        m = M.XGB(sets["prod"], params, n_prod)
        mi = m.fit(tr_in, tr_in[y]).predict(va_in)
        mo = m.fit(tr_out, tr_out[y]).predict(te)
        info = {"n_trees": n_prod, "n_features": len(sets["prod"]), "params": params}
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name in ("xgb_enriched", "xgb_ts"):
        num = list(sets["enriched"])
        if name == "xgb_ts":
            a, hl = ts_choice(ts_scores, season)
            feats = ts_features(a, hl)
            info["team_strength"] = {"alpha": a, "half_life_days": hl}
            tr_in, va_in, tr_out, te = (d.merge(feats, on="match_id", how="left") for d in (tr_in, va_in, tr_out, te))
            num += ["ts_home", "ts_away", "ts_total"]
        m = M.XGB(num, params, 1)
        m.early_trees(tr_in, tr_in[y], va_in, va_in[y])
        mi = m.fit(tr_in, tr_in[y]).predict(va_in)
        mo = m.fit(tr_out, tr_out[y]).predict(te)
        info.update({"n_trees": m.n, "n_features": len(num)})
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name == "xgb_tuned":
        num = sets["enriched"]
        p, n, trials = M.tune_xgb(tr_in, tr_in[y], va_in, va_in[y], num, TUNE_EVALS)
        m = M.XGB(num, p, n)
        mi = m.fit(tr_in, tr_in[y]).predict(va_in)
        mo = m.fit(tr_out, tr_out[y]).predict(te)
        info = {"params": p, "n_trees": n, "n_features": len(num), "evals": TUNE_EVALS,
                "best_inner_deviance": min(t["score"] for t in trials), "trials": trials}
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name in ("lgbm", "catboost"):
        m = (M.LGBM if name == "lgbm" else M.CatB)(sets["enriched"]).tune(tr_in, tr_in[y], va_in, va_in[y])
        mi = m.fit(tr_in, tr_in[y]).predict(va_in)
        mo = m.fit(tr_out, tr_out[y]).predict(te)
        info = {"params": m.params, "n_features": len(sets["enriched"])}
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name in ("poisson_glm", "nb_glm"):
        m = (M.PoissonGLM if name == "poisson_glm" else M.NBGLM)(sets["compact"])
        m.tune(tr_in, tr_in[y], va_in, va_in[y])
        mi = m.fit(tr_in, tr_in[y]).predict(va_in)
        size_in = getattr(m, "size", None)
        mo = m.fit(tr_out, tr_out[y]).predict(te)
        info = {"params": m.params, "n_features": len(sets["compact"])}
        if name == "nb_glm":
            info["size_inner_fit"], info["size_outer_fit"] = size_in, m.size
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name == "team_strength":
        a, hl = ts_choice(ts_scores, season)
        feats = ts_features(a, hl).set_index("match_id")
        info = {"alpha": a, "half_life_days": hl, "grid_inner_deviance": {k: v[str(season - 1)]
                                                                         for k, v in ts_scores.items()}}
        out = []
        for rows, part in ((va_in, "inner"), (te, "outer")):
            r = feats.loc[rows["match_id"]]
            out.append(_out(rows, r["ts_total"], part, r["ts_home"], r["ts_away"]))

    elif name == "xgb_home_away":
        side = {}
        for col in ("home_corners", "away_corners"):
            m = M.XGB(sets["enriched"], params, 1)
            m.early_trees(tr_in, tr_in[col], va_in, va_in[col])
            side[col] = (m.fit(tr_in, tr_in[col]).predict(va_in), m.fit(tr_out, tr_out[col]).predict(te), m.n)
        h, a = side["home_corners"], side["away_corners"]
        info = {"n_trees_home": h[2], "n_trees_away": a[2], "n_features": len(sets["enriched"])}
        out = [_out(va_in, h[0] + a[0], "inner", h[0], a[0]), _out(te, h[1] + a[1], "outer", h[1], a[1])]

    elif name == "xgb_ts_offset":
        a, hl = ts_choice(ts_scores, season)
        feats = ts_features(a, hl)
        info["team_strength"] = {"alpha": a, "half_life_days": hl}
        tr_in, va_in, tr_out, te = (d.merge(feats, on="match_id", how="left") for d in (tr_in, va_in, tr_out, te))

        def offset(rows, train):
            # matches before the first refit (start of 2021/22) have no team-strength value: league mean
            lm = train.groupby("league")[y].mean()
            return rows["ts_total"].fillna(rows["league"].map(lm)).fillna(train[y].mean()).to_numpy(float)
        m = M.XGBOffset(sets["enriched"], params, 1)
        m.early_trees(tr_in, tr_in[y], va_in, va_in[y], offset(tr_in, tr_in), offset(va_in, tr_in))
        mi = m.fit(tr_in, tr_in[y], offset(tr_in, tr_in)).predict(va_in, offset(va_in, tr_in))
        mo = m.fit(tr_out, tr_out[y], offset(tr_out, tr_out)).predict(te, offset(te, tr_out))
        info.update({"n_trees": m.n, "n_features": len(sets["enriched"])})
        out = [_out(va_in, mi, "inner"), _out(te, mo, "outer")]

    elif name == "naive_league":
        out = []
        for train, rows, part in ((tr_in, va_in, "inner"), (tr_out, te, "outer")):
            lm = train.groupby("league")[y].mean()
            out.append(_out(rows, rows["league"].map(lm).fillna(train[y].mean()), part))
    else:
        raise ValueError(name)

    info["seconds"] = time.time() - t0
    return pd.concat(out, ignore_index=True), info


def fit_fold(season: int, models: list[str] | None = None, refresh: bool = False) -> None:
    df = D.frame()
    sets = D.feature_sets(df)
    ts_scores = team_strength_grid()
    for name in models or MODELS:
        p_path, i_path = D.CACHE / f"pred_{season}_{name}.parquet", D.CACHE / f"pred_{season}_{name}.json"
        if p_path.exists() and i_path.exists() and not refresh:
            continue
        pred, info = fit_model(name, season, sets, ts_scores)
        pred.to_parquet(p_path)
        i_path.write_text(json.dumps(D._clean(info)))
        o = pred.loc[pred["part"] == "outer"]
        y = D.fold_rows(season)["test"].set_index("match_id").loc[o["match_id"], "total_corners"]
        log.info("fold %d %-14s %.0fs  outer deviance %.4f", season, name, info["seconds"], M.pdev(y, o["mu"]))


def load_fold(season: int) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    preds, infos = {}, {}
    for name in MODELS:
        p_path, i_path = D.CACHE / f"pred_{season}_{name}.parquet", D.CACHE / f"pred_{season}_{name}.json"
        if p_path.exists() and i_path.exists():
            preds[name] = pd.read_parquet(p_path)
            infos[name] = json.loads(i_path.read_text())
    return preds, infos
