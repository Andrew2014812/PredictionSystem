"""Walk-forward predictions of the production models (cached).

For every season s in OUTER_SEASONS the four production XGBoost models (tuned parameters and tree counts of the
active production version, the same preprocessing and recency weights as src/modelling/training.py, via
experiments/v6_research/common.py) are fitted on all seasons < s and predict season s. The corner model's
negative-binomial size is the in-sample method-of-moments value on its training rows, as in production
(src/modelling/pipeline.py / scripts/main_rule_comparison.py).

Because the inner season of fold s (s-1 predicted by models fitted on < s-1) is exactly the outer prediction of
season s-1, four fits cover the folds 2023, 2024 and 2025 and their inner seasons.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from experiments.v6_research import common as V6
from src.modelling.training import negative_binomial_size

from . import common as C

V6.CACHE = C.CACHE                      # never write into the v6 study's folder

_orig_est = V6.XGBFit._est


def _est_limited(self, n, early=False):
    est = _orig_est(self, n, early)
    est.set_params(n_jobs=C.N_JOBS)
    return est


V6.XGBFit._est = _est_limited          # <= N_JOBS threads per model (shared machine)


def _block(train: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = score[V6.BASE_COLS].copy()
    out["y"] = V6.result_index(score)
    times = {}
    m = V6.production_model("result").fit(train, V6.target("result", train))
    p = m.predict(score)
    out["pH"], out["pD"], out["pA"] = p[:, 0], p[:, 1], p[:, 2]
    times["result"] = m.seconds
    for task, col in (("home_goals", "lh"), ("away_goals", "la")):
        m = V6.production_model(task).fit(train, V6.target(task, train))
        out[col] = m.predict(score)
        times[task] = m.seconds
    cl = V6.fold_corner_leagues(train)
    ctr = train.loc[train["total_corners"].notna() & train["league"].isin(cl)]
    m = V6.production_model("corners").fit(ctr, V6.target("corners", ctr))
    out["cmu"] = np.where(score["league"].isin(cl), m.predict(score), np.nan)
    nb = negative_binomial_size(ctr["total_corners"].to_numpy(float), m.predict(ctr))
    times["corners"] = m.seconds
    return out, {"seconds": times, "corner_leagues": cl, "train_rows": int(len(train)), "nb_size": nb,
                 "train_seasons": sorted(int(x) for x in train["season"].unique())}


def outer(season: int, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    f_df, f_info = C.CACHE / f"outer_{season}.parquet", C.CACHE / f"outer_{season}.json"
    if not refresh and f_df.exists() and f_info.exists():
        return pd.read_parquet(f_df), json.loads(f_info.read_text())
    t0 = time.time()
    df = V6.features()
    pred, info = _block(V6.before(df, season), V6.seasons(df, [season]))
    info["minutes"] = (time.time() - t0) / 60
    pred.to_parquet(f_df)
    f_info.write_text(json.dumps(info))
    C.log.info("outer %s: %d rows in %.1f min", season, len(pred), info["minutes"])
    return pred, info


def build_all(refresh: bool = False) -> dict:
    infos = {}
    for s in C.OUTER_SEASONS:
        _, infos[s] = outer(s, refresh)
    return infos


if __name__ == "__main__":
    C.setup_logging("base")
    build_all()
