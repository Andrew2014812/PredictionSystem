"""Walk-forward refits of the production tasks with different sample-weight schemes (HEAVY job).

    experiments/v6_3/heavy.sh 10800 experiments/v6_3/recent_weighting/logs/fits_2026.log \
        .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026

For every season S (2025 = selection season, 2026 = evaluated season), every cut-off c (first match day of S, then
the 1st of every month; or every K played matches) and every scheme, each task is refitted with production's
features and hyper-parameters on the matches played BEFORE c and predicts the matches of S dated in
[c, next cut-off). The first cut-off model also predicts the whole season ("static" = trained once at the season
start, like production). Outputs: ``cache/wf_<S>_<scheme>.parquet`` and ``cache/static_<S>_<scheme>.parquet``
(one row per match of S, columns of the fitted tasks + refit_cutoff). Finished files are skipped (restartable).
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN

from . import lib as L

DIRS = C.study_dirs("recent_weighting")
TASKS = ["result", "home_goals", "away_goals", "corners"]


def _decay(dates: pd.Series, ref: pd.Timestamp, half_life: float) -> np.ndarray:
    if not half_life:
        return np.ones(len(dates))
    age = (ref - dates).dt.days.to_numpy() / 365.25
    return np.power(0.5, age / half_life)


def scheme_weights(scheme: str, task: str, rows: pd.DataFrame, season: int) -> tuple[np.ndarray, np.ndarray]:
    """(keep mask, weights) of the training rows for ``scheme``. Weights are rescaled so that their mean equals the
    mean production weight of the kept rows (the same effective sample mass, so XGBoost's regularisation in
    hessian units is comparable between schemes)."""
    ref = rows["date"].max()
    prod = _decay(rows["date"], ref, L.PROD_HALF_LIFE[task])
    keep = np.ones(len(rows), bool)
    if scheme == "prod" or scheme.startswith("k"):
        return keep, prod
    if scheme.startswith("hl"):
        w = _decay(rows["date"], ref, float(scheme[2:]))
    elif scheme.startswith("last"):
        w = prod * np.where(rows["season"].to_numpy() == season - 1, float(scheme[4:]), 1.0)
    elif scheme.startswith("cur"):
        w = prod * np.where(rows["season"].to_numpy() == season, float(scheme[3:]), 1.0)
    elif scheme.startswith("win"):
        keep = rows["season"].to_numpy() >= season - int(scheme[3:])
        w = prod.copy()
    else:
        raise ValueError(scheme)
    w = np.where(keep, w, 0.0)
    w = w * prod[keep].mean() / w[keep].mean()
    return keep, w


SCHEMES = ["prod", "hl0", "hl0.5", "hl1", "hl2", "hl4", "last2", "last3", "cur2", "cur4", "win2", "win3"]


def same_as_prod(scheme: str, task: str) -> bool:
    return scheme.startswith("hl") and float(scheme[2:]) == L.PROD_HALF_LIFE[task]


def fit_predict(task: str, train: pd.DataFrame, scheme: str, season: int, target: pd.DataFrame) -> np.ndarray:
    rows = train.loc[EN.TASKS[task].trainable(train)]
    prod = EN.production_bundles()[task]
    if task == "corners" and prod.extra.get("leagues"):
        rows = rows.loc[rows["league"].isin(prod.extra["leagues"])]
    keep, w = scheme_weights(scheme, task, rows, season)
    b = EN.fit_task(task, rows.loc[keep], weights=w[keep])
    p = b.predict(target)
    if task == "corners":
        p = np.where(target["league"].isin(prod.extra.get("leagues", [])), p, np.nan)
    return p


def run(season: int, schemes: list[str], cut_kind: str = "monthly", log=None) -> None:
    feats = L.features()
    m = L.season_rows(season)
    cuts = L.monthly_cutoffs(season) if cut_kind == "monthly" else L.k_match_cutoffs(season, int(cut_kind[1:]))
    for scheme in schemes:
        wf_path = DIRS["cache"] / f"wf_{season}_{scheme}.parquet"
        st_path = DIRS["cache"] / f"static_{season}_{scheme}.parquet"
        if wf_path.exists():
            continue
        t0 = time.time()
        wf = pd.DataFrame({"match_id": m["match_id"]})
        st = pd.DataFrame({"match_id": m["match_id"]})
        wf["refit_cutoff"] = pd.NaT
        for task in TASKS:
            cols = L.TASK_COLS[task]
            if same_as_prod(scheme, task):
                continue
            vals = np.full((len(m), len(cols)), np.nan)
            for i, cut in enumerate(cuts):
                end = cuts[i + 1] if i + 1 < len(cuts) else pd.Timestamp.max
                sel = ((m["date"] >= cut) & (m["date"] < end)).to_numpy()
                train = feats.loc[feats["played"].astype(bool) & (feats["date"] < cut)]
                target = m if i == 0 else m.loc[sel]
                p = fit_predict(task, train, scheme, season, target)
                p = p.reshape(len(target), -1)
                if i == 0:
                    sv = p.copy()
                    vals[sel] = p[sel]
                else:
                    vals[sel] = p
                wf.loc[sel, "refit_cutoff"] = cut
                del train
                gc.collect()
            for k, c in enumerate(cols):
                wf[c] = vals[:, k]
                st[c] = sv[:, k]
        wf.to_parquet(wf_path)
        if cut_kind == "monthly":
            st.to_parquet(st_path)
        if log:
            log.info("season %s scheme %s (%s, %d cut-offs): %.0fs", season, scheme, cut_kind, len(cuts),
                     time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--k", type=int, default=0, help="refit every K played matches (scheme k<K>)")
    a = ap.parse_args()
    log = C.setup_logging("recent_weighting.fits")
    if a.k:
        run(a.season, [f"k{a.k}"], f"k{a.k}", log)
    else:
        run(a.season, a.schemes.split(","), "monthly", log)
    log.info("done")


if __name__ == "__main__":
    main()
