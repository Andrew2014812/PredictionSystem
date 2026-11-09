"""Small walk-forward ensembles of the production models (the raw material of the uncertainty measures).

For every fold season s in SEASONS the four production XGBoost models (tuned parameters and tree counts of the
active production version, production preprocessing and recency weights — experiments/v6_research/common.py)
are refitted on all seasons < s and predict the played matches of season s, several times:

    seed   K = 8   the same training data, a different random seed (XGBoost row / column subsampling)
    boot   K = 8   a bootstrap resample of the training matches (multiplicity as a sample weight), own seed
    sub    K = 6   a 70 % subsample of the training matches without replacement, own seed

Every member gives pH, pD, pA (1X2 classifier), lh, la (goal rates) and cmu (corner mean, corner leagues only)
with the negative-binomial size of its own corner fit. Members are cached one by one
(cache/ens_<family>_<season>_<k>.parquet), so an interrupted run resumes. At most N_JOBS threads per model.

    python -m experiments.v6_2.uncertainty.ensemble seed      # one family per process
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from experiments.v6_research import common as V6
from src.modelling.training import negative_binomial_size

from . import common as C

V6.CACHE = C.CACHE                       # never write into another study's folder
OUT_COLS = ["match_id", "pH", "pD", "pA", "lh", "la", "cmu"]


def _model(task: str, seed: int) -> V6.XGBFit:
    m = V6.production_model(task)
    orig = m._est

    def est(n, early=False):
        e = orig(n, early)
        e.set_params(n_jobs=C.N_JOBS, random_state=int(seed))
        return e

    m._est = est
    return m


def _weights(family: str, n: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(C.SEED + 1000 * (1 + list(C.FAMILIES).index(family)) + k)
    if family == "seed":
        return np.ones(n)
    if family == "boot":
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    if family == "sub":
        w = np.zeros(n)
        w[rng.choice(n, int(round(C.SUB_FRACTION * n)), replace=False)] = 1.0
        return w
    raise ValueError(family)


def member(family: str, season: int, k: int, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = V6.before(df, season).reset_index(drop=True)
    score = V6.seasons(df, [season])
    w = _weights(family, len(train), k)
    keep = w > 0
    tr, wt = train.loc[keep].reset_index(drop=True), w[keep]
    seed = 7000 + 100 * list(C.FAMILIES).index(family) + k
    out = score[["match_id"]].copy()
    t0 = time.time()
    m = _model("result", seed).fit(tr, V6.target("result", tr), sample_weight=wt)
    p = m.predict(score)
    out["pH"], out["pD"], out["pA"] = p[:, 0], p[:, 1], p[:, 2]
    for task, col in (("home_goals", "lh"), ("away_goals", "la")):
        m = _model(task, seed).fit(tr, V6.target(task, tr), sample_weight=wt)
        out[col] = m.predict(score)
    cl = V6.fold_corner_leagues(train)
    cm = (tr["total_corners"].notna() & tr["league"].isin(cl)).to_numpy()
    ctr, cw = tr.loc[cm], wt[cm]
    m = _model("corners", seed).fit(ctr, V6.target("corners", ctr), sample_weight=cw)
    out["cmu"] = np.where(score["league"].isin(cl), m.predict(score), np.nan)
    # NB size on the member's own (weighted) training rows: repeat rows by integer weight
    rep = np.repeat(np.arange(len(ctr)), cw.astype(int))
    nb = negative_binomial_size(ctr["total_corners"].to_numpy(float)[rep], m.predict(ctr)[rep])
    info = {"family": family, "season": season, "k": k, "seed": seed, "train_rows": int(keep.sum()),
            "train_weight": float(wt.sum()), "nb_size": nb, "seconds": round(time.time() - t0, 1)}
    return out[OUT_COLS], info


def run_family(family: str, seasons=C.SEASONS) -> None:
    df = V6.features()
    for season in seasons:
        for k in range(C.FAMILIES[family]):
            path = C.CACHE / f"ens_{family}_{season}_{k}.parquet"
            meta = C.CACHE / f"ens_{family}_{season}_{k}.json"
            if path.exists() and meta.exists():
                continue
            out, info = member(family, season, k, df)
            out.to_parquet(path)
            meta.write_text(json.dumps(info))
            C.log.info("%s %s member %d: %d rows in %.0f s", family, season, k, len(out), info["seconds"])


def load(family: str, season: int) -> tuple[pd.DataFrame, list[dict]]:
    """Long table (match_id, member, pH, pD, pA, lh, la, cmu) and the member infos of one family and fold."""
    frames, infos = [], []
    for k in range(C.FAMILIES[family]):
        path = C.CACHE / f"ens_{family}_{season}_{k}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path).assign(member=k))
        infos.append(json.loads((C.CACHE / f"ens_{family}_{season}_{k}.json").read_text()))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), infos


if __name__ == "__main__":
    C.setup_logging("ens")
    fam = sys.argv[1]
    run_family(fam)
    (C.CACHE / f"ens_{fam}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
