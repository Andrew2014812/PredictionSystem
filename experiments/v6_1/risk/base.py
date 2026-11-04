"""Out-of-sample predictions of the production models for one season (walk-forward, cached).

Season s is predicted by the production XGBoost models (tuned parameters and tree counts of the
active production version) fitted on the seasons before s - exactly ``_predict_block`` of
experiments/v6_research/common.py, only with at most 3 threads.

Inner / outer blocks of the v6 folds coincide: the INNER block of fold s (fit on < s-1, predict s-1)
is the OUTER block of fold s-1, so one block per season is enough (2022 ... 2025).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from experiments.v6_research import common as C

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
N_JOBS = 3

_orig_est = C.XGBFit._est


def _est(self, n, early=False):                 # same estimator, fewer threads (shared machine)
    return _orig_est(self, n, early).set_params(n_jobs=N_JOBS)


C.XGBFit._est = _est


def block(season: int, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    f_rows, f_info = CACHE / f"block_{season}.parquet", CACHE / f"block_{season}.json"
    if not refresh and f_rows.exists() and f_info.exists():
        return pd.read_parquet(f_rows), json.loads(f_info.read_text())
    t0 = time.time()
    df = C.features()
    rows, info = C._predict_block(C.before(df, season), C.seasons(df, [season]))
    info["wall_seconds"] = time.time() - t0
    rows.to_parquet(f_rows)
    f_info.write_text(json.dumps(info))
    return rows, info


if __name__ == "__main__":
    import sys
    C.setup_logging("risk.base")
    for s in [int(x) for x in sys.argv[1:]] or [2022, 2023, 2024, 2025]:
        r, i = block(s)
        print(s, len(r), round(i["wall_seconds"]), "s", flush=True)
