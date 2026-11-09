"""Reproducibility check of the reused v6.1 walk-forward predictions.

Refits the production models for one season on the CURRENT feature table (models fitted on all seasons < s,
n_jobs = 2) into a scratch folder and compares the predictions with the cached v6.1 file used by this study.
Writes results/cache_check.json.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import common as C


def run(season: int = 2024) -> dict:
    from experiments.v6_1.main import base
    cached = pd.read_parquet(C.CACHE / f"outer_{season}.parquet")
    scratch = C.CACHE / "verify"
    scratch.mkdir(exist_ok=True)
    t0 = time.time()
    df = base.V6.features()
    pred, info = base._block(base.V6.before(df, season), base.V6.seasons(df, [season]))
    pred.to_parquet(scratch / f"outer_{season}.parquet")
    a = cached.set_index("match_id")
    b = pred.set_index("match_id")
    common_ids = a.index.intersection(b.index)
    out = {"season": season, "rows_cached": int(len(a)), "rows_refit": int(len(b)), "common": int(len(common_ids)),
           "minutes": (time.time() - t0) / 60}
    for col in ("pH", "pD", "pA", "lh", "la", "cmu"):
        d = (a.loc[common_ids, col] - b.loc[common_ids, col]).abs()
        out[f"max_abs_diff_{col}"] = float(np.nanmax(d.to_numpy())) if len(d) else None
    C.save_json("cache_check.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("verify")
    print(run(int(sys.argv[1]) if len(sys.argv) > 1 else 2024))
