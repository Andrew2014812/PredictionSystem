"""Stage 1 — light tuning on earlier seasons (heavy job, run through heavy.sh).

Train = seasons 2021/22 … 2024/25, validation = 2025/26 (never 2026/27). For each task and each configuration of
``models.GRID`` the model is fitted on the training seasons (boosting rounds by early stopping on 2025/26) and its
2025/26 predictions are stored (``cache/valid_<task>.parquet``) with the score (log loss / mean Poisson deviance),
the number of rounds and the seconds (``results/tune.json``). Resumable: finished configurations are skipped.

    experiments/v6_3/heavy.sh 10800 experiments/v6_3/alt_ml/logs/tune.log \
        .venv/bin/python -m experiments.v6_3.alt_ml.tune
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src.modelling.tasks import TASKS

from . import models as M

D = C.study_dirs("alt_ml")
TRAIN_TO, VALID = 2024, 2025


def run(tasks=None, algos=None) -> dict:
    log = C.setup_logging("alt_ml.tune")
    path = D["results"] / "tune.json"
    res = C.read_json(path) if path.exists() else {}
    f = M.load_features()
    f = f.loc[f["season"] <= VALID]
    for task_name in tasks or M.TASK_NAMES:
        task = TASKS[task_name]
        train = M.task_rows(task_name, f.loc[f["season"] <= TRAIN_TO])
        valid = M.task_rows(task_name, f.loc[f["season"] == VALID])
        prep, _ = M.prep_xy(task_name, f.loc[f["season"] <= TRAIN_TO], valid)
        X_tr, X_va = prep.transform(train), prep.transform(valid)
        y_tr, y_va = task.target_values(train).to_numpy(), task.target_values(valid).to_numpy()
        w = M.weights(task_name, train)
        cache = D["cache"] / f"valid_{task_name}.parquet"
        preds = pd.read_parquet(cache) if cache.exists() else pd.DataFrame({"match_id": valid["match_id"].to_numpy()})
        res.setdefault(task_name, {"train_rows": int(len(train)), "valid_rows": int(len(valid)),
                                   "features": int(X_tr.shape[1]), "configs": {}})
        for algo, cfgs in M.GRID.items():
            if algos and algo not in algos:
                continue
            for cfg in cfgs:
                key = f"{algo}:{cfg['cfg']}"
                if key in res[task_name]["configs"]:
                    continue
                log.info("%s %s ...", task_name, key)
                out = M.fit_with_validation(algo, task_name, cfg, X_tr, y_tr, w, X_va, y_va)
                p = np.asarray(out["pred"])
                if p.ndim == 2:
                    for i, c in enumerate("HDA"):
                        preds[f"{key}|{c}"] = p[:, i]
                else:
                    preds[key] = p
                res[task_name]["configs"][key] = {"algo": algo, "cfg": cfg, "score": out["score"],
                                                  "n_iter": out["n_iter"], "seconds": round(out["seconds"], 1)}
                log.info("%s %s: score %.5f, rounds %s, %.0fs", task_name, key, out["score"], out["n_iter"],
                         out["seconds"])
                preds.to_parquet(cache)
                C.write_json(path, res)
                gc.collect()
    return res


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    threadpool_limits(C.N_JOBS)          # BLAS / OpenMP pools too (linear models, HistGradientBoosting)
    t0 = time.time()
    args = sys.argv[1:]
    tasks = [a for a in args if a in M.TASK_NAMES] or None
    algos = [a for a in args if a in M.GRID] or None
    run(tasks, algos)
    print(f"done in {time.time() - t0:.0f}s")
