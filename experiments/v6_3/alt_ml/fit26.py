"""Stages 2 and 3 — fits for the season 2026/27 (heavy jobs, run through heavy.sh).

    static   for every algorithm its best configuration of stage 1 (lowest 2025/26 validation score per task) is
             fitted on every match played before 2026/27 (2021/22 … 2025/26, as production) and predicts 2026/27
             -> cache/out26_<algo>.parquet. XGBoost is production itself (Baseline B's model), not refitted.
    walk     monthly walk-forward (engine.walk_forward: matches of month M predicted by models fitted on everything
             played before the 1st of M) for XGBoost with production parameters and the two alternative algorithms
             with the best 2025/26 validation (chosen before 2026/27 is looked at) -> cache/wf_<algo>.parquet.

    experiments/v6_3/heavy.sh 5400 experiments/v6_3/alt_ml/logs/static.log .venv/bin/python -m experiments.v6_3.alt_ml.fit26 static
    experiments/v6_3/heavy.sh 7200 experiments/v6_3/alt_ml/logs/walk.log .venv/bin/python -m experiments.v6_3.alt_ml.fit26 walk
"""
from __future__ import annotations

import gc
import sys
import time

import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import engine as EN

from . import models as M

D = C.study_dirs("alt_ml")


def best_configs() -> dict:
    """{algo: {task: (cfg, n_iter, score)}} — the best configuration of every algorithm per task on 2025/26."""
    tune = C.read_json(D["results"] / "tune.json")
    out: dict = {}
    for task, t in tune.items():
        for key, v in t["configs"].items():
            algo = v["algo"]
            cur = out.setdefault(algo, {}).get(task)
            if cur is None or v["score"] < cur[2]:
                out[algo][task] = (v["cfg"], v["n_iter"], v["score"], key)
    return out


def validation_gain() -> dict:
    """Mean relative improvement of the validation score over XGBoost with production parameters, per algorithm."""
    tune = C.read_json(D["results"] / "tune.json")
    best = best_configs()
    gain = {}
    for algo, tasks in best.items():
        g = [(tune[t]["configs"]["xgb:prod"]["score"] - tasks[t][2]) / tune[t]["configs"]["xgb:prod"]["score"]
             for t in tasks]
        gain[algo] = float(sum(g) / len(g))
    return gain


def walk_algos() -> list[str]:
    g = validation_gain()
    alts = sorted([a for a in g if a != "xgb"], key=lambda a: -g[a])[:2]
    return ["xgb"] + alts


def static(log) -> None:
    f = M.load_features()
    m = pd.read_parquet(C.SHARED_CACHE / "matches.parquet")
    train = f.loc[(f["season"] < C.CURRENT) & f["played"].astype(bool)]
    for algo, tasks in best_configs().items():
        if algo == "xgb":
            continue
        path = D["cache"] / f"out26_{algo}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        bundles = {}
        for task in M.TASK_NAMES:
            cfg, n_iter, _, key = tasks[task]
            bundles[task] = M.fit_bundle(algo, task, cfg, n_iter, train)
            log.info("static %s %s (%s) fitted", algo, task, key)
        out = M.outputs(bundles, m)
        out.to_parquet(path)
        log.info("static %s: %.0fs", algo, time.time() - t0)
        del bundles
        gc.collect()


def walk(log) -> None:
    f = M.load_features()
    m = pd.read_parquet(C.SHARED_CACHE / "matches.parquet")
    best = best_configs()
    for algo in walk_algos():
        path = D["cache"] / f"wf_{algo}.parquet"
        if path.exists():
            continue
        t0 = time.time()

        def fit_fn(train, cut, algo=algo):
            out = {}
            for task in M.TASK_NAMES:
                cfg, n_iter, _, _ = best[algo][task]
                out[task] = M.fit_bundle(algo, task, cfg, n_iter, train)
            return out

        out = EN.walk_forward(f, m, fit_fn, log=log)
        out.to_parquet(path)
        log.info("walk-forward %s: %.0fs", algo, time.time() - t0)
        gc.collect()


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    threadpool_limits(C.N_JOBS)
    log = C.setup_logging("alt_ml.fit26")
    {"static": static, "walk": walk}[sys.argv[1]](log)
