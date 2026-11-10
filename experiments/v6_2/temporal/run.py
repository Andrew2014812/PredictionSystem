"""Runs every experiment of the temporal-features study and writes results/.

    python -m experiments.v6_2.temporal.run           (hours on the shared machine: run in the background)

Steps (cached in cache/, safe to re-run):
    leakage    explicit leakage tests                                       -> results/leakage_test.json
    variants   feature-set variants T1 ... T9, R0 ... R2 (three seeds, walk-forward)
    signal     residual correlations of every candidate feature             -> results/signal.json
    market     form correction of the market price                          -> results/market_form.json
    evaluate   every variant on the same matches, Holm, decisions           -> results/variants.json
The production baseline predictions are shared with the league study (experiments/v6_2/leagues/cache).
"""
from __future__ import annotations

import time

from experiments.v6_2.leagues import base as LB
from experiments.v6_2.leagues import wf
from experiments.v6_2.odds_timing.common import environment

from . import common as C
from . import evaluate, leakage, market_form, signal, variants


def run_info(timings: dict) -> None:
    C.save_json("run_info.json", {
        "git_sha": C.git_sha(),
        "data_version": C.data_version([C.ROOT / "local_data" / "processed" / "features.parquet"]),
        "params": {"seeds": C.SEEDS, "prod_seed": C.PROD_SEED, "n_jobs": C.N_JOBS, "n_boot": C.N_BOOT,
                   "bootstrap_seed": C.SEED, "select": C.SELECT, "info": C.INFO, "current": C.CURRENT,
                   "production_version": wf.V6.prod_meta().get("version"),
                   "feature_counts": variants.feature_counts()},
        "library_versions": environment()["libraries"], "timings_seconds": timings,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def main() -> None:
    C.setup_logging("temporal")
    t = {}
    for name, fn in (("base", LB.build), ("leakage", leakage.run), ("variants", variants.build_all),
                     ("signal", signal.run), ("market_form", market_form.run), ("evaluate", evaluate.run)):
        t0 = time.time()
        fn()
        t[name] = round(time.time() - t0, 1)
        C.log.info("step %s done in %.0f s", name, t[name])
    run_info(t)
    print("RUN_DONE")


if __name__ == "__main__":
    main()
