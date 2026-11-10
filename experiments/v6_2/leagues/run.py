"""Runs every experiment of the league-specialisation study and writes results/.

    python -m experiments.v6_2.leagues.run            (hours on the shared machine: run in the background)

Steps (each cached in cache/, safe to re-run):
    base       walk-forward production predictions 2022/23 ... 2026/27, three seeds, corners, prices
    profiles   league profiles and groups per fold                         -> results/profiles.json
    perf       per-league performance of the production model             -> results/perf.json
    models     refitted variants M1 ... M5 (three seeds)
    transfer   leave-one-league-out fits                                   -> results/transfer.json
    evaluate   every variant on the same matches, Holm, decisions          -> results/variants.json, variant_fits.json
"""
from __future__ import annotations

import time

from experiments.v6_2.odds_timing.common import environment

from . import base, evaluate, models, perf, profiles, transfer
from . import common as C
from . import wf


def run_info(timings: dict) -> None:
    C.save_json("run_info.json", {
        "git_sha": C.git_sha(),
        "data_version": C.data_version([C.ROOT / "local_data" / "processed" / "features.parquet"]),
        "results_csv_files": len(list((C.ROOT / "local_data" / "results").glob("*/*.csv"))),
        "params": {"seeds": C.SEEDS, "prod_seed": C.PROD_SEED, "n_jobs": C.N_JOBS, "n_boot": C.N_BOOT,
                   "bootstrap_seed": C.SEED, "select": C.SELECT, "info": C.INFO, "current": C.CURRENT,
                   "market": f"Bet365 opening 1X2, de-margined ({C.OPEN_METHOD}); closing ({C.CLOSE_METHOD}) for information",
                   "production_version": wf.V6.prod_meta().get("version"),
                   "production_params": {t: list(wf.V6.prod_params(t)) for t in ("result", "home_goals", "away_goals")}},
        "library_versions": environment()["libraries"],
        "timings_seconds": timings, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def main() -> None:
    C.setup_logging("leagues")
    t = {}
    for name, fn in (("base", base.build), ("profiles", profiles.run), ("perf", perf.run), ("models", models.build_all),
                     ("transfer_fits", transfer.build_all), ("transfer", transfer.run), ("evaluate", evaluate.run)):
        t0 = time.time()
        fn()
        t[name] = round(time.time() - t0, 1)
        C.log.info("step %s done in %.0f s", name, t[name])
    run_info(t)
    print("RUN_DONE")


if __name__ == "__main__":
    main()
