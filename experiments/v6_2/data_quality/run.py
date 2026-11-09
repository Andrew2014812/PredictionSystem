"""Runs every experiment of the data-quality study and writes results/run_info.json.

    python -m experiments.v6_2.data_quality.run            # everything (cached steps are reused)
    python -m experiments.v6_2.data_quality.run --only raw_audit teams
"""
from __future__ import annotations

import argparse
import time

from . import common as C

STEPS = ("raw_audit", "teams", "recompute", "predictions", "store_audit", "sensitivity", "strata", "variants", "figures")


def main(only=None) -> None:
    C.setup_logging("dq62")
    timer = C.Timer()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for step in only or STEPS:
        module = __import__(f"experiments.v6_2.data_quality.{step}", fromlist=["run"])
        if step == "predictions":
            for s in C.EVAL_SEASONS:
                module.season_predictions(s)
        elif step == "variants":
            module.evaluate()
        else:
            module.run()
        timer.mark(step)
    info = C.load_json("run_info.json", {}) or {}
    info.update({"started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "steps_seconds": {**info.get("steps_seconds", {}), **timer.marks},
                 "environment": C.environment(), "data_version": C.data_version(),
                 "parameters": {"eval_seasons": list(C.EVAL_SEASONS), "refit_seasons": list(C.REFIT_SEASONS),
                                "backtest_version": C.BACKTEST_VERSION, "production_version": C.PROD_VERSION,
                                "n_boot": C.N_BOOT, "seed": C.SEED, "n_jobs": C.N_JOBS}})
    C.save_json("run_info.json", info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=STEPS)
    main(ap.parse_args().only)
