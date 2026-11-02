"""Entry point of the FootPredict v6 research (research only; production code and models untouched).

    python -m experiments.v6_research.run --all
    python -m experiments.v6_research.run --base        # cached production-setup fold predictions
    python -m experiments.v6_research.run --draw        # Q1
    python -m experiments.v6_research.run --goals       # Q2
    python -m experiments.v6_research.run --corners     # Q3
    python -m experiments.v6_research.run --markets     # Q4
    python -m experiments.v6_research.run --risk        # Q5
    python -m experiments.v6_research.run --timing      # clean sequential training-time benchmark
    python -m experiments.v6_research.run --summary     # Q6 (reads the results of Q1–Q5 + timing)
    python -m experiments.v6_research.run --all --refresh   # recompute the cached base predictions
"""
from __future__ import annotations

import argparse
import time

from . import common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("all", "base", "draw", "goals", "corners", "markets", "risk", "timing", "summary", "refresh"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    log = C.setup_logging("run")
    t0 = time.time()
    if a.all or a.base or a.refresh:
        for s in C.FOLDS:
            C.base_predictions(s, refresh=a.refresh)
    steps = [("draw", "draw"), ("goals", "goals"), ("corners", "corners"), ("markets", "markets"),
             ("risk", "risk"), ("timing", "timing"), ("summary", "summary")]
    for flag, module in steps:
        if a.all or getattr(a, flag):
            started = time.time()
            mod = __import__(f"experiments.v6_research.{module}", fromlist=["run"])
            mod.run()
            log.info("%s finished in %.1f min", module, (time.time() - started) / 60)
    log.info("total %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
