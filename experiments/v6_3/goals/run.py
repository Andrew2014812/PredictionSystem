"""Goal Model Improvements (v6.3, study 10) — every experiment, results to results/, figures to figures/.

    .venv/bin/python -m experiments.v6_3.goals.run --stage ratings      # light: dynamic ratings (choose η, r)
    experiments/v6_3/heavy.sh 5400 experiments/v6_3/goals/logs/base.log .venv/bin/python -m experiments.v6_3.goals.base
    .venv/bin/python -m experiments.v6_3.goals.run --stage eval         # light: selection, 2026/27 replay, Main ROI
"""
from __future__ import annotations

import argparse
import time

from experiments.v6_3 import common as C

from . import ratings as R

DIRS = C.study_dirs("goals")


def stage_ratings(log) -> dict:
    sel = R.select(log)
    b, h = sel["best"], sel["best_ha"]
    R.compute(b["eta"], b["r"]).to_parquet(DIRS["cache"] / "ratings_plain.parquet")
    R.compute(h["eta"], h["r"], h["eta_ha"]).to_parquet(DIRS["cache"] / "ratings_ha.parquet")
    C.write_json(DIRS["results"] / "ratings_selection.json", sel)
    log.info("ratings: best %s, home/away %s", b, h)
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ratings", "eval", "all"], default="all")
    a = ap.parse_args()
    started = time.time()
    log = C.setup_logging("goals.run", DIRS["logs"] / "run.log")
    if a.stage in ("ratings", "all"):
        stage_ratings(log)
    if a.stage in ("eval", "all"):
        from . import evaluate as EVL
        EVL.run(log, started)


if __name__ == "__main__":
    main()
