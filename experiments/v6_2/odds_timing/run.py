"""Runs every experiment of the odds-timing study and writes results/ (python -m experiments.v6_2.odds_timing.run).

    --only audit,drift,stability,clv,figures     run a subset (default: all, in this order)
"""
from __future__ import annotations

import argparse
import time

from . import common as C

STEPS = ("audit", "drift", "stability", "clv", "figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    args = ap.parse_args()
    C.setup_logging("timing62")
    steps = [s for s in STEPS if s in args.only.split(",")]
    t0, times = time.time(), {}
    for step in steps:
        t = time.time()
        if step == "audit":
            from . import audit
            audit.run()
        elif step == "drift":
            from . import drift
            drift.run()
        elif step == "stability":
            from . import stability
            stability.run()
        elif step == "clv":
            from . import clv
            clv.run()
        elif step == "figures":
            from . import figures
            figures.run()
        times[step] = round(time.time() - t, 1)
        C.log.info("%s done in %.1f s", step, times[step])
    inputs = [C.ODDS_FILE, C.PRED_DIR / "picks.parquet", C.PRED_DIR / "versions.parquet",
              C.PRED_DIR / "snapshots.parquet", C.PRED_DIR / "markets.parquet",
              C.DATA / "processed" / "features.parquet"]
    inputs += sorted((C.DATA / "results").glob("2*/E0.csv"))
    info = {"environment": C.environment(), "data_version": C.data_version(inputs), "steps_seconds": times,
            "total_seconds": round(time.time() - t0, 1), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": {"bootstrap": C.N_BOOT, "near_close_hours": 3.0, "reference_lead_max_h": 3.0,
                           "demargin": "Shin (drift, CLV)", "seasons_fd": "2021/22–2026/27",
                           "walk_forward_model_seasons": "2022/23–2025/26 (v6.1 cache)"}}
    C.save_json("run_info.json", info)


if __name__ == "__main__":
    main()
