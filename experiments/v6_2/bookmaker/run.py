"""Runs every experiment of the bookmaker-benchmark study (python -m experiments.v6_2.bookmaker.run).

    --only benchmark,hybrids,multimarket,figures    a subset (default: all, in this order)

Long run (~10–20 min): start it with `bash experiments/v6_2/bookmaker/launch.sh run` (deadline + log + marker).
"""
from __future__ import annotations

import argparse
import time

from . import common as C

STEPS = ("benchmark", "hybrids", "multimarket", "figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    args = ap.parse_args()
    C.setup_logging("bench62")
    steps = [s for s in STEPS if s in args.only.split(",")]
    t0, times = time.time(), {}
    for step in steps:
        t = time.time()
        if step == "benchmark":
            from . import benchmark
            benchmark.run()
        elif step == "hybrids":
            from . import hybrids
            hybrids.run()
        elif step == "multimarket":
            from . import multimarket
            multimarket.run()
        elif step == "figures":
            from . import figures
            figures.run()
        times[step] = round(time.time() - t, 1)
        print(f"{step} done in {times[step]} s", flush=True)
    root = C.ROOT / "local_data"
    inputs = [root / "odds" / "odds.parquet", root / "predictions" / "picks.parquet",
              root / "predictions" / "markets.parquet", root / "processed" / "features.parquet"]
    inputs += sorted((root / "results").glob("2*/E0.csv"))
    inputs += sorted((C.ROOT / "experiments" / "v6_1" / "main" / "cache").glob("cand_*.parquet"))
    info = {"environment": C.environment(), "data_version": C.data_version(inputs), "steps_seconds": times,
            "total_seconds": round(time.time() - t0, 1), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": {"bootstrap": "1000 (paired, by match); encompassing 100", "select": list(C.SELECT),
                           "info": [C.INFO, C.LIVE], "hybrid_fit": "expanding window from 2022/23 to s-1",
                           "market_input": "Bet365 opening (Football-Data), de-margined by the method chosen on 2023/24+2024/25",
                           "xgb_residual": "depth 3, eta 0.03, subsample 0.8, colsample 0.5, min_child_weight 50, "
                                           "lambda 10, early stopping on s-1, nthread 2",
                           "value_rule": "bet the selection with the highest EV = p x Bet365 opening - 1 if EV > tau"}}
    C.save_json("run_info.json", info)


if __name__ == "__main__":
    main()
