"""v6.3 summary (Prompt 9, Stage 6 and §32): overview of the 13 studies, combinations, the key table.

    .venv/bin/python -m experiments.v6_3.summary.run          # light job, ~1 min
    .venv/bin/python -m experiments.v6_3.summary.build_pdf    # reports/v6_3/FootPredict_v6.3_Research_Summary.pdf

The plan of the combinations (``plan.py``) is frozen once; ``run`` never rewrites it.
"""
from __future__ import annotations

import time

from experiments.v6_3 import common as C

from . import combine, key_table, overview, plan


def main() -> None:
    started = time.time()
    plan.main(force=False)
    overview.run()
    combine.run()
    key_table.run()
    C.write_json(combine.RES / "run_info.json", C.run_info("summary", started, {
        "steps": ["plan (frozen, not rewritten)", "overview", "combine", "key_table"],
        "inputs": "experiments/v6_3/*/results, replay/cache, calibration/goals/corners/consensus caches",
        "n_boot": C.N_BOOT, "seed": C.SEED}))


if __name__ == "__main__":
    main()
