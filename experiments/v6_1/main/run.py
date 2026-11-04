"""Entry point of the v6.1 Main-strategy research (research only; production code and data untouched).

    python -m experiments.v6_1.main.run --all            # every fold analysis (not the 2026/27 check)
    python -m experiments.v6_1.main.run --base           # walk-forward production predictions (cache)
    python -m experiments.v6_1.main.run --candidates     # candidate tables + self-check against production
    python -m experiments.v6_1.main.run --strategies     # A–G on the folds (+ selection)
    python -m experiments.v6_1.main.run --calibration --bias --mix --betting --prices --models --figures
    python -m experiments.v6_1.main.run --select         # results/selection.json (pre-registered candidate)
    python -m experiments.v6_1.main.preregister --strategy   # then commit results/preregistration.json
    python -m experiments.v6_1.main.run --independent    # 2026/27 check (needs the 'strategy' stage)
    python -m experiments.v6_1.main.build_pdf            # reports/v6_1/Main_Strategy_Research.pdf

Run times, parameters, library versions and the Git SHA of every run go to results/run_info.json.
"""
from __future__ import annotations

import argparse
import time

from . import common as C

STEPS = ("base", "candidates", "strategies", "calibration", "bias", "mix", "betting", "prices", "models", "figures",
         "select", "independent")


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in STEPS + ("all", "refresh"):
        ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    log = C.setup_logging("run")
    info = C.load_json("run_info.json", {}) or {}
    info.setdefault("steps", {})
    t0 = time.time()
    fold_steps = STEPS[:-1]
    for step in STEPS:
        if not (getattr(a, step) or (a.all and step in fold_steps)):
            continue
        started = time.time()
        if step == "base":
            from . import base
            base.build_all(refresh=a.refresh)
        elif step == "candidates":
            from . import candidates as CD
            for s in C.OUTER_SEASONS:
                CD.historical(s, refresh=a.refresh)
            CD.live(refresh=a.refresh)
            C.save_json("self_check.json", {str(s): CD.self_check(s, 150) for s in C.FOLDS})
        elif step == "strategies":
            from . import strategies_eval
            strategies_eval.run()
        elif step == "calibration":
            from . import calibration
            calibration.run()
        elif step == "bias":
            from . import selection_bias
            selection_bias.run()
        elif step == "mix":
            from . import market_mix
            market_mix.run()
        elif step == "betting":
            from . import betting
            betting.run()
        elif step == "prices":
            from . import prices_clv
            prices_clv.run()
        elif step == "models":
            from . import models_cmp
            models_cmp.run()
        elif step == "figures":
            from . import figures_extra
            figures_extra.run()
        elif step == "select":
            from . import selection
            selection.run()
        elif step == "independent":
            from . import independent
            independent.run()
        info["steps"][step] = {"minutes": round((time.time() - started) / 60, 2), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        log.info("%s finished in %.1f min", step, (time.time() - started) / 60)
        info["environment"] = C.environment()
        C.save_json("run_info.json", info)
    log.info("total %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
