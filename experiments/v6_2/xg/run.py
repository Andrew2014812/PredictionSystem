"""Entry point of the v6.2 xG study (research only; production code and data untouched).

    python -m experiments.v6_2.xg.run --all          # every step below, in order
    python -m experiments.v6_2.xg.run --audit        # xG audit (results/audit*.json|csv)
    python -m experiments.v6_2.xg.run --features     # pre-match xG features + season-start coverage
    python -m experiments.v6_2.xg.run --signal       # univariate signal of the features
    python -m experiments.v6_2.xg.run --evaluate     # variants, metrics, Main / ROI, breakdowns, corners
    python -m experiments.v6_2.xg.run --sensitivity  # lag 4 days, ridge 3 / 30
    python -m experiments.v6_2.xg.run --decide       # A / B / C (results/decisions.json)
    python -m experiments.v6_2.xg.run --figures      # figures/*.png
    python -m experiments.v6_2.xg.build_pdf          # reports/v6_2/XG_Research.pdf

``--refresh`` rebuilds the caches (baseline predictions, features, variant predictions, prices).
Run times, parameters, library versions, the Git SHA and the data version go to results/run_info.json.
"""
from __future__ import annotations

import argparse
import time

from . import common as C

STEPS = ("audit", "features", "signal", "evaluate", "sensitivity", "decide", "figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in STEPS + ("all", "refresh"):
        ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    log = C.setup_logging("run")
    info = C.load_json("run_info.json", {}) or {}
    info.setdefault("steps", {})
    t0 = time.time()
    for step in STEPS:
        if not (getattr(a, step) or a.all):
            continue
        started = time.time()
        if step == "audit":
            from . import audit, data
            data.baseline(refresh=a.refresh)
            audit.run()
        elif step == "features":
            from . import data, xg_features
            f = xg_features.build(0, refresh=a.refresh)
            xg_features.build(4, refresh=a.refresh)
            C.save_json("features_coverage.json", xg_features.coverage(f, data.baseline()))
        elif step == "signal":
            from . import models, signal
            signal.run(models.frame(0))
        elif step == "evaluate":
            from . import study
            study.run(refresh=a.refresh)
        elif step == "sensitivity":
            from . import study
            study.sensitivity()
        elif step == "decide":
            from . import study
            study.decisions()
        elif step == "figures":
            from . import figures
            figures.run()
        info["steps"][step] = {"minutes": round((time.time() - started) / 60, 2),
                               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        log.info("%s finished in %.1f min", step, (time.time() - started) / 60)
        from . import models, xg_features
        info["environment"] = C.environment()
        info["data_version"] = C.data_version()
        info["parameters"] = {"glm_ridge": models.GLM_RIDGE, "mnl_ridge": models.MNL_RIDGE,
                              "xgb_residual": models.XGB_RESID, "rating_ridge": xg_features.RIDGE,
                              "k_shrink": xg_features.K_SHRINK, "k_finish": xg_features.K_FINISH,
                              "ewm_half_life_matches": xg_features.HALF_LIFE,
                              "min_league_matches": xg_features.MIN_LEAGUE_MATCHES,
                              "eval_start": str(C.EVAL_START.date()), "refit_days": C.REFIT_DAYS,
                              "n_boot": C.N_BOOT, "seed": C.SEED, "variants": models.VARIANTS}
        C.save_json("run_info.json", info)
    log.info("total %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
