"""FootPredict v6.1 — corners model research. Entry point.

    python -m experiments.v6_1.corners.run --all                 # everything (≈ 1.5–2.5 h on 3 threads)
    python -m experiments.v6_1.corners.run --prep                # frame, feature inventory, leakage check
    python -m experiments.v6_1.corners.run --current             # current production metrics (real bundles)
    python -m experiments.v6_1.corners.run --team-strength       # rolling hierarchical model grid
    python -m experiments.v6_1.corners.run --fit 2023 2024       # mean models of the given folds
    python -m experiments.v6_1.corners.run --evaluate            # distributions, metrics, selection
    python -m experiments.v6_1.corners.run --betting             # 2026/27 real corner prices + history
    python -m experiments.v6_1.corners.run --figures
    python -m experiments.v6_1.corners.build_pdf                 # reports/v6_1/Corners_Research.pdf

Read only with respect to production: nothing under local_data/ is written (caches live in
experiments/v6_1/corners/cache/, results in results/, figures in figures/).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import time

from . import data as D

log = D.setup_logging()


def versions() -> dict:
    import catboost
    import lightgbm
    import numpy
    import pandas
    import scipy
    import sklearn
    import statsmodels
    import xgboost
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=D.HERE).stdout.strip()
    return {"python": platform.python_version(), "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__, "git_sha": sha,
            "n_jobs": D.N_JOBS}


def stage(name, fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    minutes = round((time.time() - t0) / 60, 2)
    times = D.load_result("run_times.json", []) or []          # one entry per executed stage (append-only)
    times.append({"stage": name, "minutes": minutes, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    D.save_result("run_times.json", times)
    log.info("stage %s: %.1f min", name, minutes)
    return out


def prep():
    df = D.frame()
    sets = D.feature_sets(df)
    D.save_result("inventory.json", {**D.inventory(df, sets), "leakage_check": D.leakage_check(df)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--current", action="store_true")
    ap.add_argument("--team-strength", action="store_true")
    ap.add_argument("--fit", nargs="*", type=int)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--betting", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    D.save_result("versions.json", versions())
    if a.all or a.prep:
        stage("prep", prep)
    if a.all or a.current:
        from . import current
        stage("current", current.run)
    if a.all or a.team_strength:
        from . import fit
        stage("team_strength", lambda: D.save_result("team_strength_grid.json", fit.team_strength_grid()))
    if a.all or a.fit is not None:
        from . import fit
        for s in (a.fit or D.FOLDS):
            stage(f"fit_{s}", fit.fit_fold, s, a.models, a.refresh)
    if a.all or a.evaluate:
        from . import evaluate
        stage("evaluate", evaluate.run)
    if a.all or a.betting:
        from . import betting
        stage("betting", betting.run)
    if a.all or a.figures:
        from . import figures
        stage("figures", figures.run)


if __name__ == "__main__":
    main()
