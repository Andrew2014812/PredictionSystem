"""FootPredict v6.1 — draw research, all steps.

    python -m experiments.v6_1.draw.run                 # everything (cached blocks are reused)
    python -m experiments.v6_1.draw.run --refresh       # refit every block from scratch
    python -m experiments.v6_1.draw.build_pdf           # reports/v6_1/Draw_Research.pdf

Steps: blocks (production-faithful walk-forward fits) -> probs (Q3) -> rules (Q1, Q5) -> main_study (Q2, Q4)
-> figures -> findings.md. Results: experiments/v6_1/draw/results/*.json|csv, figures: figures/*.png,
run metadata (package versions, Git SHA, run times): results/run_info.json.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import subprocess
import time

from . import common as C

log = C.setup_logging("draw61.run")


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.HERE, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    times = {}
    t0 = time.time()
    from . import blocks
    blocks.all_blocks(args.refresh)
    times["blocks"] = time.time() - t0
    from . import probs
    t = time.time()
    if args.refresh:
        for b in C.BLOCKS:
            probs.extra_block(b, refresh=True)
    probs.run()
    times["probabilities"] = time.time() - t
    from . import rules
    t = time.time()
    rules.run()
    times["rules"] = time.time() - t
    from . import main_study
    t = time.time()
    main_study.run()
    times["draw_main"] = time.time() - t
    from . import figures
    t = time.time()
    figures.run()
    times["figures"] = time.time() - t
    from . import findings
    findings.write()
    times["total"] = time.time() - t0
    pk = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "statsmodels", "matplotlib", "reportlab", "pyarrow"]
    info = {"git_sha": _sha(), "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: md.version(p) for p in pk if _has(p)}, "n_jobs": C.N_JOBS,
            "active_model": C.prod_meta()["version"], "seconds": times,
            "block_seconds": {str(b): blocks.block_info(b) for b in C.BLOCKS},
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.save_json("run_info.json", info)
    log.info("done in %.1f min", times["total"] / 60)


def _has(p: str) -> bool:
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
