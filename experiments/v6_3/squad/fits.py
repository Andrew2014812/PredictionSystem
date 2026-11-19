"""Model fits of the squad study (heavy; run through ``experiments/v6_3/heavy.sh``).

    experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e1.log \
        .venv/bin/python -m experiments.v6_3.squad.fits e1
    experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e2.log \
        .venv/bin/python -m experiments.v6_3.squad.fits e2

E1 — earlier seasons, chronological by season (squad features exist with line-ups only up to 2025/26):
  test season 2024/25 (trained on 2021/22–2023/24) and 2025/26 (trained on 2021/22–2024/25);
  B0 = production feature set and parameters refitted on the same rows (control);
  DYN = B0 + dynamic squad features; PRE = B0 + pre-season squad features;
  SA = squad-aware compact model (parameters chosen on 2024/25, evaluated on 2025/26 only).
E2 — season 2026/27 (only pre-season squad features exist): every model fitted once on everything played before
  1 July 2026 (the production regime, prod_20261012 was trained on the same seasons):
  B0r = production refit (control), PRE = B0 + pre-season features, SA-pre = compact model tuned on 2025/26.
Outputs (p 1X2, λ home / away, μ corners) of every match of the test season go to ``cache/outputs_*.parquet``.
"""
from __future__ import annotations

import gc
import sys
import time

import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as D
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.squad import models as M

DIRS = C.study_dirs("squad")
CACHE = DIRS["cache"]


def load_table() -> pd.DataFrame:
    f = D.load_features()
    sq = pd.read_parquet(CACHE / "squad_features.parquet")
    return f.merge(sq, on="match_id", how="left")


def extras(columns, family: str) -> dict[str, list[str]]:
    sq = M.squad_columns(columns, family)
    return {"result": sq, "home_goals": sq, "away_goals": sq, "corners": M.corner_extra(sq)}


def predict(bundles: dict, rows: pd.DataFrame, tag: str) -> None:
    out = EN.outputs_from_bundles({**EN.production_bundles(), **bundles}, rows)
    out.to_parquet(CACHE / f"outputs_{tag}.parquet")


def fit_variant(train: pd.DataFrame, family: str | None) -> dict:
    ex = extras(train.columns, family) if family else {k: [] for k in M.TASK_NAMES}
    return {name: M.fit_plus(name, train, ex[name]) for name in M.TASK_NAMES}


def e1(log) -> None:
    f = load_table()
    played = f["played"].astype(bool)
    tuned_log = {}
    for season in (2024, 2025):
        train = f.loc[played & (f["season"] < season)]
        test = f.loc[f["season"] == season]
        for tag, fam in (("b0", None), ("dyn", "sq"), ("pre", "pre")):
            t = time.time()
            predict(fit_variant(train, fam), test, f"e1_{season}_{tag}")
            log.info("E1 %s %s: %d train rows, %.0f s", season, tag, len(train), time.time() - t)
            gc.collect()
    # squad-aware: parameters chosen on 2024/25 (trained on 2021/22–2023/24), evaluated on 2025/26
    sq = extras(f.columns, "sq")
    tr, va = f.loc[played & (f["season"] < 2024)], f.loc[played & (f["season"] == 2024)]
    bundles = {}
    for name in M.TASK_NAMES:
        tuned = M.tune_compact(name, tr, va, sq[name])
        tuned_log[name] = tuned
        bundles[name] = M.fit_compact(name, f.loc[played & (f["season"] < 2025)], sq[name], tuned["best"])
        log.info("E1 SA %s: %s", name, tuned["best"])
    predict(bundles, f.loc[f["season"] == 2025], "e1_2025_sa")
    C.write_json(DIRS["results"] / "tuning_e1_sa.json", tuned_log)


def e2(log) -> None:
    f = load_table()
    played = f["played"].astype(bool)
    cut = pd.Timestamp(f"{C.CURRENT}-07-01")
    train = f.loc[played & (f["date"] < cut)]
    test = f.loc[f["season"] == C.CURRENT]
    # self-check: fit_plus without extra columns is engine.fit_task (same rows, features, parameters)
    a = M.fit_plus("home_goals", train, []).predict(test)
    b = EN.fit_task("home_goals", train).predict(test)
    C.write_json(DIRS["results"] / "self_check_fit_plus.json",
                 {"task": "home_goals", "max_abs_diff_vs_engine_fit_task": float(abs(a - b).max()),
                  "matches": int(len(test))})
    for tag, fam in (("b0r", None), ("pre", "pre")):
        t = time.time()
        predict(fit_variant(train, fam), test, f"e2_{tag}")
        log.info("E2 %s: %d train rows, %.0f s", tag, len(train), time.time() - t)
        gc.collect()
    pre = extras(f.columns, "pre")
    tr, va = f.loc[played & (f["season"] < 2025)], f.loc[played & (f["season"] == 2025)]
    bundles, tuned_log = {}, {}
    for name in M.TASK_NAMES:
        tuned = M.tune_compact(name, tr, va, pre[name])
        tuned_log[name] = tuned
        bundles[name] = M.fit_compact(name, train, pre[name], tuned["best"])
        log.info("E2 SA %s: %s", name, tuned["best"])
    predict(bundles, test, "e2_sa")
    C.write_json(DIRS["results"] / "tuning_e2_sa.json", tuned_log)


def main() -> None:
    stage = sys.argv[1]
    log = C.setup_logging(f"squad.fits.{stage}")
    started = time.time()
    {"e1": e1, "e2": e2}[stage](log)
    log.info("stage %s done in %.0f s", stage, time.time() - started)


if __name__ == "__main__":
    main()
