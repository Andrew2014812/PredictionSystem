"""Paths and shared helpers of the v6.2 temporal-features study.

Periods, seeds, market reference and statistics are those of the league study (experiments/v6_2/leagues/common.py):
selection on 2023/24 + 2024/25 (walk-forward, models fitted on earlier seasons only), 2025/26 for information,
2026/27 the current season, already looked at (never an untouched holdout). Only the folders differ.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from experiments.v6_2.leagues.common import (  # noqa: F401  (re-exported)
    CURRENT, EVAL, FIRST_OOS, INFO, N_BOOT, N_JOBS, PROD_SEED, ROOT, SEED, SEEDS, SELECT, SMALL_N, clean, data_version,
    ece_multi, git_sha, holm, league_name, logloss_rows, mean_ci, paired_diff_ci, period_label, plt, poisson_dev_rows,
    season_label, short_period)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("temporal62")


def setup_logging(name: str = "temporal62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def save_json(name: str, obj, folder: Path = RESULTS) -> Path:
    path = folder / name
    path.write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None, folder: Path = RESULTS):
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_csv(name: str, df: pd.DataFrame, folder: Path = RESULTS) -> Path:
    path = folder / name
    df.to_csv(path, index=False)
    return path
