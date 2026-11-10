"""Paths, seasons and statistics of the v6.2 Risk study (statistics shared with main_calibration)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_2.main_calibration.common import (  # noqa: F401  (re-exported)
    N_BOOT, N_JOBS, ROOT, SEED, _clean, boot_idx, boot_stat, ci, data_version, ece, environment, git_sha, holm,
    max_drawdown, p_two_sided, plt, reliability_bins, season_label)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
LOGS = HERE / "logs"
for _d in (RESULTS, FIGURES, CACHE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

SELECT = (2023, 2024)
INFO = 2025
CURRENT = 2026
SEASONS = (2023, 2024, 2025, 2026)
HISTORY = (2022, 2023, 2024, 2025)

log = logging.getLogger("risk62")


def setup_logging(name: str = "risk62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", CURRENT: " (текущий)"}.get(s, "")


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def mean_ci(x, n_boot: int = 1000, seed: int = SEED) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [np.nan, np.nan]
    return ci(boot_stat(lambda a: np.nanmean(a, 1), x, n_boot=n_boot, seed=seed))


def longest_losing(won: np.ndarray) -> int:
    best = run = 0
    for w in won:
        run = 0 if w else run + 1
        best = max(best, run)
    return best


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


GROUP_RU = {"TOTAL": "Тотал голов", "HANDICAP": "Азиатская фора", "1X2": "1X2", "BTTS": "Обе забьют",
            "DOUBLE_CHANCE": "Двойной шанс", "CORNERS": "Угловые", "EXACT_SCORE": "Точный счёт"}
_ = pd
