"""Paths and shared helpers of the v6.2 league-specialisation study.

Periods (season = start year):
    SELECT   2023/24, 2024/25   variants are chosen here only (walk-forward: models fitted on earlier seasons)
    INFO     2025/26            used by earlier studies -> "for information"
    CURRENT  2026/27 so far     used repeatedly by the v6.1/v6.2 studies -> "current season, already looked at",
                                never an untouched holdout
Every fitted constant of a variant for season s (calibration maps, shrinkage, league groups) is fitted only on
seasons before s: the walk-forward predictions of 2022/23 ... s-1 (each predicted by models fitted before it)
and, for league profiles, the raw results of 2021/22 ... s-1.

Generic statistics (bootstrap, Holm, de-margining, IO) come from the odds-timing study's common module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_2.odds_timing.common import (  # noqa: F401  (re-exported)
    N_BOOT, ROOT, SEED, clean, data_version, demargin, git_sha, holm, mean_ci, paired_diff_ci, season_label)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
SEEDS = (42, 7, 2026)          # 42 = config.RANDOM_STATE (production); the other two estimate seed noise
PROD_SEED = 42
FIRST_OOS = 2022               # first season with walk-forward predictions (models need >= 1 earlier season)
SEASONS = (2022, 2023, 2024, 2025, 2026)
SELECT = (2023, 2024)
INFO = 2025
CURRENT = 2026
EVAL = (2023, 2024, 2025, 2026)
SMALL_N = 300                  # a league-season sample below this is marked "small"

# market reference: Bet365 opening 1X2 (available before the decision), de-margined with the method the bookmaker
# study chose for it (experiments/v6_2/bookmaker/results/benchmark.json "choice"); closing for information.
OPEN_METHOD = "power"
CLOSE_METHOD = "add"

log = logging.getLogger("leagues62")


def setup_logging(name: str = "leagues62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", CURRENT: " (текущий, уже просмотрен)"}.get(s, "")


def short_period(s: int) -> str:
    return season_label(s) + {INFO: " инф.", CURRENT: " тек."}.get(s, "")


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


def logloss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0))


def poisson_dev_rows(y, mu) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))


def ece_multi(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Class-wise ECE of a 1X2 matrix (mean over H, D, A)."""
    p = np.asarray(p, float)
    out = []
    for k in range(3):
        o, q = (np.asarray(y) == k).astype(float), p[:, k]
        idx = np.minimum((q * bins).astype(int), bins - 1)
        out.append(sum((idx == b).mean() * abs(o[idx == b].mean() - q[idx == b].mean())
                       for b in range(bins) if (idx == b).any()))
    return float(np.mean(out))


def plt():
    import matplotlib
    matplotlib.use("Agg")
    from scripts import report_kit  # noqa: F401  (series chart style)
    import matplotlib.pyplot as _plt
    return _plt


def league_name(code: str) -> str:
    from src.leagues import get_league
    lg = get_league(code)
    return f"{lg.country[:3]}. {lg.name}" if lg.country != "Other" else code


TIER = {}


def tier(code: str) -> int:
    if code not in TIER:
        from src.leagues import get_league
        TIER[code] = get_league(code).tier
    return TIER[code]
