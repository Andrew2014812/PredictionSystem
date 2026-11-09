"""Paths and shared helpers of the v6.2 bookmaker-benchmark study.

Periods (season = start year):
    SELECT  2023/24, 2024/25   variants and hybrids are chosen here only (walk-forward)
    INFO    2025/26            used by earlier studies -> "for information"
    LIVE    2026/27 so far     used repeatedly by the v6.1 studies -> NOT an untouched holdout, "for information"
Every fitted part of a hybrid for season s is fitted on seasons before s only (expanding window from 2022/23,
the first season with walk-forward model predictions).

Generic statistics (bootstrap, Holm, de-margining, IO) come from the odds-timing study's common module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_2.odds_timing.common import (  # noqa: F401  (re-exported)
    METHOD_RU, METHODS, N_BOOT, ROOT, SEED, clean, data_version, demargin, diff_ci, environment, git_sha,
    grouped_mean_ci, holm, mean_ci, paired_diff_ci, plt, roi_summary, season_label, setup_logging, uk_to_utc)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
FIRST = 2022                 # first season with walk-forward predictions (v6.1 cache)
SELECT = (2023, 2024)
INFO = 2025
LIVE = 2026
EVAL = (2023, 2024, 2025, 2026)


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", LIVE: " (инф., не holdout)"}.get(s, "")


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_csv(name: str, df: pd.DataFrame) -> Path:
    path = RESULTS / name
    df.to_csv(path, index=False)
    return path


def logloss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row log loss of a (n, k) probability matrix at the observed class y (int)."""
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def brier_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    return ((p - onehot) ** 2).sum(1)


def ece(outcome, proba, bins: int = 10) -> float:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(o[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


GROUP_RU = {"1X2": "1X2", "OU25": "Тотал 2.5", "AH": "Азиатская фора (линия FD)", "BTTS": "Обе забьют",
            "TOTAL": "Тоталы голов", "CORNERS": "Угловые", "DOUBLE_CHANCE": "Двойной шанс", "HANDICAP": "Азиатская фора"}
