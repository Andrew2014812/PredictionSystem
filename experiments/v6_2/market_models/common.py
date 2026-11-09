"""Paths, seasons and statistics of the v6.2 Market-Specific Models study (statistics shared with main_calibration).

Periods (season = start year):

    SELECT   2023/24, 2024/25   variants are chosen here only (walk-forward: every model fitted on earlier seasons)
    INFO     2025/26            looked at by earlier studies -> "for information"
    CURRENT  2026/27            the current season, already looked at by the v6.1 studies (NOT a holdout)

Every constant of a variant used to predict season s (stacking weights, tree counts) is fitted on seasons < s.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_2.main_calibration.common import (  # noqa: F401  (re-exported)
    N_BOOT, N_JOBS, ROOT, SEED, _clean, boot_idx, boot_stat, ci, data_version, ece, environment, git_sha, holm,
    p_two_sided, plt, reliability_bins, season_label)

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
SEASONS = (2023, 2024, 2025, 2026)          # evaluated seasons
FOLDS = (2022, 2023, 2024, 2025, 2026)      # outer folds (2022 serves as the inner season of 2023)

log = logging.getLogger("mm62")


def setup_logging(name: str = "mm62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def period_label(s) -> str:
    s = int(s)
    return season_label(s) + {INFO: " (инф.)", CURRENT: " (текущий)"}.get(s, "")


def save_json(name: str, obj, folder: Path = RESULTS) -> Path:
    path = folder / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None, folder: Path = RESULTS):
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


# ------------------------------------------------------------------------------------------------ scoring rules
EPS = 1e-12


def bin_ll(o, p) -> np.ndarray:
    """Per-match binary log loss."""
    o, p = np.asarray(o, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def bin_brier(o, p) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(o, float)) ** 2


def multi_ll(y, P) -> np.ndarray:
    """Per-match multiclass log loss (y = class index, P = (n, k))."""
    P = np.asarray(P, float)
    return -np.log(np.clip(P[np.arange(len(P)), np.asarray(y, int)], 1e-12, 1))


def multi_brier(y, P) -> np.ndarray:
    P = np.asarray(P, float)
    return ((P - np.eye(P.shape[1])[np.asarray(y, int)]) ** 2).sum(1)


def rps(y, P) -> np.ndarray:
    """Ranked probability score of ordered classes (per match)."""
    P = np.asarray(P, float)
    cp = np.cumsum(P, 1)[:, :-1]
    co = np.cumsum(np.eye(P.shape[1])[np.asarray(y, int)], 1)[:, :-1]
    return ((cp - co) ** 2).sum(1) / (P.shape[1] - 1)


def paired(loss_new, loss_ref, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of new - ref per match (negative = new better), bootstrap 95 % CI and two-sided p."""
    d = np.asarray(loss_new, float) - np.asarray(loss_ref, float)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return {"n": int(len(d)), "diff": np.nan, "ci95": [np.nan, np.nan], "p": np.nan}
    dist = boot_stat(lambda a: a.mean(1), d, n_boot=n_boot, seed=seed)
    from scipy import stats
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    p_t = float(2 * stats.norm.sf(abs(d.mean()) / se)) if se > 0 else (0.0 if d.mean() != 0 else 1.0)
    return {"n": int(len(d)), "diff": float(d.mean()), "ci95": ci(dist), "p": p_two_sided(dist), "p_t": p_t}


def summary_binary(o, p) -> dict:
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if not len(o):
        return {"n": 0}
    return {"n": int(len(o)), "log_loss": float(bin_ll(o, p).mean()), "brier": float(bin_brier(o, p).mean()),
            "ece": ece(o, p), "mean_p": float(p.mean()), "rate": float(o.mean())}


def calib_curve(o, p, bins: int = 10, min_n: int = 30) -> list[dict]:
    """Reliability points on equal-count bins of p."""
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if len(p) < bins * min_n:
        bins = max(2, len(p) // min_n)
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False).to_numpy()
    return [{"p": float(p[q == b].mean()), "o": float(o[q == b].mean()), "n": int((q == b).sum())}
            for b in range(bins) if (q == b).sum() >= min_n]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)
