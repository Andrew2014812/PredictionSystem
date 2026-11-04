"""Shared helpers of the v6.1 Main-strategy research.

Periods (season = start year):

    SELECT   2023/24, 2024/25   variants are chosen here only (walk-forward: models fitted on earlier seasons)
    INFO     2025/26            looked at during earlier development -> reported "for information", never chosen on
    LIVE     2026/27 so far     the independent check: production history store (unified_history + markets)

Every fitted constant of a strategy (calibration maps, blend weights, meta-models) for season s is fitted on
season s-1 predicted by the models trained on seasons < s-1 (the "inner" season), never on season s.

Resource limits: this machine is shared with four other studies, so every model uses at most N_JOBS threads.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
LOGS = HERE / "logs"
for _d in (RESULTS, FIGURES, CACHE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_JOBS = 3
SELECT = (2023, 2024)
INFO = 2025
LIVE = 2026
FOLDS = (2023, 2024, 2025)
OUTER_SEASONS = (2022, 2023, 2024, 2025)      # 2022 only serves as the inner season of the 2023 fold
SEED = 20261104
N_BOOT = 2000

log = logging.getLogger("main61")


def setup_logging(name: str = "main61") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", LIVE: " (незав.)"}.get(s, "")


# ------------------------------------------------------------------------------------------------ bootstrap / ROI
def roi_ci(profit, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean profit per unit stake (ROI as a fraction)."""
    p = np.asarray(profit, float)
    p = p[np.isfinite(p)]
    if len(p) < 2:
        return (np.nan, np.nan)
    m = _boot_means(p, n_boot, seed)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _boot_means(x: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap means of x (NaN ignored), drawn in chunks to keep memory small."""
    rng = np.random.default_rng(seed)
    out, done, n = [], 0, len(x)
    while done < n_boot:
        k = min(250, n_boot - done)
        with np.errstate(all="ignore"):
            out.append(np.nanmean(x[rng.integers(0, n, size=(k, n))], 1))
        done += k
    return np.concatenate(out)


def paired_roi_diff(profit_a, profit_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap over MATCHES of ROI(b) - ROI(a); profits are per match (NaN = no bet)."""
    a, b = np.asarray(profit_a, float), np.asarray(profit_b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs, done = [], 0
    while done < n_boot:
        k = min(250, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        with np.errstate(all="ignore"):
            diffs.append(np.nanmean(b[idx], 1) - np.nanmean(a[idx], 1))
        done += k
    d = np.concatenate(diffs)
    return {"diff": float(np.nanmean(b) - np.nanmean(a)),
            "ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))],
            "p_b_better": float(np.mean(d > 0))}


def bet_summary(won, odds, prob=None, n_boot: int = N_BOOT) -> dict:
    """won: 1 win / 0 loss (PUSH and unpriced already removed); odds: real price."""
    won, odds = np.asarray(won, float), np.asarray(odds, float)
    profit = np.where(won == 1, odds - 1, -1.0)
    n = len(profit)
    lo, hi = roi_ci(profit, n_boot) if n >= 2 else (np.nan, np.nan)
    out = {"bets": int(n), "hit_rate": float(won.mean()) if n else np.nan,
           "avg_odds": float(odds.mean()) if n else np.nan, "profit": float(profit.sum()) if n else 0.0,
           "roi": float(profit.mean()) if n else np.nan, "roi_ci95": [lo, hi],
           "breakeven_hit": float(np.mean(1 / odds)) if n else np.nan}
    if prob is not None and n:
        prob = np.asarray(prob, float)
        out["mean_prob"] = float(prob.mean())
        out["mean_ev"] = float(np.mean(prob * odds - 1))
    return out


# ------------------------------------------------------------------------------------------------ calibration metrics
def ece(outcome, proba, bins: int = 10) -> float:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    if not len(o):
        return np.nan
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(o[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def binary_scores(outcome, proba) -> dict:
    o = np.asarray(outcome, float)
    p = np.clip(np.asarray(proba, float), 1e-6, 1 - 1e-6)
    if not len(o):
        return {"n": 0}
    return {"n": int(len(o)), "mean_prob": float(p.mean()), "hit_rate": float(o.mean()),
            "gap": float(p.mean() - o.mean()), "brier": float(np.mean((p - o) ** 2)),
            "log_loss": float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))), "ece": ece(o, p)}


def reliability_bins(outcome, proba, edges=None, min_n: int = 15) -> list[dict]:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    edges = np.asarray(edges if edges is not None else np.arange(0.40, 0.90, 0.05))
    idx = np.digitize(p, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_n:
            k = int(m.sum())
            ob = float(o[m].mean())
            se = float(np.sqrt(max(ob * (1 - ob), 1e-9) / k))
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "predicted": float(p[m].mean()),
                        "observed": ob, "se": se, "n": k})
    return out


# ------------------------------------------------------------------------------------------------ IO
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_csv(name: str, df: pd.DataFrame) -> Path:
    path = RESULTS / name
    df.to_csv(path, index=False)
    return path


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "statsmodels",
                 "matplotlib", "reportlab", "pypdfium2"):
        try:
            libs[name] = importlib.import_module(name).__version__
        except Exception:
            libs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": libs,
            "git_sha": git_sha(), "n_jobs": N_JOBS, "seed": SEED, "n_boot": N_BOOT}


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> float:
        self.marks[name] = round(time.time() - self.t0, 1)
        return self.marks[name]


def plt():
    import matplotlib
    matplotlib.use("Agg")
    from scripts import report_kit  # noqa: F401  (applies the series' chart style)
    import matplotlib.pyplot as _plt
    return _plt


GROUP_RU = {"TOTAL": "Тотал голов", "HANDICAP": "Азиатская фора", "1X2": "1X2", "BTTS": "Обе забьют",
            "DOUBLE_CHANCE": "Двойной шанс", "CORNERS": "Угловые"}
