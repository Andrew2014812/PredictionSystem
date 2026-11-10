"""Shared helpers of the v6.2 Main-calibration study.

Seasons (season = start year):

    SELECT   2023/24, 2024/25   variants are chosen here only
    INFO     2025/26            reported "for information" (used by earlier studies)
    CURRENT  2026/27 so far     the current season, already looked at by the v6.1 studies (NOT a holdout)

Walk-forward predictions of 2022/23 ... 2025/26 are the v6.1 ones (experiments/v6_1/main/base.py: production
XGBoost models fitted on all seasons < s); this study only redirects their cache into its own folder.
Every fitted part of a variant for season s (calibration maps, shrinkage, blend weights) is fitted on the
walk-forward predictions of the seasons BEFORE s (expanding window 2022 ... s-1), never on season s.
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

# the v6.1 walk-forward code is reused; its cache and thread count are redirected before any call
from experiments.v6_1.main import common as V61  # noqa: E402

N_JOBS = 2
V61.CACHE = CACHE
V61.N_JOBS = N_JOBS

SELECT = (2023, 2024)
INFO = 2025
CURRENT = 2026
FOLDS = (2023, 2024, 2025)
SEASONS = (2023, 2024, 2025, 2026)
HISTORY = (2022, 2023, 2024, 2025)          # walk-forward seasons available as calibration history
SEED = 20261109
N_BOOT = 2000

log = logging.getLogger("maincal62")


def setup_logging(name: str = "maincal62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", CURRENT: " (текущий)"}.get(s, "")


# ------------------------------------------------------------------------------------------------ statistics
def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap index chunks (resampling by match)."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(200, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def boot_stat(fn, *arrays, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Bootstrap distribution of fn(*arrays[idx]) (fn vectorised over the first axis of idx)."""
    n = len(arrays[0])
    out = []
    for idx in boot_idx(n, n_boot, seed):
        out.append(fn(*[a[idx] for a in arrays]))
    return np.concatenate(out)


def ci(dist: np.ndarray) -> list[float]:
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return [np.nan, np.nan]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def p_two_sided(dist: np.ndarray, null: float = 0.0) -> float:
    """Bootstrap two-sided p-value of H0: statistic = null (percentile method)."""
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    lo, hi = np.mean(d <= null), np.mean(d >= null)
    return float(min(1.0, 2 * min(lo, hi) + 1.0 / len(d)))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a dict name -> p."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, v in pvals.items():
        out.setdefault(k, np.nan)
    return out


def ece(outcome, proba, bins: int = 10) -> float:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    if not len(o):
        return np.nan
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(o[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def max_drawdown(profit) -> float:
    c = np.cumsum(np.nan_to_num(np.asarray(profit, float)))
    if not len(c):
        return 0.0
    peak = np.maximum.accumulate(np.r_[0.0, c])[1:]
    return float(np.max(peak - c))


def reliability_bins(outcome, proba, edges=None, min_n: int = 20) -> list[dict]:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    edges = np.asarray(edges if edges is not None else np.arange(0.35, 0.95, 0.05))
    idx = np.digitize(p, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_n:
            k = int(m.sum())
            ob = float(o[m].mean())
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "predicted": float(p[m].mean()),
                        "observed": ob, "se": float(np.sqrt(max(ob * (1 - ob), 1e-9) / k)), "n": k})
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


def save_json(name: str, obj, folder: Path = RESULTS) -> Path:
    path = folder / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None, folder: Path = RESULTS):
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def data_version() -> dict:
    """Row counts and modification times of the input files."""
    from src import config
    files = {"features": config.PROCESSED_DIR / "features.parquet",
             "markets": config.PREDICTIONS_DIR / "markets.parquet",
             "picks": config.PREDICTIONS_DIR / "picks.parquet",
             "snapshots": config.PREDICTIONS_DIR / "snapshots.parquet",
             "odds": config.ODDS_DIR / "odds.parquet"}
    import pyarrow.parquet as pq
    out = {}
    for k, f in files.items():
        if f.exists():
            out[k] = {"rows": int(pq.ParquetFile(f).metadata.num_rows),
                      "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))}
    return out


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "statsmodels", "matplotlib", "reportlab"):
        try:
            libs[name] = importlib.import_module(name).__version__
        except Exception:
            libs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": libs,
            "git_sha": git_sha(), "n_jobs": N_JOBS, "seed": SEED, "n_boot": N_BOOT}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    from scripts import report_kit  # noqa: F401  (applies the series' chart style)
    import matplotlib.pyplot as _plt
    return _plt


GROUP_RU = {"TOTAL": "Тотал голов", "HANDICAP": "Азиатская фора", "1X2": "1X2", "BTTS": "Обе забьют",
            "DOUBLE_CHANCE": "Двойной шанс", "CORNERS": "Угловые"}
