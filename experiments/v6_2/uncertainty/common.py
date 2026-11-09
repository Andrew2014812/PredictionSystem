"""Shared helpers of the v6.2 prediction-uncertainty study.

Seasons (season = start year):

    SELECT   2023/24, 2024/25   every rule (threshold, measure, filter) is chosen here only
    INFO     2025/26            reported "for information" (used by earlier studies)
    CURRENT  2026/27 so far     the current season, already looked at by earlier studies (NOT a holdout)

Walk-forward: every model that predicts season s is fitted on the seasons before s only (the v6.1 production
setup, experiments/v6_1/main/base.py); every threshold applied to a match uses only matches played earlier.
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

V61_CACHE = ROOT / "experiments" / "v6_1" / "main" / "cache"     # walk-forward tables of v6.1 (not in git)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_JOBS = 2
SELECT = (2023, 2024)
INFO = 2025
CURRENT = 2026
SEASONS = (2023, 2024, 2025, 2026)
SEED = 20261109
N_BOOT = 2000

# ensemble families: name -> number of members
FAMILIES = {"seed": 8, "boot": 8, "sub": 6}
SUB_FRACTION = 0.7

log = logging.getLogger("unc62")


def setup_logging(name: str = "unc62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


def period_label(s: int) -> str:
    return season_label(s) + {INFO: " (инф.)", CURRENT: " (текущий)"}.get(s, "")


# ------------------------------------------------------------------------------------------------ statistics
def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap index chunks (resampling by match), drawn in chunks to keep memory small."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(250, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def mean_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else np.nan, "ci95": [np.nan, np.nan], "n": int(len(x))}
    b = np.concatenate([x[i].mean(1) for i in boot_idx(len(x), n_boot, seed)])
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
            "n": int(len(x))}


def gap_ci(p, y, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Over-estimation gap = mean(p) - hit rate with a bootstrap 95 % interval."""
    d = np.asarray(p, float) - np.asarray(y, float)
    out = mean_ci(d, n_boot, seed)
    out["hit_rate"] = float(np.mean(y)) if len(y) else np.nan
    out["mean_p"] = float(np.mean(p)) if len(p) else np.nan
    return out


def paired_nanmean_diff(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap over matches of nanmean(b) - nanmean(a) (NaN = the match does not count for that variant).
    Two-sided p = 2 min(P(d <= 0), P(d >= 0))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = []
    for idx in boot_idx(len(a), n_boot, seed):
        with np.errstate(all="ignore"):
            diffs.append(np.nanmean(b[idx], 1) - np.nanmean(a[idx], 1))
    d = np.concatenate(diffs)
    d = d[np.isfinite(d)]
    est = float(np.nanmean(b) - np.nanmean(a))
    p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if len(d) else np.nan
    return {"diff": est, "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))] if len(d) else
            [np.nan, np.nan], "p": p}


def holm(pvalues: dict) -> dict:
    keys = [k for k, v in pvalues.items() if v is not None and np.isfinite(v)]
    order = sorted(keys, key=lambda k: pvalues[k])
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def roc_auc(score, label) -> float:
    """AUC of ``score`` for ``label`` = 1 (Mann-Whitney)."""
    s, y = np.asarray(score, float), np.asarray(label, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else str(o)
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


def data_version() -> dict:
    out = {}
    for key, rel in (("features", "local_data/processed/features.parquet"),
                     ("picks", "local_data/predictions/picks.parquet"),
                     ("markets", "local_data/predictions/markets.parquet"),
                     ("versions", "local_data/predictions/versions.parquet"),
                     ("snapshots", "local_data/predictions/snapshots.parquet")):
        path = ROOT / rel
        if path.exists():
            import pyarrow.parquet as pq
            out[key] = {"rows": int(pq.ParquetFile(path).metadata.num_rows),
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}
    return out


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "reportlab"):
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
