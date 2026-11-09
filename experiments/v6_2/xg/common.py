"""Shared helpers of the v6.2 xG study (paths, periods, IO, bootstrap, metrics, environment).

Periods (season = start year). xG exists only in the Football-Data files of 2026/27, so the whole study
works inside that season with time-ordered splits:

    WARM-UP   31.07.2026 - 31.08.2026   only history for features and for the first refits
    EVAL      01.09.2026 - last played  every variant is refitted weekly on matches played BEFORE the week
                                        it predicts (expanding window) and scored on the same matches

2026/27 was looked at by the v6.1 studies, so it is not an untouched holdout; the only independent check
is forward (matches played after results/preregistration.json was committed).
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
LOGS = CACHE / "logs"
for _d in (RESULTS, FIGURES, CACHE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_JOBS = 2
SEASON = 2026
EVAL_START = pd.Timestamp("2026-09-01")
REFIT_DAYS = 7                       # weekly expanding-window refits
SEED = 20261109
N_BOOT = 2000
PRODUCTION_VERSION = "prod_20261012"  # active v6.1 model set (trained to 2026-05-31: seasons 2021-2025)

log = logging.getLogger("xg62")


def setup_logging(name: str = "xg62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


# ------------------------------------------------------------------------------------------------ IO
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
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


def data_version() -> dict:
    """Row counts and modification times of the input files."""
    out = {}
    files = [ROOT / "local_data/processed/features.parquet", ROOT / "local_data/predictions/markets.parquet",
             ROOT / "local_data/predictions/snapshots.parquet", ROOT / "local_data/predictions/picks.parquet",
             ROOT / "local_data/odds/odds.parquet"]
    files += sorted((ROOT / "local_data/results/2627").glob("*.csv"))
    for f in files:
        if not f.exists():
            continue
        rel = f.relative_to(ROOT).as_posix()
        info = {"mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
                "bytes": f.stat().st_size}
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            info["rows"] = pq.ParquetFile(f).metadata.num_rows
        else:
            info["rows"] = max(sum(1 for _ in f.open("rb")) - 1, 0)
        out[rel] = info
    return out


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "statsmodels", "matplotlib", "reportlab",
                 "pypdfium2"):
        try:
            libs[name] = importlib.import_module(name).__version__
        except Exception:
            libs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": libs,
            "git_sha": git_sha(), "n_jobs": N_JOBS, "seed": SEED, "n_boot": N_BOOT,
            "production_version": PRODUCTION_VERSION, "eval_start": str(EVAL_START.date()),
            "refit_days": REFIT_DAYS}


# ------------------------------------------------------------------------------------------------ bootstrap
def boot_index(n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Yields chunks of bootstrap row indices (resampling matches)."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(200, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def boot_mean_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of x with a 95 % percentile interval and the two-sided bootstrap p-value of mean = 0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float(np.mean(x)) if n else None, "ci95": [None, None], "p": None, "n": n}
    means = np.concatenate([x[idx].mean(1) for idx in boot_index(n, n_boot, seed)])
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "p": float(min(1.0, max(p, 1.0 / n_boot))), "n": n, "sd": float(x.std(ddof=1))}


def boot_ratio_diff(num_a, den_a, num_b, den_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap (by match) of sum(num_b)/sum(den_b) - sum(num_a)/sum(den_a) (e.g. ROI, hit rate)."""
    arr = [np.nan_to_num(np.asarray(v, float)) for v in (num_a, den_a, num_b, den_b)]
    n = len(arr[0])
    d0 = arr[2].sum() / max(arr[3].sum(), 1e-12) - arr[0].sum() / max(arr[1].sum(), 1e-12)
    diffs = []
    for idx in boot_index(n, n_boot, seed):
        a = arr[0][idx].sum(1) / np.maximum(arr[1][idx].sum(1), 1e-12)
        b = arr[2][idx].sum(1) / np.maximum(arr[3][idx].sum(1), 1e-12)
        diffs.append(b - a)
    d = np.concatenate(diffs)
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": float(d0), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p": float(min(1.0, max(p, 1.0 / n_boot)))}


def ratio_ci(num, den, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    num, den = np.nan_to_num(np.asarray(num, float)), np.nan_to_num(np.asarray(den, float))
    n = len(num)
    if n < 3 or den.sum() == 0:
        return [None, None]
    vals = np.concatenate([num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12) for idx in boot_index(n, n_boot, seed)])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (family = the given dict)."""
    items = [(k, v) for k, v in pvalues.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def n_required(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Matches needed for a two-sided paired z-test to detect the mean difference ``effect``."""
    from scipy.stats import norm
    if not sd or not effect or not np.isfinite(sd) or not np.isfinite(effect):
        return None
    return float(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / abs(effect)) ** 2)


# ------------------------------------------------------------------------------------------------ metrics
def ece(outcome, proba, bins: int = 10) -> float:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    if not len(o):
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(o[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(outcome, proba, bins: int = 10, min_n: int = 15) -> list[dict]:
    o, p = np.asarray(outcome, float), np.asarray(proba, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= min_n:
            ob = float(o[m].mean())
            out.append({"predicted": float(p[m].mean()), "observed": ob, "n": int(m.sum()),
                        "se": float(np.sqrt(max(ob * (1 - ob), 1e-9) / m.sum()))})
    return out


def poisson_dev(y, mu) -> np.ndarray:
    """Per-match Poisson deviance."""
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def binary_loss(o, p) -> tuple[np.ndarray, np.ndarray]:
    """Per-match log loss and Brier score of a binary event."""
    o = np.asarray(o, float)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p)), (p - o) ** 2


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    from scripts import report_kit  # noqa: F401  (applies the series' chart style)
    import matplotlib.pyplot as _plt
    return _plt


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def minutes(self) -> float:
        return round((time.time() - self.t0) / 60, 2)
