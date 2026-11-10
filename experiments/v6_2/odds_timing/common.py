"""Shared helpers of the v6.2 odds-timing study (paths, clocks, bootstrap, result files).

Clocks. The production tables use two clocks (src/prediction/store.py, src/providers/schema.py):

    generated_at                           naive UTC
    kickoff, collected_at, priced_at,      naive UK local time (Europe/London: BST = UTC+1 until 25.10.2026,
    fetched_at, odds_priced_at             GMT = UTC afterwards)

Every comparison in this study is done in UTC (``uk_to_utc``); comparing the raw columns would put a quote
published up to one hour BEFORE a prediction after it during British summer time.
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
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_JOBS = 2
SEED = 20261109
N_BOOT = 2000
LIVE = 2026
DATA = ROOT / "local_data"
ODDS_FILE = DATA / "odds" / "odds.parquet"
PRED_DIR = DATA / "predictions"

log = logging.getLogger("timing62")


def setup_logging(name: str = "timing62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


# ------------------------------------------------------------------------------------------------ clocks
def uk_to_utc(values) -> pd.Series:
    """Naive UK local time -> naive UTC (ambiguous / missing hours of the clock change -> the later reading)."""
    s = pd.to_datetime(pd.Series(values))
    loc = s.dt.tz_localize("Europe/London", ambiguous=False, nonexistent="shift_forward")
    return loc.dt.tz_convert("UTC").dt.tz_localize(None)


# ------------------------------------------------------------------------------------------------ bootstrap
def _boot_index(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(200, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def mean_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean and 95 % percentile bootstrap interval (NaN ignored)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci95": [None, None]}
    means = np.concatenate([x[idx].mean(1) for idx in _boot_index(len(x), n_boot, seed)])
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def grouped_mean_ci(x, groups, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of x with a CLUSTER bootstrap over ``groups`` (e.g. matches: several selections of one match)."""
    x = np.asarray(x, float)
    g = pd.factorize(pd.Series(groups))[0]
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci95": [None, None]}
    k = g.max() + 1
    sums, cnts = np.bincount(g, x, k), np.bincount(g, None, k)
    means = []
    for idx in _boot_index(k, n_boot, seed):
        means.append(sums[idx].sum(1) / cnts[idx].sum(1))
    means = np.concatenate(means)
    return {"n": int(len(x)), "clusters": int(k), "mean": float(x.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def roi_summary(profit, odds=None, won=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """ROI (= mean profit per unit stake) with its bootstrap interval; profit NaN = no bet."""
    p = np.asarray(profit, float)
    m = np.isfinite(p)
    out = mean_ci(p[m], n_boot, seed)
    out = {"bets": out["n"], "roi": out["mean"], "roi_ci95": out["ci95"], "profit": float(p[m].sum())}
    if odds is not None:
        o = np.asarray(odds, float)[m]
        out["avg_odds"] = float(o.mean()) if len(o) else None
    if won is not None:
        w = np.asarray(won, float)[m]
        out["hit_rate"] = float(np.nanmean(w)) if len(w) else None
    return out


def diff_ci(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(b) - mean(a) for two INDEPENDENT samples (each resampled on its own)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"diff": None, "ci95": [None, None]}
    ma = np.concatenate([a[i].mean(1) for i in _boot_index(len(a), n_boot, seed)])
    mb = np.concatenate([b[i].mean(1) for i in _boot_index(len(b), n_boot, seed + 1)])
    d = mb - ma
    return {"diff": float(b.mean() - a.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_two_sided": float(min(1.0, 2 * min((d > 0).mean(), (d < 0).mean())))}


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(b - a) over the same units (matches), bootstrap over units; two-sided bootstrap p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "diff": None, "ci95": [None, None], "p_two_sided": None}
    means = np.concatenate([d[i].mean(1) for i in _boot_index(len(d), n_boot, seed)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": int(len(d)), "diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)),
                                                                float(np.quantile(means, 0.975))],
            "p_two_sided": float(min(1.0, max(p, 1.0 / n_boot)))}


def holm(pvalues: dict) -> dict:
    """Holm–Bonferroni adjusted p-values for {name: p}."""
    items = [(k, v) for k, v in pvalues.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ------------------------------------------------------------------------------------------------ de-margining
def proportional(implied: np.ndarray) -> np.ndarray:
    pi = np.asarray(implied, float)
    return pi / pi.sum(1, keepdims=True)


def shin(implied: np.ndarray) -> np.ndarray:
    """Shin (1993) probabilities from an (n, k) array of 1/odds (bisection on the insider share z)."""
    pi = np.asarray(implied, float)
    s = pi.sum(1, keepdims=True)
    lo, hi = np.zeros((len(pi), 1)), np.full((len(pi), 1), 0.5)

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / s) - z) / (2 * (1 - z))

    for _ in range(60):
        mid = (lo + hi) / 2
        tot = probs(mid).sum(1, keepdims=True)
        hi = np.where(tot < 1, mid, hi)
        lo = np.where(tot >= 1, mid, lo)
    p = probs((lo + hi) / 2)
    return p / p.sum(1, keepdims=True)


def power(implied: np.ndarray) -> np.ndarray:
    """Power method: q_i = pi_i^k with k chosen so that sum q_i = 1 (bisection on k >= 1)."""
    pi = np.clip(np.asarray(implied, float), 1e-9, 1 - 1e-9)
    lo, hi = np.full((len(pi), 1), 0.5), np.full((len(pi), 1), 3.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = (pi ** mid).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot <= 1, mid, hi)
    q = pi ** ((lo + hi) / 2)
    return q / q.sum(1, keepdims=True)


def logarithmic(implied: np.ndarray) -> np.ndarray:
    """'Logarithmic' (odds-ratio) method: logit q_i = logit pi_i - c with c chosen so that sum q_i = 1."""
    pi = np.clip(np.asarray(implied, float), 1e-9, 1 - 1e-9)
    lg = np.log(pi / (1 - pi))
    lo, hi = np.full((len(pi), 1), -1.0), np.full((len(pi), 1), 3.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = (1 / (1 + np.exp(-(lg - mid)))).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot <= 1, mid, hi)
    q = 1 / (1 + np.exp(-(lg - (lo + hi) / 2)))
    return q / q.sum(1, keepdims=True)


def additive(implied: np.ndarray) -> np.ndarray:
    """Additive method: subtract the same share of the margin from every outcome (clipped at a small floor)."""
    pi = np.asarray(implied, float)
    k = pi.shape[1]
    q = np.clip(pi - (pi.sum(1, keepdims=True) - 1) / k, 1e-4, None)
    return q / q.sum(1, keepdims=True)


METHODS = {"prop": proportional, "shin": shin, "power": power, "log": logarithmic, "add": additive}
METHOD_RU = {"prop": "пропорциональный", "shin": "Шин", "power": "степенной", "log": "логарифмический (odds ratio)",
             "add": "аддитивный"}


def demargin(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    """(n, k) decimal prices of ONE complete market -> (n, k) probabilities; rows with a NaN stay NaN."""
    o = np.asarray(odds, float)
    out = np.full(o.shape, np.nan)
    ok = np.all(np.isfinite(o) & (o > 1.0), 1)
    if ok.any():
        out[ok] = METHODS[method](1.0 / o[ok])
    return out


# ------------------------------------------------------------------------------------------------ IO
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return None if pd.isna(o) else str(o)
    if o is pd.NaT:
        return None
    return o


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


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def data_version(paths) -> dict:
    """Row counts and modification times of the input files."""
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            out[str(p.relative_to(ROOT))] = None
            continue
        info = {"mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
                "bytes": p.stat().st_size}
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            info["rows"] = pq.ParquetFile(p).metadata.num_rows
        out[str(p.relative_to(ROOT))] = info
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
    from scripts import report_kit  # noqa: F401  (series chart style)
    import matplotlib.pyplot as _plt
    return _plt


GROUP_RU = {"TOTAL": "Тотал голов", "HANDICAP": "Азиатская фора", "1X2": "1X2", "BTTS": "Обе забьют",
            "DOUBLE_CHANCE": "Двойной шанс", "CORNERS": "Угловые", "EHANDICAP": "Европейская фора",
            "EXACT_SCORE": "Точный счёт"}


def market_group(market: str) -> str:
    if market in ("1X2", "BTTS", "DOUBLE_CHANCE", "EXACT_SCORE"):
        return market
    return market.split("_")[0]
