"""Shared helpers of the v6.2 data-quality study (paths, IO, bootstrap, git access, run information).

Seasons are named by their starting year. Model evaluation periods (README of experiments/v6_2):

    2023, 2024   walk-forward refits of the production models (fit on all seasons before)
    2025         the production back-test bundle ``backtest_2526`` (trained to 01.06.2025) -> "for information"
    2026         the active production bundle (trained to 31.05.2026) -> 2026/27 so far, not an untouched holdout

Resources: the machine is shared, so every model uses at most N_JOBS threads and only the needed columns are read.
"""
from __future__ import annotations

import io
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
REFIT_SEASONS = (2023, 2024)
BACKTEST_SEASON = 2025
LIVE_SEASON = 2026
EVAL_SEASONS = (2023, 2024, 2025, 2026)
BACKTEST_VERSION = "backtest_2526"
PROD_VERSION = "prod_20261012"
PHANTOMS = {"B1-20261104-kortrijk-kv-mechelen": ("B1-20261104-kortrijk-mechelen", "a"),
            "E1-20261103-preston-sheffield-utd": ("E1-20261103-preston-sheffield-united", "a"),
            "E2-20261104-oxford-utd-reading": ("E2-20261104-oxford-reading", "h")}

DATA = ROOT / "local_data"
FEATURES_FILE = DATA / "processed" / "features.parquet"
LIVE_FILE = DATA / "results_live" / "fixtures.parquet"
FIXTURES_FILE = DATA / "fixtures" / "fixtures.csv"
PRED_DIR = DATA / "predictions"

log = logging.getLogger("dq62")


def setup_logging(name: str = "dq62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def season_label(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


# ------------------------------------------------------------------------------------------------ IO
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
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
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))
    return path


def load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def features(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(FEATURES_FILE, columns=columns)


# ------------------------------------------------------------------------------------------------ git (read only)
def git(*args) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def git_sha() -> str:
    try:
        return git("rev-parse", "HEAD").decode().strip()
    except Exception:
        return "unknown"


def file_versions(path: str) -> list[tuple[str, pd.Timestamp, str]]:
    """(sha, commit time as naive UTC, subject) of every commit that changed ``path``, oldest first."""
    out = git("log", "--format=%H|%aI|%s", "--", path).decode().splitlines()
    rows = []
    for line in out:
        sha, when, subject = line.split("|", 2)
        ts = pd.Timestamp(when).tz_convert("UTC").tz_localize(None)
        rows.append((sha, ts, subject))
    return rows[::-1]


def read_version_bytes(sha: str, path: str) -> bytes:
    return git("show", f"{sha}:{path}")


def read_version_csv(sha: str, path: str) -> pd.DataFrame:
    raw = read_version_bytes(sha, path)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def read_version_parquet(sha: str, path: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_version_bytes(sha, path)), columns=columns)


# ------------------------------------------------------------------------------------------------ statistics
def _boot_idx(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(200, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def mean_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean and 95 % percentile bootstrap interval (resampled by match)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci95": [None, None]}
    means = np.concatenate([x[idx].mean(1) for idx in _boot_idx(len(x), n_boot, seed)])
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(b - a) over the same matches with a bootstrap interval; NaN pairs are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    out = mean_ci(d, n_boot, seed)
    out["p_positive"] = (float(np.mean(np.concatenate([d[idx].mean(1) for idx in _boot_idx(len(d), n_boot, seed)])
                                       > 0)) if len(d) >= 2 else None)
    return out


def group_diff_ci(x_a, x_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(x_b) - mean(x_a) for two independent groups, each resampled separately."""
    a, b = np.asarray(x_a, float), np.asarray(x_b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"diff": None, "ci95": [None, None], "n_a": int(len(a)), "n_b": int(len(b))}
    ma = np.concatenate([a[i].mean(1) for i in _boot_idx(len(a), n_boot, seed)])
    mb = np.concatenate([b[i].mean(1) for i in _boot_idx(len(b), n_boot, seed + 1)])
    d = mb - ma
    return {"diff": float(b.mean() - a.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "n_a": int(len(a)), "n_b": int(len(b)), "p_b_greater": float(np.mean(d > 0))}


def roi_summary(profit, won=None) -> dict:
    """profit per staked bet (odds - 1 / -1); ROI as a fraction with a bootstrap interval."""
    p = np.asarray(profit, float)
    p = p[np.isfinite(p)]
    ci = mean_ci(p)
    out = {"bets": int(len(p)), "roi": ci["mean"], "roi_ci95": ci["ci95"], "profit": float(p.sum()) if len(p) else 0.0}
    if won is not None:
        w = np.asarray(won, float)
        w = w[np.isfinite(w)]
        out["hit_rate"] = float(w.mean()) if len(w) else None
        out["settled"] = int(len(w))
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    items = sorted(((k, v) for k, v in pvalues.items() if v is not None), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ------------------------------------------------------------------------------------------------ run information
def data_version() -> dict:
    files = [FEATURES_FILE, LIVE_FILE, FIXTURES_FILE, *(PRED_DIR / f for f in
             ("snapshots.parquet", "picks.parquet", "markets.parquet", "versions.parquet")),
             DATA / "teams" / "provider_ids.json", DATA / "api" / "unmatched_teams.csv",
             DATA / "api" / "team_id_conflicts.csv"]
    out = {}
    for f in files:
        if f.exists():
            info = {"mtime": pd.Timestamp(f.stat().st_mtime, unit="s").isoformat(), "bytes": f.stat().st_size}
            if f.suffix == ".parquet":
                import pyarrow.parquet as pq
                info["rows"] = pq.ParquetFile(f).metadata.num_rows
            out[str(f.relative_to(ROOT))] = info
    res = sorted((DATA / "results").glob("*/*.csv"))
    out["local_data/results"] = {"files": len(res),
                                 "latest_mtime": pd.Timestamp(max(p.stat().st_mtime for p in res), unit="s").isoformat()}
    return out


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "matplotlib", "reportlab", "pypdfium2"):
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
        log.info("step %s done at %.1f s", name, self.marks[name])
        return self.marks[name]
