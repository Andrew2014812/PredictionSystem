"""Shared helpers of the v6.3 studies: folders, run information, statistics.

Seasons (season = start year): the studies of v6.3 replay the whole season 2026/27 (``CURRENT``). That season was
already examined by the v6.1 / v6.2 studies, so it is NOT an independent holdout: a result found on it is
reported as "chronological check on an already studied season", never as proof. The only independent evidence
is forward (matches played after a hypothesis was frozen, see ``preregister``).
"""
from __future__ import annotations

import os

# thread pools of numpy / BLAS / OpenMP are sized when numpy is first imported: limit them before that
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

V63 = Path(__file__).resolve().parent
SHARED_CACHE = V63 / "replay" / "cache"
SHARED_RESULTS = V63 / "replay" / "results"
for _d in (SHARED_CACHE, SHARED_RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

CURRENT = 2026
SEED = 20261118
N_BOOT = 2000
N_JOBS = 2                       # the machine is shared: at most 2 threads per model

DATA_FILES = {
    "features": ROOT / "local_data/processed/features.parquet",
    "odds": ROOT / "local_data/odds/odds.parquet",
    "picks": ROOT / "local_data/predictions/picks.parquet",
    "markets": ROOT / "local_data/predictions/markets.parquet",
    "snapshots": ROOT / "local_data/predictions/snapshots.parquet",
}


def study_dirs(name: str) -> dict[str, Path]:
    """results / figures / cache / logs folders of a study (created)."""
    base = V63 / name
    out = {k: base / k for k in ("results", "figures", "cache", "logs")}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(name: str, logfile: Path | None = None) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers, force=True)
    for noisy in ("hyperopt", "src", "matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def data_version() -> dict:
    out = {}
    for key, path in DATA_FILES.items():
        if path.exists():
            st = path.stat()
            out[key] = {"path": str(path.relative_to(ROOT)), "bytes": st.st_size,
                        "mtime": pd.Timestamp(st.st_mtime, unit="s").isoformat(timespec="seconds")}
    try:
        out["data_commit"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%h %cs %s", "--", "local_data/predictions/picks.parquet"],
            cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        pass
    return out


def run_info(study: str, started: float, params: dict | None = None, extra: dict | None = None) -> dict:
    import sklearn
    import xgboost
    return {"study": study, "git_sha": git_sha(), "created": pd.Timestamp.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1), "python": platform.python_version(),
            "libraries": {"pandas": pd.__version__, "numpy": np.__version__, "xgboost": xgboost.__version__,
                          "sklearn": sklearn.__version__},
            "data": data_version(), "params": params or {}, **(extra or {})}


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ statistics
def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap index chunks (resampling by match)."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(200, n_boot - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(values, a)), float(np.quantile(values, 1 - a))]


def roi_ci(profit: np.ndarray, staked: np.ndarray) -> dict:
    """ROI = Σ profit / Σ stake over bets, with a bootstrap 95 % interval (resampling matches)."""
    profit = np.nan_to_num(np.asarray(profit, float))
    staked = np.asarray(staked, float)
    n = len(profit)
    if staked.sum() == 0:
        return {"roi": None, "ci95": [None, None], "bets": 0}
    vals = []
    for idx in boot_idx(n):
        s = staked[idx].sum(1)
        vals.append(np.where(s > 0, profit[idx].sum(1) / np.maximum(s, 1e-9), np.nan))
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    return {"roi": float(profit.sum() / staked.sum()), "ci95": ci(vals), "bets": int(staked.sum())}


def paired_roi_diff(p_new: np.ndarray, s_new: np.ndarray, p_old: np.ndarray, s_old: np.ndarray) -> dict:
    """ROI(new) − ROI(old) on the same matches (arrays aligned by match; profit 0 / stake 0 where no bet).
    Paired bootstrap by match; two-sided p-value of "difference = 0"."""
    p_new, p_old = np.nan_to_num(np.asarray(p_new, float)), np.nan_to_num(np.asarray(p_old, float))
    s_new, s_old = np.asarray(s_new, float), np.asarray(s_old, float)
    point = p_new.sum() / max(s_new.sum(), 1e-9) - p_old.sum() / max(s_old.sum(), 1e-9)
    vals = []
    for idx in boot_idx(len(p_new)):
        a = p_new[idx].sum(1) / np.maximum(s_new[idx].sum(1), 1e-9)
        b = p_old[idx].sum(1) / np.maximum(s_old[idx].sum(1), 1e-9)
        vals.append(a - b)
    vals = np.concatenate(vals)
    p = float(min(1.0, 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))))
    return {"diff": float(point), "ci95": ci(vals), "p": p}


def paired_mean_diff(x_new: np.ndarray, x_old: np.ndarray) -> dict:
    d = np.asarray(x_new, float) - np.asarray(x_old, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"diff": None, "ci95": [None, None], "p": None, "n": 0}
    vals = np.concatenate([d[idx].mean(1) for idx in boot_idx(len(d))])
    p = float(min(1.0, 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))))
    return {"diff": float(d.mean()), "ci95": ci(vals), "p": p, "n": int(len(d))}


def holm(pvalues: dict) -> dict:
    keys = sorted([k for k, v in pvalues.items() if v is not None and np.isfinite(v)], key=lambda k: pvalues[k])
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * pvalues[k]))
        out[k] = running
    return out
