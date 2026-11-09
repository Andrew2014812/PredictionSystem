"""Shared helpers of the v6.2 data-sources study: paths, git history reader, result files.

Everything is read from the committed data of this checkout (``local_data/``) and from its git history
(``git show <sha>:<path>``) — no network, no API call. Clocks as in production:

    generated_at                                 naive UTC
    kickoff, collected_at, priced_at, fetched_at naive UK local time (Europe/London)
    git commit dates                             with offset, converted to UTC here
"""
from __future__ import annotations

import io
import json
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

DATA = ROOT / "local_data"
API = DATA / "api"
SEASON = 2026
SEASON_CODE = "2627"
LIVE_FILE = "local_data/results_live/fixtures.parquet"
ODDS_FILE = DATA / "odds" / "odds.parquet"
PICKS_FILE = DATA / "predictions" / "picks.parquet"
TIMING_AUDIT = ROOT / "experiments" / "v6_2" / "odds_timing" / "results" / "audit_picks.csv"
KYIV = "Europe/Kyiv"


# ------------------------------------------------------------------------------------------------ git
def git(*args: str, binary: bool = False):
    out = subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
    return out if binary else out.decode("utf-8", "replace")


def commits(path: str) -> pd.DataFrame:
    """Every commit that changed ``path`` (oldest first): sha, UTC time, subject."""
    raw = git("log", "--format=%H|%aI|%s", "--", path)
    rows = []
    for line in raw.splitlines():
        sha, when, subject = line.split("|", 2)
        rows.append({"sha": sha, "at_utc": pd.Timestamp(when).tz_convert("UTC").tz_localize(None),
                     "subject": subject})
    return pd.DataFrame(rows).sort_values("at_utc").reset_index(drop=True)


def blob(sha: str, path: str) -> bytes | None:
    try:
        return git("show", f"{sha}:{path}", binary=True)
    except subprocess.CalledProcessError:
        return None


def parquet_at(sha: str, path: str, columns=None) -> pd.DataFrame | None:
    raw = blob(sha, path)
    if raw is None:
        return None
    return pd.read_parquet(io.BytesIO(raw), columns=columns)


def csv_at(sha: str, path: str, **kw) -> pd.DataFrame | None:
    raw = blob(sha, path)
    if raw is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip", **kw)
    except Exception:                                                     # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ clocks
def uk_to_utc(values) -> pd.Series:
    s = pd.to_datetime(pd.Series(values))
    loc = s.dt.tz_localize("Europe/London", ambiguous=False, nonexistent="shift_forward")
    return loc.dt.tz_convert("UTC").dt.tz_localize(None)


def utc_to_kyiv(values) -> pd.Series:
    s = pd.to_datetime(pd.Series(values))
    return s.dt.tz_localize("UTC").dt.tz_convert(KYIV).dt.tz_localize(None)


# ------------------------------------------------------------------------------------------------ files
def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, set):
        return sorted(v)
    return str(v)


def _sanitize(obj):
    """NaN -> None everywhere (strict JSON), numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if obj != obj else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_sanitize(obj), indent=1, ensure_ascii=False, default=_clean), encoding="utf-8")
    return path


def load_json(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_csv(name: str, df: pd.DataFrame) -> Path:
    path = RESULTS / name
    df.to_csv(path, index=False)
    return path


def run_info(params: dict, started: float) -> dict:
    files = {}
    for p in (LIVE_FILE, "local_data/odds/odds.parquet", "local_data/predictions/picks.parquet",
              "local_data/api/diagnostics/history.csv", "local_data/api/oddspapi/ledger.json",
              "local_data/api/the_odds_api/ledger.json"):
        f = ROOT / p
        files[p] = {"bytes": f.stat().st_size if f.exists() else None,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)) if f.exists()
                    else None}
    return {"git_sha": git("rev-parse", "HEAD").strip(), "data_commit": git("log", "-1", "--format=%H %aI",
                                                                              "--", "local_data").strip(),
            "files": files, "python": platform.python_version(), "pandas": pd.__version__,
            "params": params, "seconds": round(time.time() - started, 1),
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
