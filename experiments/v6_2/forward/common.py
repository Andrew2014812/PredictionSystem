"""Shared helpers of the v6.2 forward-validation study (experiments/v6_2/forward)."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
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

log = logging.getLogger("forward62")


def setup_logging(name: str = "forward62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


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
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    import importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "pyarrow"):
        try:
            libs[name] = importlib.import_module(name).__version__
        except Exception:
            libs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": libs,
            "git_sha": git_sha()}
