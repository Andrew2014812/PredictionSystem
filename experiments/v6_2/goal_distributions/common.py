"""Paths and helpers of the v6.2 Goal-Distribution study (statistics shared with market_models / main_calibration)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from experiments.v6_2.market_models.common import (  # noqa: F401  (re-exported)
    CURRENT, FOLDS, INFO, N_BOOT, N_JOBS, ROOT, SEASONS, SEED, SELECT, Timer, _clean, bin_brier, bin_ll, boot_stat,
    calib_curve, ci, data_version, ece, environment, git_sha, holm, multi_brier, multi_ll, p_two_sided, paired,
    period_label, plt, rps, season_label, summary_binary)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
LOGS = HERE / "logs"
for _d in (RESULTS, FIGURES, CACHE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("gd62")


def setup_logging(name: str = "gd62") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    return logging.getLogger(name)


def save_json(name: str, obj, folder: Path = RESULTS) -> Path:
    path = folder / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def load_json(name: str, default=None, folder: Path = RESULTS):
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
