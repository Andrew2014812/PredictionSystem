"""Candidate tables of the calibration and meta-selection studies (one row per match x selection).

* ``backtest()`` — season 2025/26: the stored production backtest (``local_data/predictions/markets.parquet`` rows
  of the matches whose source is ``backtest`` in picks, model ``backtest_2526`` trained on matches up to
  2025-06-01, so every 2025/26 prediction is out of sample). Every selection of every market; real prices only
  where production attached one (Football-Data Bet365: 1X2, total 2.5, the match's Asian line). Settled with the
  score and corners stored in picks, by production's ``settle_selection``.
* ``season(universe)`` — season 2026/27: the shared replay candidates (``replay/cache/candidates_{close,t09}``),
  production model ``prod_20261012`` and the price known at the decision moment of the universe.

Every row carries the no-refund probability ``pc = p / (1 - p_push)`` (identical to ``p`` except for Asian whole
lines) and ``y`` = 1 WON / 0 LOST / NaN (refund, not played, unknown). ``month`` is the calendar month of the match
(the last days of July are joined to August: the seasons start on the last weekend of July).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

STUDY = "calibration"
DIRS = C.study_dirs(STUDY)
GROUPS_ALL = ["1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "CORNERS", "EHANDICAP", "EXACT_SCORE"]
MAIN_GROUPS = ["1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "CORNERS"]
COLS = ["match_id", "market", "selection", "group", "probability", "odds", "bookmaker", "provider", "ev",
        "push_probability", "priced_at", "date", "league", "status", "profit", "bet"]
GROUP_RU = {"1X2": "1X2", "DOUBLE_CHANCE": "Двойной шанс", "TOTAL": "Тоталы голов", "BTTS": "Обе забьют",
            "HANDICAP": "Азиатские форы", "CORNERS": "Угловые", "EHANDICAP": "Европейские форы",
            "EXACT_SCORE": "Точный счёт"}


def month_key(date: pd.Series) -> pd.Series:
    d = pd.to_datetime(date)
    d = d.where(~((d.dt.month == 7) & (d.dt.day >= 20)), d + pd.offsets.MonthBegin(1))
    return d.dt.to_period("M").astype(str)


def _finish(c: pd.DataFrame) -> pd.DataFrame:
    c = c.reset_index(drop=True)
    c["probability"] = c["probability"].astype(float)
    c["push_probability"] = pd.to_numeric(c["push_probability"], errors="coerce").fillna(0.0).astype(float)
    c["pc"] = (c["probability"] / np.clip(1 - c["push_probability"], 1e-6, None)).clip(1e-4, 1 - 1e-4)
    c["y"] = np.where(c["status"] == "WON", 1.0, np.where(c["status"] == "LOST", 0.0, np.nan))
    c["month"] = month_key(c["date"])
    for col in ("market", "selection", "group", "league"):
        c[col] = c[col].astype(str)
    c["key"] = c["group"] + "|" + c["selection"]
    return c


def backtest(refresh: bool = False) -> pd.DataFrame:
    path = DIRS["cache"] / "backtest_candidates.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    picks = pd.read_parquet(C.DATA_FILES["picks"], columns=["match_id", "source", "league", "date", "home_goals",
                                                            "away_goals", "total_corners", "model_version"])
    bt = picks.loc[picks["source"] == "backtest"].drop_duplicates("match_id").set_index("match_id")
    mk = pd.read_parquet(C.DATA_FILES["markets"])
    mk = mk.loc[mk["match_id"].isin(bt.index)].copy()
    mk["odds"] = pd.to_numeric(mk["odds"], errors="coerce")
    mk = mk.join(bt[["league", "date"]], on="match_id")
    res = bt[["home_goals", "away_goals", "total_corners"]].astype(float)
    c = EV.settle_rows(mk, res)
    c["date"] = pd.to_datetime(c["date"])
    c = _finish(c.reindex(columns=COLS))
    c["model_version"] = "backtest_2526"
    c.to_parquet(path)
    return c


def season(universe: str = "close") -> pd.DataFrame:
    c = pd.read_parquet(C.SHARED_CACHE / f"candidates_{universe}.parquet", columns=COLS)
    c = _finish(c)
    c["model_version"] = "prod_20261012"
    return c


def matches() -> pd.DataFrame:
    return pd.read_parquet(C.SHARED_CACHE / "matches.parquet")


def results(m: pd.DataFrame | None = None) -> pd.DataFrame:
    m = matches() if m is None else m
    r = m.loc[m["played"].astype(bool), ["match_id", "home_goals", "away_goals", "total_corners"]]
    return r.set_index("match_id")


def baseline(name: str) -> pd.DataFrame:
    """baseline_a, baseline_b_close, baseline_b_t09 (played matches only for B)."""
    b = pd.read_parquet(C.SHARED_CACHE / f"{name}.parquet")
    if name.startswith("baseline_b"):
        b = b.loc[b["status"].isin(["WON", "LOST", "PUSH"])]
    return b.reset_index(drop=True)
