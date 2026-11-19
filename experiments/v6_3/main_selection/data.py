"""Candidate tables used by the Main-selection and market-profitability studies.

Three universes, one row per (match, selection), production market mathematics in all of them:

* ``close`` / ``t09`` — season 2026/27, the shared replay (``experiments/v6_3/replay/cache/candidates_*``):
  production models ``prod_20261012``, prices of the decision moment (last quote before kick-off / known at T).
* ``bt`` — season 2025/26, the production backtest stored by the site (``source = backtest``, model
  ``backtest_2526`` trained on matches up to 2025-06-01, i.e. out of sample for 2025/26) with the prices that
  production attached to it: Football-Data Bet365 1X2, total 2.5 and the one Asian line of the match. All
  other markets have no price there. Used ONLY to choose parameters (never reported as evidence).

Derived columns added here (vectorised, used by ``rules.py``):
``has_odds``, ``eff_odds`` (real price or break-even price), ``p_mkt`` (de-margined bookmaker probability of the
selection when every selection of its market is priced, proportional method, on the refund-free scale times
1 − P(push) of the model), ``excluded`` (EHANDICAP / EXACT_SCORE, never Main), ``month``, ``mi`` (match index).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

NAME = "main_selection"
DIRS = C.study_dirs(NAME)
CACHE = DIRS["cache"]
EXCLUDED = {"EHANDICAP", "EXACT_SCORE"}
SETTLED = ("WON", "LOST", "PUSH")
COLS = ["match_id", "market", "selection", "group", "probability", "odds", "bookmaker", "provider", "ev",
        "push_probability", "priced_at", "date", "league", "home_team", "away_team", "status", "profit", "bet"]


def _backtest() -> pd.DataFrame:
    path = CACHE / "candidates_bt.parquet"
    if path.exists():
        return pd.read_parquet(path)
    picks = pd.read_parquet(C.DATA_FILES["picks"], columns=["match_id", "source", "season", "role"])
    ids = set(picks.loc[(picks["source"] == "backtest") & (picks["role"] == "main"), "match_id"])
    mk = pd.read_parquet(C.DATA_FILES["markets"])
    mk = mk.loc[mk["match_id"].isin(ids)]
    f = pd.read_parquet(C.DATA_FILES["features"], columns=["match_id", "season", "date", "league", "home_team",
                                                             "away_team", "played", "home_goals", "away_goals",
                                                             "total_corners"])
    f = f.loc[f["match_id"].isin(ids)].copy()
    f["date"] = pd.to_datetime(f["date"])
    res = f.loc[f["played"].astype(bool)].set_index("match_id")[["home_goals", "away_goals", "total_corners"]]
    c = EV.candidates(mk, f, res)
    c = c.reindex(columns=COLS)
    c.to_parquet(path)
    return c


def raw(universe: str) -> pd.DataFrame:
    if universe == "bt":
        return _backtest()
    c = pd.read_parquet(C.SHARED_CACHE / f"candidates_{universe}.parquet")
    return c.reindex(columns=COLS)


def market_probability(c: pd.DataFrame) -> np.ndarray:
    """De-margined (proportional) bookmaker probability of each selection, NaN unless its whole market is priced.
    Double chance: the three selections sum to 2. Whole Asian lines: the two sides give P(win | no refund),
    multiplied by the model's 1 − P(push) to be on the scale of ``probability``."""
    inv = 1.0 / c["odds"].astype(float)
    key = [c["match_id"], c["market"]]
    tot = inv.groupby(key).transform("sum")
    n_priced = c["odds"].notna().groupby(key).transform("sum")
    n_all = c["odds"].groupby(key).transform("size")
    full = (n_priced == n_all) & (n_all >= 2)
    scale = np.where(c["group"] == "DOUBLE_CHANCE", 2.0, 1.0)
    p = inv / tot * scale * (1.0 - c["push_probability"].astype(float))
    p = p.where(full & ~c["group"].isin(["EXACT_SCORE"]))
    return p.to_numpy(float)


def load(universe: str, played_only: bool = True) -> pd.DataFrame:
    """Candidate table of a universe with the derived columns, sorted by match (``mi`` = match index)."""
    c = raw(universe)
    c["odds"] = pd.to_numeric(c["odds"], errors="coerce").astype(float)
    c["probability"] = c["probability"].astype(float)
    c["push_probability"] = c["push_probability"].astype(float).fillna(0.0)
    c.loc[c["odds"] <= 1.0, "odds"] = np.nan
    if played_only:
        done = c.groupby("match_id")["status"].transform(lambda s: s.isin(SETTLED + ("UNKNOWN",)).any())
        c = c.loc[done]
    c = c.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    c["has_odds"] = c["odds"].notna()
    c["ev"] = np.where(c["has_odds"], c["probability"] * c["odds"] - (1 - c["push_probability"]), np.nan)
    c["eff_odds"] = np.where(c["has_odds"], c["odds"],
                             np.maximum(1 - c["push_probability"], 1e-9) / np.maximum(c["probability"], 1e-9))
    c["p_mkt"] = market_probability(c)
    c["excluded"] = c["group"].isin(EXCLUDED)
    # the season starts on 31 July: those few matches count with August
    c["month"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m").replace({"2026-07": "2026-08", "2025-07": "2025-08"})
    c["mi"] = pd.factorize(c["match_id"])[0]
    c["universe"] = universe
    return c
