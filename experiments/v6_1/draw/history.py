"""The 2026/27 production history of the website and the stored 2025/26 backtest (read-only).

    history()   one row per 2026/27 match of the unified history (a prediction saved before kick-off when it
                exists, otherwise the reconstruction from pre-match data — never both), with the recorded
                probabilities, recorded named outcome, model version, Main pick (odds, status, profit) and the
                recorded pre-match draw price, plus the result.
    backtest()  the 2025/26 predictions of the evaluation model backtest_2526 (what the Model Analysis page
                reports as the test season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.performance import unified_history
from src.features.store import load_features
from src.prediction import store

from . import common as C

_CACHE: dict = {}


def _results() -> pd.DataFrame:
    f = load_features()
    return f.loc[f["played"].astype(bool) & f["result"].notna(), ["match_id", "result", "home_goals", "away_goals"]]


def _attach(snap: pd.DataFrame, picks: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    out = snap[["match_id", "league", "season", "date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                "predicted_outcome", "model_version", "source", "generated_at"]].copy()
    out = out.rename(columns={"p_home": "pH", "p_draw": "pD", "p_away": "pA", "predicted_outcome": "recorded_outcome"})
    for c in ("pH", "pD", "pA"):
        out[c] = out[c].astype(float)
    out = out.merge(_results(), on="match_id", how="left")
    out = out.loc[out["result"].notna()].reset_index(drop=True)
    out["y"] = C.y_index(out["result"])
    main = picks.loc[picks["role"] == "main"].drop_duplicates("match_id").set_index("match_id")
    for col in ("market", "selection", "probability", "odds", "status", "profit", "kind"):
        out[f"main_{col}"] = out["match_id"].map(main[col])
    risk = picks.loc[picks["role"] == "risk"].drop_duplicates("match_id").set_index("match_id")
    for col in ("market", "selection", "odds", "status", "profit"):
        out[f"risk_{col}"] = out["match_id"].map(risk[col])
    one = markets.loc[markets["market"] == "1X2"]
    for sel, col in (("H", "odds_home"), ("D", "odds_draw"), ("A", "odds_away")):
        r = one.loc[one["selection"] == sel].drop_duplicates("match_id").set_index("match_id")
        out[col] = out["match_id"].map(r["odds"]).astype(float)
        if sel == "D":
            out["draw_bookmaker"] = out["match_id"].map(r["bookmaker"])
            out["draw_provider"] = out["match_id"].map(r["provider"])
    return out


def history() -> pd.DataFrame:
    if "history" not in _CACHE:
        picks = unified_history(store.load_picks())
        snap = store.load_snapshots()
        snap = snap.loc[snap["source"].isin(["live", "backfill"])]
        live = set(snap.loc[snap["source"] == "live", "match_id"])
        snap = pd.concat([snap.loc[snap["source"] == "live"],
                          snap.loc[(snap["source"] == "backfill") & ~snap["match_id"].isin(live)]])
        mk = store.load_markets()
        _CACHE["history"] = _attach(snap, picks, mk.loc[mk["match_id"].isin(snap["match_id"])])
    return _CACHE["history"]


def backtest() -> pd.DataFrame:
    if "backtest" not in _CACHE:
        picks = store.load_picks()
        picks = picks.loc[picks["source"] == "backtest"]
        snap = store.load_snapshots()
        snap = snap.loc[snap["source"] == "backtest"]
        mk = store.load_markets()
        _CACHE["backtest"] = _attach(snap, picks, mk.loc[mk["match_id"].isin(snap["match_id"])])
    return _CACHE["backtest"]


def probs(fr: pd.DataFrame) -> np.ndarray:
    return fr[["pH", "pD", "pA"]].to_numpy(float)
