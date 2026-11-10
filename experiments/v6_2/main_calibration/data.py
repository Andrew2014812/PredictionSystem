"""Candidate tables of the Main-calibration study (one row per match x selection).

* 2022/23 ... 2025/26: the v6.1 walk-forward tables (experiments/v6_1/main/candidates.py): production mathematics on
  predictions of models fitted on earlier seasons; real prices = Football-Data Bet365 opening (1X2, total 2.5,
  the one Asian line of the match), as production attaches them. Reproduced exactly (results/cache_check.json).
* 2026/27: the production store (latest pre-kick-off version of every match of the unified history) with the
  recorded prices, missing ones filled with the last OddsPapi quote strictly before kick-off (as production's
  history repair does). The Main that production actually showed is flagged (``stored_main``).

Universes
    site  every candidate, as the website chooses (prices only where they exist; ROI only over priced picks)
    bet   matches with a full real 1X2 and total-2.5 book; only selections with a real price are candidates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_1.main import candidates as CD
from experiments.v6_1.main import strategies as ST

from . import common as C

COLS = ["match_id", "league", "season", "date", "market", "selection", "p", "p_push", "status", "group", "line",
        "odds", "q_prod_shin"]
BAND = (0.40, 0.85)


def load(season: int) -> pd.DataFrame:
    if season == C.CURRENT:
        path = C.CACHE / "cand_live.parquet"
        if not path.exists():
            CD.live(refresh=True)
        d = pd.read_parquet(path, columns=COLS + ["source"])
    else:
        path = C.CACHE / f"cand_{season}.parquet"
        if not path.exists():
            CD.historical(season)
        d = pd.read_parquet(path, columns=COLS)
        d["source"] = "walk_forward"
    d = d.reset_index(drop=True)
    for c in ("league", "market", "selection", "group", "source"):
        d[c] = d[c].astype(str)
    if season == C.CURRENT:
        d = _flag_stored(d)
    d["mi"] = pd.factorize(d["match_id"])[0]
    return d


def _flag_stored(d: pd.DataFrame) -> pd.DataFrame:
    stored = CD.stored_live_mains()
    key = set(zip(stored["match_id"], stored["market"], stored["selection"]))
    d["stored_main"] = [k in key for k in zip(d["match_id"], d["market"], d["selection"])]
    played = set(stored.loc[stored["status"].isin(["WON", "LOST", "PUSH"]), "match_id"])
    return d.loc[d["match_id"].isin(played)].reset_index(drop=True)


def universe(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    """site: as is; bet: matches with a full 1X2 and total-2.5 book, priced candidates only (mi re-indexed)."""
    if kind == "site":
        out = d.copy()
    else:
        has = d["odds"].notna()
        k = d.assign(_h=has).loc[d["market"].isin(["1X2", "TOTAL_2.5"])].groupby("match_id")["_h"].agg(["sum", "size"])
        ok = set(k.index[(k["sum"] == 5) & (k["size"] == 5)])
        out = d.loc[d["match_id"].isin(ok) & has].copy()
    out["mi"] = pd.factorize(out["match_id"])[0]
    p = out["p"].to_numpy(float)
    cand = (~out["group"].isin(ST.EXCLUDED)).to_numpy() & (p >= BAND[0]) & (p <= BAND[1])
    out["ncand"] = pd.Series(cand.astype(int)).groupby(out["mi"].to_numpy()).transform("sum").to_numpy()
    return out.reset_index(drop=True)


def p_cond(df: pd.DataFrame, col: str = "p") -> np.ndarray:
    """Probability on the no-refund scale (Asian whole lines: P(win | no refund))."""
    return df[col].to_numpy(float) / np.clip(1 - df["p_push"].to_numpy(float), 1e-6, None)
