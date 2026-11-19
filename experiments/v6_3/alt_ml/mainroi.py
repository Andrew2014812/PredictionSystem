"""Main ROI of any model outputs through production's engine, in both decision universes, paired with Baseline B.

Shared by the two studies of this folder pair (alt_ml, quality). Nothing here re-implements production: the
outputs go through ``engine.batch_from_outputs`` (production market mathematics and production Main rule), the
candidates are settled by ``evaluate.candidates`` and compared with ``evaluate.compare``.

Universes (experiments/v6_3/README.md): ``close`` (primary; last quote before kick-off) and ``t09`` (prices known
at T = min(09:00 UK, kick-off − 30 min)). Earlier seasons (2024/25, 2025/26) use the Football-Data Bet365 prices of
the season files (1X2, total 2.5, one Asian line), as production attaches them.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as RD
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV

UNIVERSES = {"close": "closing", "t09": "decision"}
CACHE = C.SHARED_CACHE


@lru_cache(maxsize=None)
def matches() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "matches.parquet")


@lru_cache(maxsize=None)
def results() -> pd.DataFrame:
    return RD.results(matches())


@lru_cache(maxsize=None)
def lookup(universe: str) -> dict:
    return RD.lookup(pd.read_parquet(CACHE / f"prices_{UNIVERSES[universe]}.parquet"))


@lru_cache(maxsize=None)
def prices(universe: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"prices_{UNIVERSES[universe]}.parquet")


@lru_cache(maxsize=None)
def baseline_b(universe: str) -> pd.DataFrame:
    b = pd.read_parquet(CACHE / f"baseline_b_{universe}.parquet")
    return b.loc[b["match_id"].isin(results().index)].reset_index(drop=True)


@lru_cache(maxsize=None)
def baseline_a() -> pd.DataFrame:
    a = pd.read_parquet(CACHE / "baseline_a.parquet")
    return a


@lru_cache(maxsize=None)
def candidates_b(universe: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"candidates_{universe}.parquet")


def market_probs(universe: str) -> pd.DataFrame:
    """De-margined (proportional) 1X2 probabilities of the universe's price per match: q_home, q_draw, q_away."""
    p = prices(universe)
    p = p.loc[p["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="odds", aggfunc="first")
    p = p.reindex(columns=["H", "D", "A"]).dropna()
    inv = 1 / p
    q = inv.div(inv.sum(axis=1), axis=0)
    q.columns = ["q_home", "q_draw", "q_away"]
    return q


def run_main(outputs: pd.DataFrame, universe: str, label: str, extras: dict | None = None,
             keep_candidates: bool = False):
    """One Main per played 2026/27 match for ``outputs`` (match_id + engine.OUTPUT_COLS) in ``universe``."""
    m = matches()
    batch = EN.batch_from_outputs(m, outputs, lookup(universe), label, extras)
    cands = EV.candidates(batch.markets, m, results())
    main = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
    main = cands.merge(main, on=["match_id", "market", "selection"])
    main = main.loc[main["match_id"].isin(results().index)].reset_index(drop=True)
    main["method"] = label
    return (main, cands) if keep_candidates else main


def hist_main(features_season: pd.DataFrame, outputs: pd.DataFrame, label: str, keep_candidates: bool = False):
    """Main of an earlier season with the Football-Data Bet365 prices of the season files."""
    from src.providers.football_data import odds_from_matches
    m = features_season.copy()
    fd = odds_from_matches(m)
    fd["collected_at"] = pd.to_datetime(fd["collected_at"])
    lk = RD.lookup(RD._best(fd))
    batch = EN.batch_from_outputs(m, outputs, lk, label)
    res = RD.results(m)
    cands = EV.candidates(batch.markets, m, res)
    main = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
    main = cands.merge(main, on=["match_id", "market", "selection"])
    main = main.loc[main["match_id"].isin(res.index)].reset_index(drop=True)
    main["method"] = label
    return (main, cands) if keep_candidates else main


def evaluate(main: pd.DataFrame, universe: str) -> dict:
    """Summary, ROI CI, paired comparison with Baseline B of the universe and with Baseline A."""
    return {"summary": EV.summary(main), "roi_ci": EV.roi_with_ci(main),
            "vs_b": EV.compare(main, baseline_b(universe)),
            "vs_a": EV.compare(main, baseline_a().loc[baseline_a()["match_id"].isin(results().index)])}


def top_win_dependence(main: pd.DataFrame, k: int = 5) -> dict:
    bets = main.loc[main["bet"] == 1]
    if bets.empty:
        return {}
    p = bets["profit"].sort_values(ascending=False)
    return {"roi": float(p.sum() / len(p)), f"roi_without_top{k}": float(p.iloc[k:].sum() / len(p)),
            "top_win": float(p.iloc[0]), "share_profit_top5": float(p.iloc[:k].sum())}


def odds_sensitivity(main: pd.DataFrame, factors=(0.98, 0.95)) -> dict:
    """ROI if every price had been ``factor`` × the stored one (worse prices)."""
    bets = main.loc[main["bet"] == 1]
    out = {}
    for f in factors:
        o = bets["odds"] * f
        prof = np.where(bets["status"] == "WON", o - 1, -1.0)
        out[str(f)] = float(prof.sum() / len(bets)) if len(bets) else None
    return out


def monthly_diff(main: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = main.assign(month=pd.to_datetime(main["date"]).dt.to_period("M").astype(str))
    for mo, g in a.groupby("month"):
        b = base.loc[base["match_id"].isin(g["match_id"])]
        s, sb = EV.summary(g), EV.summary(b)
        rows.append({"month": mo, "matches": s["matches"], "bets": s["bets"], "roi": s["roi"], "roi_b": sb["roi"],
                     "hit": s["hit_rate"], "hit_b": sb["hit_rate"], "small": s["bets"] < 100})
    return pd.DataFrame(rows)


def league_diff(main: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, g in main.groupby("league"):
        b = base.loc[base["match_id"].isin(g["match_id"])]
        s, sb = EV.summary(g), EV.summary(b)
        rows.append({"league": lg, "bets": s["bets"], "roi": s["roi"], "roi_b": sb["roi"],
                     "d_roi": (s["roi"] - sb["roi"]) if s["roi"] is not None and sb["roi"] is not None else None,
                     "small": s["bets"] < 100})
    return pd.DataFrame(rows)


def market_calibration(cands: pd.DataFrame) -> pd.DataFrame:
    """Per market group: Brier score and expected calibration error (10 bins) of every settled WON / LOST
    candidate (probability on the no-refund scale), plus the per-match mean Brier (for paired tests)."""
    c = cands.loc[cands["status"].isin(["WON", "LOST"])].copy()
    c["y"] = (c["status"] == "WON").astype(float)
    c["pc"] = c["probability"] / np.clip(1 - c["push_probability"].fillna(0), 1e-6, None)
    c["pc"] = c["pc"].clip(0, 1)
    c["brier"] = (c["pc"] - c["y"]) ** 2
    return c[["match_id", "group", "market", "selection", "pc", "y", "brier"]]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def calibration_compare(cal_new: pd.DataFrame, cal_base: pd.DataFrame) -> dict:
    out = {}
    for g in sorted(set(cal_base["group"])):
        a = cal_new.loc[cal_new["group"] == g]
        b = cal_base.loc[cal_base["group"] == g]
        ma = a.groupby("match_id")["brier"].mean()
        mb = b.groupby("match_id")["brier"].mean()
        common = ma.index.intersection(mb.index)
        d = C.paired_mean_diff(ma.loc[common].to_numpy(), mb.loc[common].to_numpy()) if len(common) else {}
        out[g] = {"n": int(len(a)), "brier": float(a["brier"].mean()) if len(a) else None,
                  "brier_base": float(b["brier"].mean()) if len(b) else None,
                  "ece": ece(a["pc"].to_numpy(), a["y"].to_numpy()) if len(a) else None,
                  "ece_base": ece(b["pc"].to_numpy(), b["y"].to_numpy()) if len(b) else None,
                  "d_brier_paired": d}
    return out
