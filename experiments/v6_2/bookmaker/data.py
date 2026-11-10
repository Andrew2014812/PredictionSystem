"""Benchmark tables: model probability, bookmaker prices (two snapshots) and the result, one row per match.

Model probabilities
    2022/23–2025/26  walk-forward production models (v6.1 cache experiments/v6_1/main/cache: outer_<s> 1X2 and
                     cand_<s> derived markets — models fitted on seasons < s with the production parameters);
    2026/27          the stored production predictions (unified history: live when recorded, else backfill;
                     markets.parquet of the stored version).
Prices
    Football-Data opening ("open", Friday / Tuesday snapshot, available before the decision) and closing ("close")
    snapshots of Bet365, Pinnacle (to 2025/26), the market average and the Betfair exchange (from 2024/25),
    experiments/v6_2/odds_timing/fd.py. Markets: 1X2, total 2.5 and the one Asian line Football-Data quotes
    (half lines only, so no refund).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_2.odds_timing import fd

from . import common as C

BOOKS = ("b365", "ps", "avg", "bfe")


def _v61_cache():
    from experiments.v6_1.main import common as V61
    return V61.CACHE


def _model_1x2_ou(season: int) -> pd.DataFrame:
    if season == C.LIVE:
        from experiments.v6_2.odds_timing.drift import live_model_probs
        return live_model_probs()
    c = pd.read_parquet(_v61_cache() / f"cand_{season}.parquet", columns=["match_id", "market", "selection", "p"])
    c = c.loc[c["market"].isin(["1X2", "TOTAL_2.5"])]
    w = c.pivot_table(index="match_id", columns="selection", values="p", aggfunc="first")
    return pd.DataFrame({"match_id": w.index, "p_H": w["H"].to_numpy(), "p_D": w["D"].to_numpy(),
                         "p_A": w["A"].to_numpy(), "p_OVER": w["OVER"].to_numpy()})


def table_1x2(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"t1x2_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    d = fd.prices(season)
    m = _model_1x2_ou(season)
    t = d.merge(m, on="match_id", how="inner")
    out = pd.DataFrame({"match_id": t["match_id"], "league": t["league"], "season": season,
                        "date": pd.to_datetime(t["date"]),
                        "y": np.where(t["hg"] > t["ag"], 0, np.where(t["hg"] == t["ag"], 1, 2)),
                        "p_H": t["p_H"], "p_D": t["p_D"], "p_A": t["p_A"]})
    if "source" in t:
        out["source"] = t["source"].to_numpy()
    for b in BOOKS:
        for suf in ("", "_c"):
            for s in "HDA":
                col = f"x_{s}_{b}{suf}"
                out[f"o_{s}_{b}{'_c' if suf else ''}"] = t[col].to_numpy(float) if col in t else np.nan
    out = out.loc[out[["p_H", "p_D", "p_A"]].notna().all(1)].reset_index(drop=True)
    out.to_parquet(path)
    return out


def table_ou(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"tou_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    d = fd.prices(season)
    m = _model_1x2_ou(season)
    t = d.merge(m, on="match_id", how="inner")
    out = pd.DataFrame({"match_id": t["match_id"], "league": t["league"], "season": season,
                        "date": pd.to_datetime(t["date"]), "y": np.where(t["hg"] + t["ag"] > 2.5, 0, 1),
                        "p_OVER": t["p_OVER"]})
    for b in BOOKS:
        for suf in ("", "_c"):
            for s, k in (("OVER", "O"), ("UNDER", "U")):
                col = f"ou_{s}_{b}{suf}"
                out[f"o_{k}_{b}{suf}"] = t[col].to_numpy(float) if col in t else np.nan
    out = out.loc[out["p_OVER"].notna()].reset_index(drop=True)
    out.to_parquet(path)
    return out


def table_ah(season: int, refresh: bool = False) -> pd.DataFrame:
    """The Football-Data Asian line of each match when it is a half line (no refund) and did not move."""
    path = C.CACHE / f"tah_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from src.markets.markets import handicap_market
    d = fd.prices(season)
    line, line_c = d["ah_line"].to_numpy(float), d["ah_line_c"].to_numpy(float)
    half = np.isfinite(line) & (np.abs((line * 2) % 2 - 1) < 1e-9) & np.isclose(line, line_c)
    d = d.loc[half].reset_index(drop=True)
    d["market"] = [handicap_market(x) for x in d["ah_line"]]
    if season == C.LIVE:
        from src.analytics.performance import unified_history
        hist = unified_history(pd.read_parquet(C.ROOT / "local_data/predictions/picks.parquet"))
        ids = set(hist.loc[hist["season"] == C.LIVE, "match_id"])
        mk = pd.read_parquet(C.ROOT / "local_data/predictions/markets.parquet",
                             columns=["match_id", "market", "selection", "probability"])
        mk = mk.loc[mk["match_id"].isin(ids) & mk["market"].str.startswith("HANDICAP_") & (mk["selection"] == "H")]
        mk = mk.drop_duplicates(["match_id", "market"], keep="last").rename(columns={"probability": "p"})
    else:
        mk = pd.read_parquet(_v61_cache() / f"cand_{season}.parquet", columns=["match_id", "market", "selection", "p"])
        mk = mk.loc[mk["market"].str.startswith("HANDICAP_") & (mk["selection"] == "H")]
    t = d.merge(mk[["match_id", "market", "p"]], on=["match_id", "market"], how="inner")
    margin = t["hg"].to_numpy() + t["ah_line"].to_numpy() - t["ag"].to_numpy()
    out = pd.DataFrame({"match_id": t["match_id"], "league": t["league"], "season": season,
                        "date": pd.to_datetime(t["date"]), "line": t["ah_line"], "y": np.where(margin > 0, 0, 1),
                        "p_HC": t["p"].astype(float)})
    for b in BOOKS:
        for suf in ("", "_c"):
            for s, k in (("H", "H"), ("A", "A")):
                col = f"ah_{s}_{b}{suf}"
                out[f"o_{k}_{b}{suf}"] = t[col].to_numpy(float) if col in t else np.nan
    out.to_parquet(path)
    return out


MARKETS = {"1X2": (table_1x2, ("H", "D", "A"), ("p_H", "p_D", "p_A")),
           "OU25": (table_ou, ("O", "U"), ("p_OVER",)),
           "AH": (table_ah, ("H", "A"), ("p_HC",))}


def model_matrix(t: pd.DataFrame, market: str) -> np.ndarray:
    if market == "1X2":
        p = t[["p_H", "p_D", "p_A"]].to_numpy(float)
        return p / p.sum(1, keepdims=True)
    col = MARKETS[market][2][0]
    p = np.clip(t[col].to_numpy(float), 1e-6, 1 - 1e-6)
    return np.column_stack([p, 1 - p])


def odds_matrix(t: pd.DataFrame, market: str, book: str, closing: bool = False) -> np.ndarray:
    suf = "_c" if closing else ""
    return t[[f"o_{s}_{book}{suf}" for s in MARKETS[market][1]]].to_numpy(float)


def load(market: str, seasons) -> pd.DataFrame:
    fn = MARKETS[market][0]
    return pd.concat([fn(s) for s in seasons], ignore_index=True)
