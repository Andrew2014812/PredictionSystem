"""Candidate tables: one row per (match, market, selection) with everything a Risk rule needs.

History (seasons 2022 ... 2025): the production chain on walk-forward predictions (base.block):
score_matrix -> reconcile_with_1x2 -> goal_markets(); corners: corners_distribution(mu, r) with the
NB size r from the previous season's block -> corner_markets(); real prices attached with
src.markets.odds.price from Football-Data rows only (Bet365 pre-match: 1X2, total 2.5 and the one
Asian line per match; never the closing columns). Main = production main_prediction().

Independent check (season 2026/27): the stored production history - probabilities recorded before
kick-off (local_data/predictions/markets.parquet) with their recorded / archived real prices, the
matches of unified_history(store.load_picks()) and their results.

Columns: season, match_id, date, league, market, selection, group, p, push_p, odds (NaN = no real
price), ev, status (WON / LOST / PUSH / None), eligible (production Risk candidate apart from the
p >= RISK_MIN_PROBABILITY threshold), is_main, p_mkt (de-vigged price of a complete two- or
three-way book, NaN otherwise).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_research import common as C
from src.analytics.performance import unified_history
from src.markets.distributions import corners_distribution, reconcile_with_1x2, score_matrix
from src.markets.markets import corner_markets, goal_markets, settle_selection
from src.markets.odds import PricedSelection, price
from src.markets.recommendation import (EXCLUDED_GROUPS, main_prediction, opposite_outcome, risk_prediction,
                                        same_outcome)
from src.modelling.training import negative_binomial_size
from src.prediction import store
from src.providers.football_data import odds_from_matches
from src.providers.schema import odds_lookup

from .base import CACHE, block

HISTORY = (2022, 2023, 2024, 2025)
INDEPENDENT = 2026
COMPLETE_BOOK = {"1X2": 3}        # other de-vigged markets: two-way (over/under, yes/no, H/A)
NO_BOOK = {"DOUBLE_CHANCE", "EXACT_SCORE", "EHANDICAP"}


def nb_size(season: int) -> float:
    prev = block(season - 1)[0] if season - 1 in HISTORY else block(season)[0]   # 2022: own season (calibration only)
    ok = prev["cmu"].notna() & prev["total_corners"].notna()
    return float(negative_binomial_size(prev.loc[ok, "total_corners"].to_numpy(float), prev.loc[ok, "cmu"].to_numpy()))


def _rows(match: dict, sels: list[PricedSelection], result: dict, extra: dict | None = None,
          main_key: tuple | None = None) -> list[dict]:
    """``main_key``: the (market, selection) of the Main actually shown (stored history); otherwise
    Main is recomputed with the production main_prediction()."""
    found = [s for s in sels if (s.market, s.selection) == main_key] if main_key else []
    if found:
        mp = found[0]
    else:
        main = main_prediction(sels)
        mp = main.pick if main else None
    rows = []
    for s in sels:
        same = mp is not None and (same_outcome(s, mp) or opposite_outcome(s, mp))
        rows.append({**match, "market": s.market, "selection": s.selection, "group": s.group,
                     "p": float(s.probability), "push_p": float(s.push_probability or 0.0),
                     "odds": float(s.odds) if s.has_odds else np.nan, "ev": float(s.ev) if s.has_odds else np.nan,
                     "status": settle_selection(s.market, s.selection, result["home_goals"], result["away_goals"],
                                                result.get("total_corners")),
                     "eligible": bool(s.group not in EXCLUDED_GROUPS and not same),
                     "is_main": bool(mp is not None and (s.market, s.selection) == (mp.market, mp.selection)),
                     **(extra or {})})
    return rows


def add_market_probability(df: pd.DataFrame) -> pd.DataFrame:
    """De-vigged price p_mkt = (1/odds) / sum(1/odds) over a complete book of the same market."""
    df = df.copy()
    inv = np.where(df["odds"] > 1, 1.0 / df["odds"], np.nan)
    df["_inv"] = inv
    g = df.groupby(["match_id", "market"])["_inv"]
    n_priced, total = g.transform("count"), g.transform("sum")
    need = df["group"].map(lambda x: COMPLETE_BOOK.get(x, 2))
    ok = (n_priced == need) & ~df["group"].isin(NO_BOOK) & df["_inv"].notna()
    df["p_mkt"] = np.where(ok, df["_inv"] / total, np.nan)
    df["overround"] = np.where(ok, total - 1.0, np.nan)
    return df.drop(columns="_inv")


def history_season(season: int, refresh: bool = False) -> pd.DataFrame:
    path = CACHE / f"cand_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    t0 = time.time()
    rows_block, _ = block(season)
    r = nb_size(season)
    feats = C.seasons(C.features(), [season])
    lookup = odds_lookup(odds_from_matches(feats))
    out = []
    for row in rows_block.itertuples():
        m = reconcile_with_1x2(score_matrix(row.lh, row.la), np.array([row.pH, row.pD, row.pA]))
        sels = goal_markets(m)
        if row.cmu == row.cmu:
            sels += corner_markets(corners_distribution(row.cmu, r))
        sels = price(sels, lookup.get(row.match_id))
        match = {"season": season, "match_id": row.match_id, "date": row.date, "league": row.league}
        result = {"home_goals": row.home_goals, "away_goals": row.away_goals, "total_corners": row.total_corners}
        out += _rows(match, sels, result)
    df = add_market_probability(pd.DataFrame(out))
    df.attrs["seconds"] = time.time() - t0
    df.to_parquet(path)
    return df


def independent_season(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Season 2026/27 from the stored production history (pre-kick-off probabilities, real prices)."""
    path, info_path = CACHE / "cand_2026.parquet", CACHE / "cand_2026.json"
    if path.exists() and info_path.exists() and not refresh:
        import json
        return pd.read_parquet(path), json.loads(info_path.read_text())
    picks = unified_history(store.load_picks())
    picks = picks.loc[picks["season"] == INDEPENDENT]
    markets = store.load_markets()
    markets = markets.loc[markets["match_id"].isin(set(picks["match_id"]))]
    per_match = picks.groupby("match_id").first()
    stored = {(r.match_id, r.role): (r.market, r.selection) for r in picks.itertuples() if r.role in ("main", "risk")}
    out, agree_main, agree_risk, agree_risk_stored_main, n = [], 0, 0, 0, 0
    for mid, g in markets.groupby("match_id", sort=False):
        info = per_match.loc[mid]
        sels = [PricedSelection(r.market, r.selection, float(r.probability),
                                float(r.odds) if r.odds == r.odds and r.odds is not None else None,
                                r.bookmaker, r.provider,
                                float(r.push_probability) if r.push_probability == r.push_probability else 0.0)
                for r in g.itertuples()]
        main = main_prediction(sels)
        risk = risk_prediction(sels, main.pick if main else None)
        shown = [s for s in sels if (s.market, s.selection) == stored.get((mid, "main"))]
        risk_s = risk_prediction(sels, shown[0] if shown else None)
        n += 1
        agree_main += main is not None and stored.get((mid, "main")) == (main.pick.market, main.pick.selection)
        agree_risk += risk is not None and stored.get((mid, "risk")) == (risk.pick.market, risk.pick.selection)
        agree_risk_stored_main += risk_s is not None and stored.get((mid, "risk")) == (risk_s.pick.market,
                                                                                        risk_s.pick.selection)
        match = {"season": INDEPENDENT, "match_id": mid, "date": info["date"], "league": info["league"]}
        result = {"home_goals": info["home_goals"], "away_goals": info["away_goals"],
                  "total_corners": info["total_corners"]}
        out += _rows(match, sels, result, {"source": info["source"]}, main_key=stored.get((mid, "main")))
    df = add_market_probability(pd.DataFrame(out))
    keys = pd.Series([stored.get((m, "risk")) for m in df["match_id"]], index=df.index)
    df["stored_risk"] = [k is not None and (mk, sl) == k for k, mk, sl in zip(keys, df["market"], df["selection"])]
    meta = {"matches": n, "main_reproduced": int(agree_main), "risk_reproduced": int(agree_risk),
            "risk_reproduced_with_stored_main": int(agree_risk_stored_main),
            "stored_main": int(sum(1 for k in stored if k[1] == "main")),
            "stored_risk": int(sum(1 for k in stored if k[1] == "risk"))}
    df.to_parquet(path)
    import json
    info_path.write_text(json.dumps(meta))
    return df, meta


def history(refresh: bool = False) -> pd.DataFrame:
    return pd.concat([history_season(s, refresh) for s in HISTORY], ignore_index=True)


if __name__ == "__main__":
    C.setup_logging("risk.data")
    for s in HISTORY:
        t = time.time()
        d = history_season(s, refresh=True)
        print(s, len(d), d["match_id"].nunique(), f"{time.time() - t:.0f}s", flush=True)
    t = time.time()
    d, meta = independent_season(refresh=True)
    print(2026, len(d), meta, f"{time.time() - t:.0f}s", flush=True)
