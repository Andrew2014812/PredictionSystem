"""Inputs shared by the Market-Models and Goal-Distribution studies.

* ``base(season)`` — walk-forward predictions of the four production models for one season (fitted on all
  seasons before it, tuned parameters of the active production version): pH/pD/pA, lh/la (goal means), cmu
  (corner mean) and the corner NB size, from experiments/v6_1/main/base.py (cache shared with that study).
  2022/23–2025/26 come from the v6.1 cache; 2026/27 is fitted the same way (seasons 2021/22–2025/26 ->
  played 2026/27 matches), i.e. the same training data as the production model prod_20261012.
* ``prices(season)`` — real prices per (match, market, selection): Football-Data Bet365 opening (the production
  price of past seasons: 1X2, total 2.5, the one Asian line of the match) and closing; for 2026/27 the stored
  production prices (recorded at generation, missing ones filled with the last OddsPapi quote before kick-off,
  experiments/v6_2/risk/data.py) plus the last pre-kick-off OddsPapi quote for selections production did not
  store (e.g. other exact scores).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.markets.markets import handicap_market, is_half_line

from . import common as C

V61_CACHE = C.ROOT / "experiments" / "v6_1" / "main" / "cache"


def base(season: int) -> tuple[pd.DataFrame, dict]:
    """One row per played match of ``season``: production walk-forward predictions + results."""
    from experiments.v6_1.main import base as B
    from experiments.v6_1.main import common as BC
    BC.N_JOBS = C.N_JOBS
    pred, info = B.outer(season)
    pred = pred.loc[pred["home_goals"].notna() & pred["away_goals"].notna()].reset_index(drop=True)
    pred["home_goals"] = pred["home_goals"].astype(int)
    pred["away_goals"] = pred["away_goals"].astype(int)
    return pred, info


def check_base_against_features(seasons=C.FOLDS) -> dict:
    """The cached predictions belong to the current feature table (same matches, same results)."""
    from src.features.store import load_features
    f = load_features()[["match_id", "season", "played", "home_goals", "away_goals", "total_corners"]]
    out = {}
    for s in seasons:
        p, _ = base(s)
        fs = f.loc[(f["season"] == s) & f["played"].astype(bool) & f["home_goals"].notna()].set_index("match_id")
        common = p["match_id"].isin(fs.index)
        j = fs.reindex(p["match_id"])
        same = ((j["home_goals"].to_numpy() == p["home_goals"].to_numpy())
                & (j["away_goals"].to_numpy() == p["away_goals"].to_numpy()))
        out[str(s)] = {"pred_rows": int(len(p)), "feature_rows": int(len(fs)), "in_features": int(common.sum()),
                       "same_goals": int(same.sum()),
                       "missing_from_pred": int((~fs.index.isin(p["match_id"])).sum())}
    return out


# ------------------------------------------------------------------------------------------------ prices
def _fd_long(season: int) -> pd.DataFrame:
    """Football-Data Bet365 opening / closing prices, long (match_id, market, selection, odds, odds_c)."""
    from experiments.v6_1.main import fd_prices
    fd = fd_prices.prices(season)
    rows = []
    for closing in (False, True):
        suf = "_c" if closing else ""
        col = "odds_c" if closing else "odds"
        parts = []
        for s in "HDA":
            parts.append(pd.DataFrame({"match_id": fd["match_id"], "market": "1X2", "selection": s,
                                       col: fd[f"x_{s}_b365{suf}"]}))
        for s in ("OVER", "UNDER"):
            parts.append(pd.DataFrame({"match_id": fd["match_id"], "market": "TOTAL_2.5", "selection": s,
                                       col: fd[f"ou_{s}_b365{suf}"]}))
        line = fd[f"ah_line{suf}"].astype(float)
        ok = line.notna() & line.map(lambda x: x == x and (is_half_line(x) or x % 1 == 0))
        mk = line.where(ok).map(lambda x: handicap_market(x) if x == x else None)
        for s in "HA":
            parts.append(pd.DataFrame({"match_id": fd["match_id"], "market": mk, "selection": s,
                                       col: fd[f"ah_{s}_b365{suf}"]}))
        d = pd.concat(parts, ignore_index=True)
        d = d.loc[d["market"].notna() & d[col].notna() & (d[col] > 1)]
        rows.append(d.drop_duplicates(["match_id", "market", "selection"]).set_index(["match_id", "market",
                                                                                      "selection"]))
    out = rows[0].join(rows[1], how="outer").reset_index()
    return out


def _current_long() -> pd.DataFrame:
    """2026/27 prices: stored production prices (+ archive fill) and the OddsPapi archive for other keys."""
    path = C.CACHE / "cand_2026.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing: copy experiments/v6_2/risk/cache/cand_2026.parquet "
                         "(python -m experiments.v6_2.risk.data)")
    st = pd.read_parquet(path, columns=["match_id", "market", "selection", "odds", "odds_origin", "source"])
    st = st.loc[st["odds"].notna() & (st["odds"] > 1)].rename(columns={"source": "hist_source"})
    st["origin"] = st["odds_origin"].fillna("recorded")
    st = st.drop(columns="odds_origin")
    arch = _archive_long(set(st["match_id"]))
    if not arch.empty:
        key = pd.MultiIndex.from_frame(arch[["match_id", "market", "selection"]])
        have = pd.MultiIndex.from_frame(st[["match_id", "market", "selection"]])
        arch = arch.loc[~key.isin(have)].assign(origin="archive_extra")
        st = pd.concat([st, arch], ignore_index=True)
    return st


def _archive_long(match_ids: set) -> pd.DataFrame:
    """Last OddsPapi quote strictly before kick-off per (match, market, selection) (production's repair rule)."""
    from src.features.store import load_features
    from src.prediction.history_repair import archive_prices
    feats = load_features()
    feats = feats.loc[feats["match_id"].isin(match_ids)]
    keys = pd.read_parquet(C.ROOT / "local_data" / "odds" / "odds.parquet", columns=["match_id", "market",
                                                                                    "selection"])
    keys = keys.loc[keys["match_id"].isin(match_ids)].drop_duplicates()
    if keys.empty:
        return pd.DataFrame(columns=["match_id", "market", "selection", "odds"])
    arch = archive_prices(keys.assign(source="live", odds=np.nan), feats)
    if arch.empty:
        return pd.DataFrame(columns=["match_id", "market", "selection", "odds"])
    return arch[["match_id", "market", "selection", "odds"]]


def prices(season: int, refresh: bool = False) -> pd.DataFrame:
    """Columns match_id, market, selection, odds (decision price), odds_c (closing where known), origin."""
    path = C.CACHE / f"prices_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if season == C.CURRENT:
        d = _current_long()
        d["odds_c"] = np.nan
    else:
        d = _fd_long(season)
        d["origin"] = "fd_b365_open"
    d.to_parquet(path)
    return d


def reproducibility() -> dict:
    """Max |difference| between the refitted folds and the v6.1 cache (when that cache is present)."""
    out = {}
    for s in (2022, 2023, 2024, 2025):
        f = V61_CACHE / f"outer_{s}.parquet"
        if not f.exists():
            out[str(s)] = None
            continue
        a, (b, _) = pd.read_parquet(f), base(s)
        a = a.loc[a["home_goals"].notna()].reset_index(drop=True)
        same = bool((a["match_id"].to_numpy() == b["match_id"].to_numpy()).all())
        out[str(s)] = {"same_matches": same,
                       "max_abs_diff": float(max(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
                                                 for c in ("pH", "pD", "pA", "lh", "la", "cmu")))}
    return out


def run_check() -> dict:
    out = {"base": check_base_against_features(), "reproducibility_vs_v61_cache": reproducibility()}
    out["base_info"] = {}
    for s in C.FOLDS:
        _, info = base(s)
        out["base_info"][str(s)] = info
    return out


if __name__ == "__main__":
    C.setup_logging()
    res = run_check()
    print(json.dumps(res["base"], indent=1))
