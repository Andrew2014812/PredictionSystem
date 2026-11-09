"""Risk candidate tables (one row per match x selection).

History 2022/23 ... 2025/26: the v6.1 Risk tables (experiments/v6_1/risk/data.py) built with the production
functions on the v6.1 walk-forward predictions (identical to the Main study's, results/cache_check.json of
main_calibration): goal_markets / corner_markets, Football-Data Bet365 pre-match prices (1X2, total 2.5, the one
Asian line), production main_prediction() and the production Risk pool (``eligible``).

2026/27: the production store — the latest pre-kick-off probabilities of every match of the unified history
(markets.parquet, all groups incl. exact score), recorded prices, missing ones filled with the last OddsPapi quote
strictly before kick-off (as production's history repair); Main = the Main production showed; ``eligible`` =
the production Risk pool against that Main; ``stored_risk`` = the Risk production showed.

Columns: season, match_id, date, league, market, selection, group, p, push_p, odds, ev, status (WON / LOST / PUSH /
None), eligible, is_main, p_mkt (proportional de-vig of a complete book; whole Asian lines: P(win | no refund)),
overround, q (de-vigged probability of a WIN = p_mkt (1 - push_p)).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

HISTORY = (2022, 2023, 2024, 2025)
CURRENT = 2026
COMPLETE_BOOK = {"1X2": 3}
NO_BOOK = {"DOUBLE_CHANCE", "EXACT_SCORE", "EHANDICAP"}
COLS = ["season", "match_id", "date", "league", "market", "selection", "group", "p", "push_p", "odds", "ev",
        "status", "eligible", "is_main"]


def add_market_probability(df: pd.DataFrame) -> pd.DataFrame:
    inv = np.where(df["odds"] > 1, 1.0 / df["odds"], np.nan)
    df = df.assign(_inv=inv)
    g = df.groupby(["match_id", "market"])["_inv"]
    n_priced, total = g.transform("count"), g.transform("sum")
    need = df["group"].map(lambda x: COMPLETE_BOOK.get(x, 2))
    ok = (n_priced == need) & ~df["group"].isin(NO_BOOK) & df["_inv"].notna()
    df["p_mkt"] = np.where(ok, df["_inv"] / total, np.nan)
    df["overround"] = np.where(ok, total - 1.0, np.nan)
    df["q"] = df["p_mkt"] * (1 - df["push_p"].fillna(0.0))
    return df.drop(columns="_inv")


def history_season(season: int) -> pd.DataFrame:
    path = CACHE / f"cand_{season}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing: copy it from experiments/v6_1/risk/cache or run "
                         "python -m experiments.v6_1.risk.data")
    d = pd.read_parquet(path, columns=COLS)
    return add_market_probability(d)


def _footprints(keys) -> dict:
    from src.markets.recommendation import _footprint
    return {k: _footprint(*k) for k in keys}


def current_season(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    path, info_path = CACHE / "cand_2026.parquet", CACHE / "cand_2026.json"
    if path.exists() and info_path.exists() and not refresh:
        return pd.read_parquet(path), json.loads(info_path.read_text())
    from src.analytics.performance import unified_history
    from src.features.store import load_features
    from src.markets.markets import settle_selection
    from src.prediction import store
    from src.prediction.history_repair import archive_prices
    t0 = time.time()
    picks = unified_history(store.load_picks())
    picks = picks.loc[picks["season"] == CURRENT]
    stored = {(r.match_id, r.role): (r.market, r.selection) for r in picks.itertuples() if r.role in ("main", "risk")}
    ids = set(picks["match_id"])
    mk = store.load_markets()
    mk = mk.loc[mk["match_id"].isin(ids)].drop_duplicates(["match_id", "market", "selection"], keep="last").copy()
    feats = load_features()
    feats = feats.loc[feats["match_id"].isin(ids)]
    need = mk.loc[mk["odds"].isna(), ["match_id", "market", "selection"]].assign(source="live", odds=np.nan)
    arch = archive_prices(need, feats)
    mk["odds_origin"] = np.where(mk["odds"].notna(), "recorded", None)
    if not arch.empty:
        a = arch.set_index(["match_id", "market", "selection"])
        key = pd.MultiIndex.from_frame(mk[["match_id", "market", "selection"]])
        pos = a.index.get_indexer(key)
        fill = (pos >= 0) & mk["odds"].isna().to_numpy()
        mk.loc[fill, "odds"] = a["odds"].to_numpy()[pos[fill]]
        mk.loc[fill, "odds_origin"] = "archive"
    per = picks.groupby("match_id").first()
    meta = feats.set_index("match_id")
    out = pd.DataFrame({"season": CURRENT, "match_id": mk["match_id"].to_numpy(), "market": mk["market"].to_numpy(),
                        "selection": mk["selection"].to_numpy(), "group": mk["group"].to_numpy(),
                        "p": mk["probability"].astype(float).to_numpy(),
                        "push_p": mk["push_probability"].fillna(0).astype(float).to_numpy(),
                        "odds": mk["odds"].astype(float).to_numpy(), "odds_origin": mk["odds_origin"].to_numpy()})
    out["date"] = pd.to_datetime(out["match_id"].map(per["date"]))
    out["league"] = out["match_id"].map(per["league"])
    out["source"] = out["match_id"].map(per["source"])
    out["ev"] = np.where(out["odds"] > 1, out["p"] * out["odds"] - (1 - out["push_p"]), np.nan)
    hg = out["match_id"].map(meta["home_goals"]).to_numpy(float)
    ag = out["match_id"].map(meta["away_goals"]).to_numpy(float)
    tc = out["match_id"].map(meta["total_corners"]).to_numpy(float)
    out["status"] = [settle_selection(m, s, h, a, c) for m, s, h, a, c in
                     zip(out["market"], out["selection"], hg, ag, tc)]
    # production Risk pool against the Main production showed
    main_key = out["match_id"].map(lambda m: stored.get((m, "main")))
    risk_key = out["match_id"].map(lambda m: stored.get((m, "risk")))
    keys = set(zip(out["market"], out["selection"])) | {k for k in main_key.dropna()}
    fp = _footprints({k for k in keys if not k[0].startswith("CORNERS")})
    flip = {"WON": "LOST", "LOST": "WON", "PUSH": "PUSH"}
    elig, is_main, is_risk = [], [], []
    for mkt, sel, grp, mkey, rkey in zip(out["market"], out["selection"], out["group"], main_key, risk_key):
        im = mkey is not None and (mkt, sel) == tuple(mkey)
        same = opp = False
        if mkey is not None and not im:
            if grp == "CORNERS" or mkey[0].startswith("CORNERS"):
                opp = mkt == mkey[0] and sel != mkey[1]
            else:
                a, b = fp[(mkt, sel)], fp[tuple(mkey)]
                same = a == b
                opp = a == tuple(flip.get(x, x) for x in b)
        elig.append(grp != "EHANDICAP" and not im and not same and not opp)
        is_main.append(im)
        is_risk.append(rkey is not None and (mkt, sel) == tuple(rkey))
    out["eligible"], out["is_main"], out["stored_risk"] = elig, is_main, is_risk
    out = add_market_probability(out)
    played = out.groupby("match_id")["status"].transform(lambda s: s.notna().any())
    out = out.loc[played].reset_index(drop=True)
    info = {"matches": int(out["match_id"].nunique()), "rows": int(len(out)),
            "archive_filled": int((out["odds_origin"] == "archive").sum()),
            "stored_risk_rows": int(out["stored_risk"].sum()), "seconds": round(time.time() - t0, 1)}
    out.to_parquet(path)
    info_path.write_text(json.dumps(info))
    return out, info


def load(season: int) -> pd.DataFrame:
    if season == CURRENT:
        return current_season()[0]
    return history_season(season)


if __name__ == "__main__":
    d, info = current_season(refresh=True)
    print(info)
