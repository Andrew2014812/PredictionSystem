"""Football-Data pre-match prices of 2021/22–2026/27 in two snapshots (cached per season).

Football-Data publishes an "opening" snapshot (columns B365H, PSH, AvgH, BFEH, ... — collected on Friday afternoon
for weekend matches and on Tuesday for midweek matches, i.e. roughly 1–4 days before kick-off) and a closing
snapshot (B365CH, PSCH, AvgCH, BFECH, ... — the last prices before kick-off). Both are real pre-match prices.
Books used: Bet365 (the production book), Pinnacle (2021/22–2025/26), the Football-Data market average and the
Betfair exchange (BFE, from 2024/25; an exchange price, used only as a reference probability, never as a bet).

The loader is the v6.1 one (experiments/v6_1/main/fd_prices.py: the match id is built exactly as in
src/data/cleaning.py) with the exchange columns added.
"""
from __future__ import annotations

import pandas as pd

from . import common as C

SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
BOOKS = ("b365", "ps", "avg", "bfe", "max")


def prices(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"fd_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from experiments.v6_1.main import fd_prices as F
    F.BOOKS_1X2 = {**F.BOOKS_1X2, "bfe": "BFE"}
    F.BOOKS_OU = {**F.BOOKS_OU, "bfe": "BFE"}
    F.BOOKS_AH = {**F.BOOKS_AH, "bfe": "BFE"}
    df = F.prices(season)
    raw = F.load_raw(season)
    keep = [c for c in df.columns if c == "match_id" or any(c.endswith(f"_{b}") or c.endswith(f"_{b}_c")
                                                            for b in BOOKS) or c.startswith("ah_line")]
    df = df[keep].copy()
    df["season"] = season
    # league and result from the raw file (same row order as F.prices before de-duplication)
    from src.data.cleaning import _parse_dates, make_match_id
    dates = _parse_dates(raw["Date"].astype(str).str.strip())
    ids = [make_match_id(str(lg).strip(), d, str(h).strip(), str(a).strip()) if d == d else None
           for lg, d, h, a in zip(raw["Div"], dates, raw["HomeTeam"], raw["AwayTeam"])]
    res = pd.DataFrame({"match_id": ids, "league": raw["Div"].astype(str).str.strip(), "date": dates,
                        "hg": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
                        "ag": pd.to_numeric(raw.get("FTAG"), errors="coerce")})
    res = res.dropna(subset=["match_id"]).drop_duplicates("match_id", keep="last")
    df = df.merge(res, on="match_id", how="left")
    df = df.loc[df["hg"].notna() & df["ag"].notna()].reset_index(drop=True)
    df.to_parquet(path)
    return df


def all_seasons(seasons=SEASONS) -> pd.DataFrame:
    return pd.concat([prices(s) for s in seasons], ignore_index=True)
