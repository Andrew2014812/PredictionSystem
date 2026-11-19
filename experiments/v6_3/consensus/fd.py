"""Football-Data multi-book prices, 2021/22–2026/27 (local_data/results/<code>/<div>.csv).

Football-Data stores two snapshots of every match for several bookmakers at once:

* opening ("open"): columns B365H, BWH, PSH, ... — one snapshot of all books (Friday afternoon for weekend matches,
  Tuesday for midweek matches; the exact time is not published);
* closing ("close"): B365CH, BWCH, PSCH, ... — the last prices before kick-off.

Within a snapshot every book was quoted at the same moment, so a consensus of one snapshot never mixes quotes of
different times. Markets: 1X2, total 2.5, and the Asian line of the snapshot (AHh / AHCh, half lines only: no
refund). ``Max`` / ``Avg`` (Football-Data's market maximum / average) and ``BFE`` (Betfair exchange) are kept as
references, not as books of the consensus.

The match id is built as in ``src/data/cleaning.py`` (league, date, raw team names) — the v6.1 / v6.2 loader.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src import config
from src.data.cleaning import _parse_dates, make_match_id
from src.data.dataset import _read_raw_csv

SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
REFERENCE = {"Max", "Avg", "BFE"}
NAMES = {"B365": "bet365", "PS": "pinnacle", "P": "pinnacle", "WH": "williamhill", "BW": "bwin", "IW": "interwetten",
         "VC": "vcbet", "BV": "betvictor", "1XB": "1xbet", "BF": "betfair_sb", "BFD": "betfred", "BMGM": "betmgm",
         "CL": "coral", "LB": "ladbrokes", "PP": "paddypower", "SKB": "skybet", "BFE": "betfair_exchange",
         "Max": "max", "Avg": "avg"}


def load_raw(season: int) -> pd.DataFrame:
    folder = config.RESULTS_DIR / config.season_code(season)
    frames = []
    for path in sorted(folder.glob("*.csv")):
        raw = _read_raw_csv(path)
        if raw.empty or "Div" not in raw.columns:
            continue
        frames.append(raw.loc[raw["HomeTeam"].notna() & raw["AwayTeam"].notna() & raw["Date"].notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(raw, col):
    if col not in raw.columns:
        return pd.Series(np.nan, index=raw.index)
    v = pd.to_numeric(raw[col], errors="coerce")
    return v.where(v > 1.0)


def _prefixes(cols, suffixes) -> set[str]:
    out = set()
    for c in cols:
        for s in suffixes[:1]:
            if c.endswith(s):
                p = c[: -len(s)]
                if p and all(p + x in cols for x in suffixes):
                    out.add(p)
    return out


def _split(prefixes: set[str]) -> dict[str, tuple[str, str]]:
    """prefix -> (book code, snapshot)."""
    out = {}
    for p in prefixes:
        if p.endswith("C") and p[:-1] in prefixes:
            out[p] = (p[:-1], "close")
        elif p not in out:
            out[p] = (p, "open")
    return out


def prices(season: int) -> pd.DataFrame:
    """Long table: match_id, season, league, date, hg, ag, market, snapshot, book, line, o1, o2, o3 (side prices:
    1X2 H/D/A; total OVER/UNDER; Asian home/away of the snapshot's line)."""
    path = C.V63 / "consensus" / "cache" / f"fd_books_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = load_raw(season)
    cols = set(raw.columns)
    dates = _parse_dates(raw["Date"].astype(str).str.strip())
    base = pd.DataFrame({
        "match_id": [make_match_id(str(lg).strip(), d, str(h).strip(), str(a).strip()) if d == d else None
                     for lg, d, h, a in zip(raw["Div"], dates, raw["HomeTeam"], raw["AwayTeam"])],
        "season": season, "league": raw["Div"].astype(str).str.strip(), "date": dates,
        "hg": pd.to_numeric(raw.get("FTHG"), errors="coerce"), "ag": pd.to_numeric(raw.get("FTAG"), errors="coerce")})
    frames = []
    for market, suf in (("1X2", ("H", "D", "A")), ("TOTAL_2.5", (">2.5", "<2.5")), ("AH", ("AHH", "AHA"))):
        for p, (book, snap) in _split(_prefixes(cols, suf)).items():
            if market == "AH" and snap == "close" and not p.endswith("C"):
                continue
            if market == "AH":           # closing Asian columns are <book>CAHH, so the prefix ends with C as well
                pass
            o = [_num(raw, p + s) for s in suf]
            f = base.copy()
            f["market"], f["snapshot"], f["book"] = market, snap, NAMES.get(book, book.lower())
            f["o1"], f["o2"] = o[0].to_numpy(), o[1].to_numpy()
            f["o3"] = o[2].to_numpy() if len(o) > 2 else np.nan
            if market == "AH":
                f["line"] = pd.to_numeric(raw.get("AHCh" if snap == "close" else "AHh"), errors="coerce").to_numpy()
            else:
                f["line"] = np.nan
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    need = ["o1", "o2"] + (["o3"] if True else [])
    is3 = out["market"] == "1X2"
    ok = out[["o1", "o2"]].notna().all(1) & (~is3 | out["o3"].notna()) & out["match_id"].notna()
    ah = out["market"] == "AH"
    half = np.isfinite(out["line"]) & (np.abs((out["line"] * 2) % 2 - 1) < 1e-9)
    ok &= ~ah | half
    out = out.loc[ok & out["hg"].notna()].drop_duplicates(["match_id", "market", "snapshot", "book"], keep="last")
    out = out.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def all_seasons() -> pd.DataFrame:
    return pd.concat([prices(s) for s in SEASONS], ignore_index=True)


def outcome(df: pd.DataFrame) -> np.ndarray:
    """Index of the winning side (1X2: 0 H / 1 D / 2 A; total: 0 over / 1 under; Asian: 0 home / 1 away)."""
    hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
    y = np.full(len(df), -1)
    m = df["market"].to_numpy()
    x = m == "1X2"
    y[x] = np.where(hg[x] > ag[x], 0, np.where(hg[x] == ag[x], 1, 2))
    t = m == "TOTAL_2.5"
    y[t] = np.where(hg[t] + ag[t] > 2.5, 0, 1)
    a = m == "AH"
    y[a] = np.where(hg[a] + df["line"].to_numpy(float)[a] - ag[a] > 0, 0, 1)
    return y


def candidate_market(df: pd.DataFrame) -> pd.Series:
    """FD market -> production market key (Asian: HANDICAP_<line> from the home side)."""
    from src.markets.markets import handicap_market
    return pd.Series(np.where(df["market"] == "AH", [handicap_market(x) if x == x else None for x in df["line"]],
                              df["market"]), index=df.index)


__all__ = ["prices", "all_seasons", "outcome", "candidate_market", "re"]
