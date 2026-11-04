"""Real pre-match prices of several bookmakers from the raw Football-Data files (local_data/results/<code>/<div>.csv).

Football-Data publishes two snapshots per match:

* "opening" columns (B365H, PSH, MaxH, AvgH, ...) — collected on Friday afternoon for weekend matches and on
  Tuesday for midweek matches, i.e. before the prediction is made for most matches;
* "closing" columns (B365CH, PSCH, MaxCH, ...) — the last prices before kick-off.

Both are pre-match. Only bookmakers whose prices were quoted at the SAME snapshot are combined into a "best
named" price (Bet365, Bet&Win, Pinnacle, William Hill, 1XBet; Betfair sportsbook where present); exchanges (BFE*)
are excluded, as in production. "Max"/"Avg" are Football-Data's market maximum / average over a wider set of
books at the same snapshot — shown for information (a maximum over books a bettor may not hold).

The match id is built exactly as in src/data/cleaning.py (league, date, raw team names), so rows join the
feature table by ``match_id``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.cleaning import _parse_dates, make_match_id
from src.data.dataset import _read_raw_csv

from . import common as C

BOOKS_1X2 = {"b365": "B365", "bw": "BW", "ps": "PS", "wh": "WH", "1xb": "1XB", "bf": "BF", "max": "Max", "avg": "Avg"}
BOOKS_OU = {"b365": "B365", "ps": "P", "max": "Max", "avg": "Avg"}
BOOKS_AH = {"b365": "B365", "ps": "P", "max": "Max", "avg": "Avg"}
NAMED = ("b365", "bw", "ps", "wh", "1xb", "bf")       # real books (no exchange, no aggregate)


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw.columns:
        return pd.Series(np.nan, index=raw.index)
    v = pd.to_numeric(raw[col], errors="coerce")
    return v.where(v > 1.0)


def _cols_1x2(prefix: str, closing: bool) -> tuple[str, str, str]:
    c = "C" if closing else ""
    return f"{prefix}{c}H", f"{prefix}{c}D", f"{prefix}{c}A"


def _cols_ou(prefix: str, closing: bool) -> tuple[str, str]:
    c = "C" if closing else ""
    return f"{prefix}{c}>2.5", f"{prefix}{c}<2.5"


def _cols_ah(prefix: str, closing: bool) -> tuple[str, str]:
    c = "C" if closing else ""
    return f"{prefix}{c}AHH", f"{prefix}{c}AHA"


def load_raw(season: int) -> pd.DataFrame:
    folder = config.RESULTS_DIR / config.season_code(season)
    frames = []
    for path in sorted(folder.glob("*.csv")):
        raw = _read_raw_csv(path)
        if raw.empty or "Div" not in raw.columns:
            continue
        raw = raw.loc[raw["HomeTeam"].notna() & raw["AwayTeam"].notna() & raw["Date"].notna()]
        raw = raw.assign(_file=path.stem)
        frames.append(raw)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prices(season: int) -> pd.DataFrame:
    """One row per match: opening and closing prices of 1X2, O/U 2.5 and the Asian line (wide)."""
    raw = load_raw(season)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=raw.index)
    dates = _parse_dates(raw["Date"].astype(str).str.strip())
    out["match_id"] = [make_match_id(str(lg).strip(), d, str(h).strip(), str(a).strip()) if d == d else None
                       for lg, d, h, a in zip(raw["Div"], dates, raw["HomeTeam"], raw["AwayTeam"])]
    for closing in (False, True):
        suf = "_c" if closing else ""
        for key, pre in BOOKS_1X2.items():
            for sel, col in zip("HDA", _cols_1x2(pre, closing)):
                out[f"x_{sel}_{key}{suf}"] = _num(raw, col)
        for key, pre in BOOKS_OU.items():
            for sel, col in zip(("OVER", "UNDER"), _cols_ou(pre, closing)):
                out[f"ou_{sel}_{key}{suf}"] = _num(raw, col)
        line_col = "AHCh" if closing else "AHh"
        line = pd.to_numeric(raw[line_col], errors="coerce") if line_col in raw.columns else pd.Series(np.nan, index=raw.index)
        if not closing and "BbAHh" in raw.columns:
            line = line.fillna(pd.to_numeric(raw["BbAHh"], errors="coerce"))
        out[f"ah_line{suf}"] = line
        for key, pre in BOOKS_AH.items():
            for sel, col in zip("HA", _cols_ah(pre, closing)):
                out[f"ah_{sel}_{key}{suf}"] = _num(raw, col)
    out = out.loc[out["match_id"].notna()].drop_duplicates("match_id", keep="last")
    return out.reset_index(drop=True)


def audit() -> dict:
    """Which price columns really exist (share of non-empty values) per season and league."""
    rows = []
    for season in range(config.FIRST_SEASON, C.LIVE + 1):
        raw = load_raw(season)
        if raw.empty:
            continue
        checks = {"B365 1X2": "B365H", "Pinnacle 1X2": "PSH", "Max 1X2": "MaxH", "Avg 1X2": "AvgH",
                  "B365 O/U2.5": "B365>2.5", "Pinnacle O/U2.5": "P>2.5", "B365 AH": "B365AHH",
                  "Pinnacle AH": "PAHH", "AH line": "AHh", "B365 closing 1X2": "B365CH",
                  "Pinnacle closing 1X2": "PSCH", "B365 closing O/U": "B365C>2.5", "Pinnacle closing O/U": "PC>2.5",
                  "B365 closing AH": "B365CAHH", "Closing AH line": "AHCh", "BW 1X2": "BWH", "WH 1X2": "WHH",
                  "1XB 1X2": "1XBH", "BF 1X2": "BFH"}
        for lg, g in raw.groupby("_file"):
            row = {"season": season, "league": lg, "rows": int(len(g))}
            for name, col in checks.items():
                row[name] = float(pd.to_numeric(g[col], errors="coerce").notna().mean()) if col in g.columns else 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    by_season = df.groupby("season").apply(
        lambda g: pd.Series({c: float(np.average(g[c], weights=g["rows"])) for c in df.columns
                             if c not in ("season", "league", "rows")} | {"rows": int(g["rows"].sum())}),
        include_groups=False).reset_index()
    C.save_csv("fd_price_columns_by_league.csv", df)
    out = {"by_season": by_season.to_dict("records"),
           "leagues_without_pinnacle_closing": df.loc[df["Pinnacle closing 1X2"] < 0.5, ["season", "league"]]
           .to_dict("records")}
    C.save_json("fd_price_audit.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("fd")
    a = audit()
    print(pd.DataFrame(a["by_season"]).T)
