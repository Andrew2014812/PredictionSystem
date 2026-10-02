"""Normalising raw Football-Data rows into one canonical match schema.

Canonical columns (one row = one match):

    match_id, league, country, season, date, time, kickoff,
    home_team, away_team, played,
    home_goals, away_goals, result,
    home_shots, away_shots, home_sot, away_sot, home_fouls, away_fouls,
    home_corners, away_corners, home_yellows, away_yellows,
    home_reds, away_reds,
    odds_home, odds_draw, odds_away, odds_over25, odds_under25

Odds are kept as market information only; they never enter the feature
matrix unless ``config.USE_ODDS_FEATURES`` is switched on.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config
from ..leagues import league_country

STAT_COLUMNS = {
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HF": "home_fouls", "AF": "away_fouls",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellows", "AY": "away_yellows",
    "HR": "home_reds", "AR": "away_reds",
}

# Each canonical odds column takes the first source column that is present
# (market average first, then the older Betbrain average, then Bet365, then
# the market maximum).
ODDS_SOURCES = {
    "odds_home": ["AvgH", "BbAvH", "B365H", "MaxH", "BbMxH"],
    "odds_draw": ["AvgD", "BbAvD", "B365D", "MaxD", "BbMxD"],
    "odds_away": ["AvgA", "BbAvA", "B365A", "MaxA", "BbMxA"],
    "odds_over25": ["Avg>2.5", "BbAv>2.5", "B365>2.5", "Max>2.5", "P>2.5"],
    "odds_under25": ["Avg<2.5", "BbAv<2.5", "B365<2.5", "Max<2.5", "P<2.5"],
}

NUMERIC_COLUMNS = [c for c in STAT_COLUMNS.values() if c != "result"] + list(ODDS_SOURCES)

CANONICAL_COLUMNS = [
    "match_id", "league", "country", "season", "date", "time", "kickoff",
    "home_team", "away_team", "played", "result", *NUMERIC_COLUMNS,
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def make_match_id(league: str, day: pd.Timestamp, home: str, away: str) -> str:
    """Stable identity of a match: league + date + home team + away team.

    Kick-off time is deliberately not part of the id: Football-Data
    fixtures and results sometimes disagree on the time of the same
    match, and a club never plays two league matches on one day.
    """
    return f"{league}-{day:%Y%m%d}-{slug(home)}-{slug(away)}"


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def _parse_dates(values: pd.Series) -> pd.Series:
    # Football-Data uses dd/mm/yy in older files and dd/mm/yyyy in newer.
    parsed = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(values, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(short)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw Football-Data frame (results or fixtures) to canonical form."""
    if raw.empty or "Div" not in raw.columns:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    raw = raw.loc[raw["HomeTeam"].notna() & raw["AwayTeam"].notna() & raw["Date"].notna()]
    df = pd.DataFrame(index=raw.index)
    df["league"] = raw["Div"].astype(str).str.strip()
    df["date"] = _parse_dates(raw["Date"].astype(str).str.strip())
    df["time"] = raw["Time"].astype(str).str.strip() if "Time" in raw.columns else ""
    df.loc[~df["time"].str.match(r"^\d{1,2}:\d{2}$"), "time"] = ""
    df["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    df["away_team"] = raw["AwayTeam"].astype(str).str.strip()

    for src, dst in STAT_COLUMNS.items():
        if src in raw.columns:
            df[dst] = raw[src] if dst == "result" else pd.to_numeric(raw[src], errors="coerce")
        else:
            df[dst] = np.nan
    for dst, candidates in ODDS_SOURCES.items():
        df[dst] = _pick_first(raw, candidates)
    # Odds of 1.0 or less are data errors, not prices.
    for col in ODDS_SOURCES:
        df.loc[df[col] <= 1.0, col] = np.nan

    df = df.loc[df["date"].notna()].copy()
    df["played"] = df["home_goals"].notna() & df["away_goals"].notna()
    goal_diff = df["home_goals"] - df["away_goals"]
    df["result"] = np.where(~df["played"], None,
                            np.where(goal_diff > 0, "H", np.where(goal_diff < 0, "A", "D")))
    # A played match must have a result; stats of an unplayed one must be empty.
    stat_cols = [c for c in STAT_COLUMNS.values() if c != "result"]
    df.loc[~df["played"], stat_cols] = np.nan

    df["country"] = df["league"].map(league_country)
    df["season"] = df["date"].map(config.season_of).astype(int)
    kickoff_time = df["time"].where(df["time"] != "", "00:00")
    df["kickoff"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + kickoff_time,
                                   errors="coerce")
    df["match_id"] = [make_match_id(lg, d, h, a) for lg, d, h, a in
                      zip(df["league"], df["date"], df["home_team"], df["away_team"])]
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove true duplicates only.

    Rows describe the same match when they share ``match_id`` (league,
    date, home, away). Among duplicates a played row wins over a fixture,
    then the most complete row wins. Matches between the same teams on
    other dates (another season, the second meeting of a season) are kept.
    """
    if df.empty:
        return df
    df = df.assign(_played=df["played"].astype(int),
                   _filled=df[NUMERIC_COLUMNS].notna().sum(axis=1))
    df = df.sort_values(["match_id", "_played", "_filled"], ascending=[True, False, False])
    df = df.drop_duplicates("match_id", keep="first")
    return df.drop(columns=["_played", "_filled"]).reset_index(drop=True)


def drop_stale_fixtures(df: pd.DataFrame, window_days: int = 60) -> pd.DataFrame:
    """Drop unplayed rows that a played row of the same pairing replaced.

    A fixture whose result was then published under a different date
    (rescheduled match) would otherwise survive as a phantom upcoming match.
    Only played rows of the same league and pairing within ``window_days``
    count, so a genuine later meeting of the same teams is kept.
    """
    played = df.loc[df["played"], ["league", "home_team", "away_team", "date"]]
    pending = df.loc[~df["played"], ["league", "home_team", "away_team", "date"]].reset_index()
    if played.empty or pending.empty:
        return df
    pairs = pending.merge(played, on=["league", "home_team", "away_team"], suffixes=("", "_res"))
    close = (pairs["date_res"] - pairs["date"]).dt.days.abs() <= window_days
    stale = set(pairs.loc[close, "index"])
    return df.drop(index=list(stale)).reset_index(drop=True)
