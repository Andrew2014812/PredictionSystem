"""Team- and league-level statistics for the UI (tables, form, H2H).

All functions take an optional ``before`` date; with it only matches played
strictly before that date are used, which is how the match page shows the
state of both teams *before* kick-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..leagues import get_league

TEAM_STAT_COLUMNS = ["gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against",
                     "corners_for", "corners_against", "yellows", "reds"]


def _played(features: pd.DataFrame, before=None) -> pd.DataFrame:
    played = features.loc[features["played"].astype(bool)]
    if before is not None:
        played = played.loc[played["date"] < pd.Timestamp(before)]
    return played


def team_matches(features: pd.DataFrame, team: str, country: str | None = None,
                 before=None) -> pd.DataFrame:
    """Played matches of ``team`` from its own perspective, newest first."""
    played = _played(features, before)
    if country:
        played = played.loc[played["country"] == country]
    home = played.loc[played["home_team"] == team]
    away = played.loc[played["away_team"] == team]

    def view(df: pd.DataFrame, venue: str) -> pd.DataFrame:
        me, op = ("home", "away") if venue == "H" else ("away", "home")
        return pd.DataFrame({
            "match_id": df["match_id"], "date": df["date"], "league": df["league"],
            "season": df["season"], "venue": venue, "opponent": df[f"{op}_team"],
            "gf": df[f"{me}_goals"], "ga": df[f"{op}_goals"],
            "shots_for": df[f"{me}_shots"], "shots_against": df[f"{op}_shots"],
            "sot_for": df[f"{me}_sot"], "sot_against": df[f"{op}_sot"],
            "corners_for": df[f"{me}_corners"], "corners_against": df[f"{op}_corners"],
            "yellows": df[f"{me}_yellows"], "reds": df[f"{me}_reds"],
        })

    out = pd.concat([view(home, "H"), view(away, "A")], ignore_index=True)
    out["result"] = np.where(out["gf"] > out["ga"], "W", np.where(out["gf"] < out["ga"], "L", "D"))
    out["points"] = out["result"].map({"W": 3, "D": 1, "L": 0})
    return out.sort_values("date", ascending=False).reset_index(drop=True)


def form_summary(matches: pd.DataFrame) -> dict:
    """Averages over a set of team matches (e.g. the last N)."""
    if matches.empty:
        return {"games": 0}
    total = matches["gf"] + matches["ga"]
    out = {
        "games": len(matches),
        "wins": int((matches["result"] == "W").sum()),
        "draws": int((matches["result"] == "D").sum()),
        "losses": int((matches["result"] == "L").sum()),
        "points": int(matches["points"].sum()),
        "ppg": matches["points"].mean(),
        "btts_rate": ((matches["gf"] > 0) & (matches["ga"] > 0)).mean(),
        "over25_rate": (total > 2.5).mean(),
        "clean_sheet_rate": (matches["ga"] == 0).mean(),
        "form": "".join(matches["result"].head(5)),
    }
    for col in TEAM_STAT_COLUMNS:
        out[col] = matches[col].mean()
    out["gd"] = out["gf"] - out["ga"]
    return out


def league_table(features: pd.DataFrame, league: str, season: int, before=None) -> pd.DataFrame:
    """Standings of ``league`` in ``season`` (points, GD, GF, name)."""
    played = _played(features, before)
    played = played.loc[(played["league"] == league) & (played["season"] == season)]
    if played.empty:
        return pd.DataFrame()
    rows = []
    for team in sorted(set(played["home_team"]) | set(played["away_team"])):
        tm = team_matches(played, team)
        rows.append({
            "Team": team, "P": len(tm),
            "W": int((tm["result"] == "W").sum()), "D": int((tm["result"] == "D").sum()),
            "L": int((tm["result"] == "L").sum()),
            "GF": int(tm["gf"].sum()), "GA": int(tm["ga"].sum()),
            "GD": int(tm["gf"].sum() - tm["ga"].sum()), "Pts": int(tm["points"].sum()),
            "Form": "".join(tm["result"].head(5)),
            "xCorners": tm["corners_for"].mean(),
        })
    table = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True])
    table.insert(0, "Pos", range(1, len(table) + 1))
    return table.reset_index(drop=True)


def head_to_head(features: pd.DataFrame, home: str, away: str, country: str | None = None,
                 before=None, limit: int = 5) -> pd.DataFrame:
    played = _played(features, before)
    if country:
        played = played.loc[played["country"] == country]
    pair = played.loc[((played["home_team"] == home) & (played["away_team"] == away))
                      | ((played["home_team"] == away) & (played["away_team"] == home))]
    return pair.sort_values("date", ascending=False).head(limit)


def h2h_summary(meetings: pd.DataFrame, home: str) -> dict:
    if meetings.empty:
        return {"games": 0}
    gf = np.where(meetings["home_team"] == home, meetings["home_goals"], meetings["away_goals"])
    ga = np.where(meetings["home_team"] == home, meetings["away_goals"], meetings["home_goals"])
    total = meetings["home_goals"] + meetings["away_goals"]
    return {
        "games": len(meetings), "wins": int((gf > ga).sum()), "draws": int((gf == ga).sum()),
        "losses": int((gf < ga).sum()), "avg_goals": float(total.mean()),
        "btts_rate": float(((meetings["home_goals"] > 0) & (meetings["away_goals"] > 0)).mean()),
        "over25_rate": float((total > 2.5).mean()),
    }


def all_teams(features: pd.DataFrame) -> pd.DataFrame:
    """Every team with its latest league and country."""
    rows = pd.concat([
        features[["home_team", "league", "country", "date"]].rename(columns={"home_team": "team"}),
        features[["away_team", "league", "country", "date"]].rename(columns={"away_team": "team"}),
    ])
    latest = rows.sort_values("date").groupby(["country", "team"], as_index=False).last()
    latest["league_name"] = latest["league"].map(lambda c: get_league(c).name)
    return latest.sort_values("team").reset_index(drop=True)
