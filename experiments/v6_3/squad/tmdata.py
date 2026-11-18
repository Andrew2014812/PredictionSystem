"""Player data of the squad study: the open Transfermarkt dataset (parsed by ``parse_tm.py``) and the mapping of its
clubs onto FootPredict's (Football-Data) team names.

Source: github.com/dcaribou/transfermarkt-datasets (CC0 licence, public R2 bucket, no login), frozen on 2026-07-06:
games up to 2026-07-06, appearances up to 2026-06-28, market values up to 2026-06-12. Covers the top divisions of
11 FootPredict leagues (E0 SP1 I1 D1 F1 N1 P1 B1 T1 G1 SC0) with every club game (league, cups, Europe), starting
line-ups, substitutes, positions, minutes, goals, assists, cards and dated market values. Nothing of 2026/27.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C

TM = C.V63 / "squad" / "cache" / "tm"
LEAGUE_MAP = {"E0": "GB1", "SP1": "ES1", "I1": "IT1", "D1": "L1", "F1": "FR1", "N1": "NL1", "P1": "PO1",
              "B1": "BE1", "T1": "TR1", "G1": "GR1", "SC0": "SC1"}
NATIONAL = {"EURO", "FIWC", "COPA", "AFCN", "AFAC"}

POS_GROUP = {"Goalkeeper": "GK", "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF", "Sweeper": "DEF",
             "Defender": "DEF", "Defensive Midfield": "MID", "Central Midfield": "MID", "Attacking Midfield": "MID",
             "Left Midfield": "MID", "Right Midfield": "MID", "Midfield": "MID", "midfield": "MID",
             "Centre-Forward": "ATT", "Second Striker": "ATT", "Left Winger": "ATT", "Right Winger": "ATT",
             "Attack": "ATT"}
WIDE = {"Left-Back", "Right-Back", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger"}


def load(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(TM / f"{name}.parquet", columns=columns)


def fd_matches(columns=("match_id", "league", "season", "date", "home_team", "away_team", "home_goals",
                        "away_goals", "played")) -> pd.DataFrame:
    f = pd.read_parquet(C.DATA_FILES["features"], columns=list(columns))
    f["date"] = pd.to_datetime(f["date"])
    return f


def team_mapping(fd: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FootPredict team name -> Transfermarkt club_id, learnt from matches that agree on league, date (±1 day) and
    score. Every (league, team) is assigned the club it co-occurs with most often; ties / weak evidence are
    reported, never guessed."""
    rows, report = [], {}
    played = fd.loc[fd["played"].astype(bool) & fd["league"].isin(LEAGUE_MAP) & (fd["season"] <= 2025)]
    for lg, comp in LEAGUE_MAP.items():
        a = played.loc[played["league"] == lg, ["match_id", "date", "home_team", "away_team", "home_goals",
                                                 "away_goals"]]
        b = games.loc[games["competition_id"] == comp, ["date", "home_club_id", "away_club_id", "home_club_goals",
                                                        "away_club_goals", "home_club_name", "away_club_name"]]
        pairs = []
        for shift in (-1, 0, 1):
            bb = b.assign(date=b["date"] + pd.Timedelta(days=shift))
            j = a.merge(bb, left_on=["date", "home_goals", "away_goals"],
                        right_on=["date", "home_club_goals", "away_club_goals"])
            pairs.append(j[["home_team", "home_club_id", "home_club_name"]].set_axis(["team", "club_id", "club"], axis=1))
            pairs.append(j[["away_team", "away_club_id", "away_club_name"]].set_axis(["team", "club_id", "club"], axis=1))
        p = pd.concat(pairs).groupby(["team", "club_id", "club"]).size().rename("n").reset_index()
        tot = p.groupby("team")["n"].transform("sum")
        p["share"] = p["n"] / tot
        best = p.sort_values(["team", "n"], ascending=[True, False]).drop_duplicates("team")
        best["league"] = lg
        rows.append(best)
        teams = set(a["home_team"]) | set(a["away_team"])
        report[lg] = {"teams": len(teams), "mapped": int(len(best)), "weak": best.loc[best["share"] < 0.5, "team"].tolist(),
                      "unmapped": sorted(teams - set(best["team"]))}
    m = pd.concat(rows, ignore_index=True)
    m = m.loc[m["share"] >= 0.5]
    dup = m.loc[m.duplicated(["league", "club_id"], keep=False)]
    report["duplicate_clubs"] = dup[["league", "team", "club"]].to_dict("records")
    return m[["league", "team", "club_id", "club", "n", "share"]], report


__all__ = ["load", "fd_matches", "team_mapping", "LEAGUE_MAP", "POS_GROUP", "WIDE", "NATIONAL", "np", "Path"]
