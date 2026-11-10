"""Why a past match still has unsettled predictions.

A prediction stays UPCOMING until its match has the data its market needs. For a match whose date has
passed this explains, per match, what every source knows and why the system still waits:

* the score is known but a corners market waits for Football-Data's statistics (API-Football gives goals only);
* API-Football reported the match as postponed, cancelled, abandoned...;
* the same pairing is scheduled or played on another date (moved match);
* no source has a result yet.

Nothing here changes data; the pipeline writes the table to ``api/diagnostics/waiting.csv``.
"""
from __future__ import annotations

import pandas as pd

from .. import config

NOT_PLAYED = {"PST": "postponed", "CANC": "cancelled", "ABD": "abandoned", "SUSP": "suspended",
              "INT": "interrupted", "AWD": "awarded", "WO": "walkover", "TBD": "time to be defined"}
STATISTICS = "result known, waiting for corner statistics (Football-Data)"
SETTLES_NEXT = "result known, settled at the next run"
NO_RESULT = "no result from any source yet"
COLUMNS = ["date", "league", "home_team", "away_team", "match_id", "waiting_predictions", "waiting_markets",
           "reason", "api_football_status", "api_football_result", "football_data_result", "moved_to",
           "void_after"]


def _pairs(frame: pd.DataFrame) -> dict:
    return {k: g for k, g in frame.groupby(["league", "home_team", "away_team"])} if not frame.empty else {}


def waiting_matches(picks: pd.DataFrame, features: pd.DataFrame, fd_results: pd.DataFrame | None = None,
                    live: pd.DataFrame | None = None, not_played: pd.DataFrame | None = None,
                    today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    if picks.empty:
        return pd.DataFrame(columns=COLUMNS)
    pending = picks.loc[(picks["status"] == "UPCOMING")
                        & (pd.to_datetime(picks["date"]) < today - pd.Timedelta(days=1))]
    if pending.empty:
        return pd.DataFrame(columns=COLUMNS)
    feats = features.set_index("match_id")
    fd_played = set(fd_results.loc[fd_results["played"].astype(bool), "match_id"]) \
        if fd_results is not None and not fd_results.empty else set()
    api_played = set(live.loc[live["played"].astype(bool), "match_id"]) \
        if live is not None and not live.empty else set()
    by_pair = _pairs(features[["league", "home_team", "away_team", "date", "played", "match_id"]])
    statuses = not_played if not_played is not None and not not_played.empty else pd.DataFrame(
        columns=["league", "date", "home_team", "away_team", "status"])
    statuses = statuses.assign(date=pd.to_datetime(statuses["date"]))

    rows = []
    for match_id, g in pending.groupby("match_id"):
        first = g.iloc[0]
        date = pd.Timestamp(first["date"]).normalize()
        row = feats.loc[match_id] if match_id in feats.index else None
        scored = row is not None and bool(row["played"]) and pd.notna(row.get("home_goals"))
        markets = sorted(set(g["market"]))
        api_status = statuses.loc[(statuses["league"] == first["league"]) & (statuses["home_team"] == first["home_team"])
                                  & (statuses["away_team"] == first["away_team"])
                                  & ((statuses["date"] - date).abs() <= pd.Timedelta(days=3)), "status"]
        api_status = str(api_status.iloc[-1]) if len(api_status) else ""
        others = by_pair.get((first["league"], first["home_team"], first["away_team"]))
        moved = ""
        if others is not None:
            later = others.loc[(others["match_id"] != match_id)
                               & ((others["date"] - date).abs() <= pd.Timedelta(days=config.VOID_AFTER_DAYS * 3))]
            if len(later):
                moved = str(pd.Timestamp(later["date"].iloc[-1]).date())
        if scored:
            reason = STATISTICS if all(m.startswith("CORNERS_") for m in markets) else SETTLES_NEXT
        elif api_status in NOT_PLAYED:
            reason = f"{NOT_PLAYED[api_status]} (API-Football {api_status})"
        elif moved:
            reason = f"same pairing on {moved} (moved match)"
        else:
            reason = NO_RESULT
        rows.append({"date": str(date.date()), "league": first["league"], "home_team": first["home_team"],
                     "away_team": first["away_team"], "match_id": match_id,
                     "waiting_predictions": int(len(g)), "waiting_markets": ", ".join(markets),
                     "reason": reason, "api_football_status": api_status,
                     "api_football_result": match_id in api_played, "football_data_result": match_id in fd_played,
                     "moved_to": moved,
                     "void_after": str((date + pd.Timedelta(days=config.VOID_AFTER_DAYS)).date())})
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["date", "league"]).reset_index(drop=True)


def reason_counts(table: pd.DataFrame) -> dict[str, int]:
    """Short counts for the run summary: matches without a result vs. matches waiting only for statistics."""
    if table.empty:
        return {}
    key = table["reason"].map(lambda r: "waiting for corner statistics" if r == STATISTICS
                              else "no result yet" if r == NO_RESULT
                              else "postponed / cancelled" if "API-Football" in r
                              else "moved" if "moved" in r else "settle next run")
    return key.value_counts().to_dict()
